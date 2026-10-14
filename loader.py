import torch
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data import random_split
from torchvision import transforms
import numpy as np


KAGGLE_DATASET  = "figolm10/the-birds"
LOCAL_DATA_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "the-birds")
IMG_EXTENSIONS  = (".jpg", ".jpeg", ".png")


def find_dataset_root(path):
    """Busca la carpeta cuyos subdirectorios (uno por clase) contienen las imágenes."""

    for current, dirs, _ in os.walk(path):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        has_images = [
            any(f.lower().endswith(IMG_EXTENSIONS) for f in os.listdir(os.path.join(current, d)))
            for d in dirs
        ]
        if len(dirs) > 1 and all(has_images):
            return current

    raise FileNotFoundError(f"No se encontró una carpeta con subcarpetas de imágenes en {path}")


def get_data_dir(local_dir=LOCAL_DATA_DIR):
    """Retorna la ruta del dataset; si no existe localmente lo descarga de Kaggle con kagglehub."""

    if os.path.isdir(local_dir):
        return find_dataset_root(local_dir)

    import kagglehub
    path = kagglehub.dataset_download(KAGGLE_DATASET)

    return find_dataset_root(path)


class BirdsDataset(Dataset):

    def __init__(self, root_dir, transform=None, MLP=False):

        self.root_dir       = root_dir
        self.transform      = transform
        self.MLP            = MLP

        # Obtener nombres de las clases (solo carpetas, se ignoran archivos ocultos como .DS_Store)
        self.classes = sorted(
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith(".")
        )

        # Asignar un número a cada clase
        self.class_to_idx = {
            clase: idx
            for idx, clase in enumerate(self.classes)
        }

        # Guardar rutas y etiquetas
        self.samples = []

        for clase in self.classes:

            class_dir = os.path.join(root_dir, clase)

            for filename in sorted(os.listdir(class_dir)):

                if not filename.lower().endswith(IMG_EXTENSIONS):
                    continue

                image_path  = os.path.join(class_dir, filename)
                label       = self.class_to_idx[clase]

                self.samples.append((image_path, label))

        self.targets = [label for _, label in self.samples]

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        image_path, label = self.samples[idx]

        # Cargar imagen
        image = Image.open(image_path).convert("RGB")

        # Aplicar transformaciones (Resize -> ToTensor -> Normalize)
        if self.transform:
            image = self.transform(image)

        # Aplanar la imagen si MLP es True
        if self.MLP:
            image = image.flatten()

        return image, label


def compute_mean_std(dataset, indices, batch_size=64):
    """Media y desviación estándar por canal calculadas solo sobre los índices dados (train)."""

    loader      = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False)
    n_pixels    = 0
    channel_sum = torch.zeros(3)
    channel_sq  = torch.zeros(3)

    for images, _ in loader:
        n_pixels    += images.size(0) * images.size(2) * images.size(3)
        channel_sum += images.sum(dim=[0, 2, 3])
        channel_sq  += (images ** 2).sum(dim=[0, 2, 3])

    mean = channel_sum / n_pixels
    std  = torch.sqrt(channel_sq / n_pixels - mean ** 2)

    return mean.tolist(), std.tolist()


class Loader():

    def __init__(self, root_dir, img_size=128, splits=(0.70, 0.15, 0.15), seed=42, MLP=False):

        train_frac, val_frac, test_frac = splits

        self.root_dir   = root_dir
        self.img_size   = img_size

        # Dataset sin transformaciones, solo para conocer etiquetas y clases
        base_dataset    = BirdsDataset(root_dir)
        self.classes    = base_dataset.classes
        indices         = np.arange(len(base_dataset))

        # Tamaños
        train_size  = int(train_frac * len(base_dataset))
        val_size    = int(val_frac * len(base_dataset))
        test_size   = len(base_dataset) - train_size - val_size

        # División aleatoria (con semilla para que sea reproducible)
        train_split, val_split, test_split = random_split(
            indices,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(seed),
        )
        self.train_idx  = np.array(train_split.indices)
        self.val_idx    = np.array(val_split.indices)
        self.test_idx   = np.array(test_split.indices)

        # Media y desviación estándar calculadas solo con el conjunto de entrenamiento
        raw_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        self.mean, self.std = compute_mean_std(BirdsDataset(root_dir, transform=raw_transform), self.train_idx)

        # Transformaciones finales (sin Data Augmentation)
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])

        # Crear datasets
        self.dataset        = BirdsDataset(root_dir, transform=self.transform, MLP=MLP)
        self.train_dataset  = Subset(self.dataset, self.train_idx)
        self.val_dataset    = Subset(self.dataset, self.val_idx)
        self.test_dataset   = Subset(self.dataset, self.test_idx)

    def split_counts(self):
        """Cantidad de imágenes por clase en cada conjunto."""

        targets = np.array(self.dataset.targets)
        counts  = {}

        for name, idx in [("train", self.train_idx), ("val", self.val_idx), ("test", self.test_idx)]:
            counts[name] = {
                clase: int((targets[idx] == i).sum())
                for i, clase in enumerate(self.classes)
            }

        return counts

    def train_loader(self, batch_size=32, shuffle=True, num_workers=0):

        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

    def val_loader(self, batch_size=32, shuffle=False, num_workers=0):

        return DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

    def test_loader(self, batch_size=32, shuffle=False, num_workers=0):

        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )


if __name__ == "__main__":

    data_dir = get_data_dir()
    loader   = Loader(root_dir=data_dir)

    print(f"Ruta del dataset: {data_dir}")
    print(f"Clases ({len(loader.classes)}): {loader.classes}")
    print(f"Train: {len(loader.train_dataset)} | Val: {len(loader.val_dataset)} | Test: {len(loader.test_dataset)}")
    print(f"Mean: {loader.mean} | Std: {loader.std}")

    images, labels = next(iter(loader.train_loader()))
    print(f"Batch de imágenes: {list(images.shape)} | Batch de etiquetas: {list(labels.shape)}")
