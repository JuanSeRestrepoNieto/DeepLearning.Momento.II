import torch
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from torchvision import transforms
import numpy as np



class CIFAR10Dataset(Dataset):

    def __init__(self, root_dir, transform=None, MLP=False):

        self.root_dir       = root_dir
        self.transform      = transform
        self.MLP            = MLP

        # Obtener nombres de las clases
        self.classes = sorted(os.listdir(root_dir))
        self.classes.remove('.DS_Store')  # Eliminar archivo oculto si existe

        # Asignar un número a cada clase
        self.class_to_idx = {
            clase: idx
            for idx, clase in enumerate(self.classes)
        }

        # Guardar rutas y etiquetas
        self.samples = []

        for clase in self.classes:

            class_dir = os.path.join(root_dir, clase)

            for filename in os.listdir(class_dir):

                image_path  = os.path.join(class_dir, filename)
                label       = self.class_to_idx[clase]

                self.samples.append((image_path, label))

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        image_path, label = self.samples[idx]

        # Cargar imagen
        image = Image.open(image_path).convert("RGB")

        image = np.array(image)                     # Convertir a array de numpy
        image = image.astype("float32") / 255.0     # Normalizar la imagen

        # Aplicar transformaciones
        if self.transform:
            image = self.transform(image)

         # Aplanar la imagen si MLP es True
        if self.MLP:
            image = image.flatten()

        return image, label


class Loader():

    def __init__(self, root_dir, transform=None, MLP=False):

        self.path_train = os.path.join(root_dir, "train")
        self.path_test  = os.path.join(root_dir, "test")

        if transform is None:
            transform = transforms.ToTensor()

        # Crear datasets
        self.train_dataset = CIFAR10Dataset(self.path_train, transform=transform, MLP=MLP)
        self.test_dataset  = CIFAR10Dataset(self.path_test, transform=transform, MLP=MLP)

        # Tamaños
        train_size = int(0.8 * len(self.train_dataset))
        val_size = len(self.train_dataset) - train_size

        # División
        self.train_dataset, self.val_dataset = random_split(
            self.train_dataset,
            [train_size, val_size]
        )

    def train_loader(self, batch_size=32, shuffle=True):

        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
        )

    def val_loader(self, batch_size=32, shuffle=True):

        return DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
        )

    def test_loader(self, batch_size=32, shuffle=False):

        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
        )

if __name__ == "__main__":
    # Example usage of the loader module
    # Load a model or data here
    print("Loader module executed.")

