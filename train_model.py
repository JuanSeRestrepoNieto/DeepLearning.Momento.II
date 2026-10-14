from loader import Loader, get_data_dir
from model_cnn import CNNModel
from model_mlp import MLPModel
import argparse
import copy
import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt


CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")


def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(model, loader, criterion, optimizer, device):

    # Establecer el modelo en modo de entrenamiento
    model.train()
    running_loss    = 0.0
    correct         = 0
    total           = 0

    for images, labels in loader:

        # Mover al dispositivo
        images  = images.to(torch.float32).to(device)
        labels  = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss    = criterion(outputs, labels)

        # Backward pass y optimización
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Predicciones
        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total   += labels.size(0)

    return running_loss / len(loader), 100 * correct / total


def evaluate(model, loader, criterion, device):

    # Modo evaluación: desactiva Dropout y usa las estadísticas fijas de BatchNorm
    model.eval()
    running_loss    = 0.0
    correct         = 0
    total           = 0

    with torch.no_grad():

        for images, targets in loader:

            images  = images.to(torch.float32).to(device)
            targets = targets.to(device)

            # Forward propagation
            logits = model(images)

            # Calcular pérdida
            loss = criterion(logits, targets)
            running_loss += loss.item()

            # Predicciones
            predictions = logits.argmax(dim=1)

            correct += (predictions == targets).sum().item()
            total   += targets.size(0)

    return running_loss / len(loader), 100 * correct / total


def fit(model, train_loader, val_loader, criterion, optimizer, device, num_epochs,
        checkpoint_path=None, verbose=True):
    """Entrena el modelo y conserva los pesos de la época con menor Validation Loss."""

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    best_val_loss   = float("inf")
    best_epoch      = 0
    best_state      = None

    for epoch in range(num_epochs):

        train_loss, train_acc   = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc       = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss   = val_loss
            best_epoch      = epoch + 1
            best_state      = copy.deepcopy(model.state_dict())

        if verbose:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}] "
                f"Train Loss: {train_loss:.4f} "
                f"Train Acc: {train_acc:.2f}% "
                f"Val Loss: {val_loss:.4f} "
                f"Val Acc: {val_acc:.2f}%"
                + (" *" if best_epoch == epoch + 1 else "")
            )

    history["best_epoch"]       = best_epoch
    history["best_val_loss"]    = best_val_loss

    if checkpoint_path is not None:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(best_state, checkpoint_path)

    # Dejar el modelo con los pesos de la mejor época
    model.load_state_dict(best_state)

    return history


def predict(model, loader, device):
    """Retorna etiquetas reales, predicciones y probabilidades (softmax)."""

    model.eval()
    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():

        for images, targets in loader:

            logits  = model(images.to(torch.float32).to(device))
            probs   = torch.softmax(logits, dim=1).cpu()

            y_true.append(targets)
            y_pred.append(probs.argmax(dim=1))
            y_prob.append(probs)

    return (
        torch.cat(y_true).numpy(),
        torch.cat(y_pred).numpy(),
        torch.cat(y_prob).numpy(),
    )


def plot_history(history, path=None, show=True):

    epochs = range(1, len(history["train_loss"]) + 1)
    best   = history.get("best_epoch")

    fig = plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    if best:
        plt.axvline(best, color="gray", linestyle="--", label=f"Mejor época ({best})")
    plt.title("Training Loss vs. Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Val Accuracy")
    if best:
        plt.axvline(best, color="gray", linestyle="--", label=f"Mejor época ({best})")
    plt.title("Training Accuracy vs. Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    plt.tight_layout()

    # Guardar antes de mostrar (plt.show() limpia la figura)
    if path:
        fig.savefig(path, dpi=120)
    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="CNN", choices=["CNN", "MLP"])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img-size", type=int, default=128)
    args = parser.parse_args()

    set_seed(42)
    device = get_device()

    # Crear un objeto Loader
    is_mlp = args.model == "MLP"
    loader = Loader(root_dir=get_data_dir(), img_size=args.img_size, MLP=is_mlp)
    num_classes = len(loader.classes)

    # Crear un modelo según el tipo especificado
    if args.model == "CNN":
        model = CNNModel(num_classes=num_classes, img_size=args.img_size)
    else:
        model = MLPModel(args.img_size * args.img_size * 3, num_classes)

    # Mover el modelo al dispositivo
    model.to(device)

    # Obtener DataLoaders
    train_loader = loader.train_loader(batch_size=args.batch_size)
    val_loader   = loader.val_loader(batch_size=args.batch_size)
    test_loader  = loader.test_loader(batch_size=args.batch_size)

    # Mostrar cantidad de datos en cada conjunto
    print(f"Dispositivo: {device}")
    print(f"Tamaño del conjunto de entrenamiento: {len(train_loader.dataset)}")
    print(f"Tamaño del conjunto de validación: {len(val_loader.dataset)}")
    print(f"Tamaño del conjunto de prueba: {len(test_loader.dataset)}")

    # Compilar el modelo
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    history = fit(
        model, train_loader, val_loader, criterion, optimizer, device, args.epochs,
        checkpoint_path=os.path.join(CHECKPOINT_DIR, f"best_{args.model.lower()}.pt"),
    )
    print(f"Mejor época: {history['best_epoch']} (Val Loss: {history['best_val_loss']:.4f})")

    # Test del modelo (con los pesos de la mejor época)
    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} Test Accuracy: {test_accuracy:.2f}%")

    # Graficar pérdidas y precisiones
    plot_history(history, path="loss_accuracy_plot.png", show=False)
