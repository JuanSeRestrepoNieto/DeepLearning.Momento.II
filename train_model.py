from loader import Loader
from model_cnn import CNNModel
from model_mlp import MLPModel
import torch
import matplotlib.pyplot as plt

## Dictionary/enums with CNN and MLP


path_data           = "/Users/kevin/Library/CloudStorage/GoogleDrive-kevinosornocastillo@gmail.com/Mi unidad/07_Docencia/01_ITM/01_Subjects/05_IntroduccionDeepLearning/02_Presentaciones/05_CNN/00_code/cifar10"
model_type          = "CNN"     # Cambiar a "MLP" para usar el modelo MLP
output_size         = 10        # Número de clases
num_epochs          = 15        # Numero de épocas de entrenamiento
batch_size          = 128        # Tamaño del batch
device              = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# Crear un objeto Loader
loader = Loader(root_dir=path_data, transform=None, MLP=False)

# Crear un modelo según el tipo especificado
if model_type == "CNN":
    model       = CNNModel(num_classes=10)

elif model_type == "MLP":
    input_size  = 32 * 32 * 3  # Tamaño de la imagen
    model       = MLPModel(input_size, output_size)

# Mover el modelo al dispositivo
model.to(device)


# Obtener DataLoaders
train_loader = loader.train_loader(batch_size=batch_size)
val_loader   = loader.val_loader(batch_size=batch_size)
test_loader  = loader.test_loader(batch_size=batch_size)

# Mostrar cantidad de datos en cada conjunto
print(f'Tamaño del conjunto de entrenamiento: {len(train_loader.dataset)}')
print(f'Tamaño del conjunto de validación: {len(val_loader.dataset)}')
print(f'Tamaño del conjunto de prueba: {len(test_loader.dataset)}')

# Compilar el modelo
criterion   = torch.nn.CrossEntropyLoss()
criterion.to(device)
optimizer   = torch.optim.Adam(model.parameters(), lr=0.001)

# Entrenamiento del modelo
train_losses        = []
val_losses          = []
train_accuracies    = []
val_accuracies      = []

for epoch in range(num_epochs):

    # Establecer el modelo en modo de entrenamiento
    model.train()
    running_loss    = 0.0
    correct         = 0
    total           = 0

    for batch_idx, (images, labels) in enumerate(train_loader):

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
        total += labels.size(0)

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total

    running_val_loss    = 0.0
    correct             = 0
    total               = 0
    
    with torch.no_grad():

        for images, targets in val_loader:

            images = images.to(device)
            targets = targets.to(device)

            # Forward propagation
            logits = model(images)

            # Calcular pérdida
            loss = criterion(logits, targets)

            running_val_loss += loss.item()

            # Predicciones
            predictions = logits.argmax(dim=1)

            correct += (predictions == targets).sum().item()
            total += targets.size(0)

    val_loss = running_val_loss / len(val_loader)
    val_accuracy = 100 * correct / total
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)


    print(
        f"Epoch [{epoch + 1}/{num_epochs}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_accuracy:.2f}% "
        f"Val Loss: {val_loss:.4f} "
        f"Val Acc: {val_accuracy:.2f}%"
    )



# Test del modelo
model.eval()
test_loss   = 0.0
correct     = 0
total       = 0

with torch.no_grad():

    for images, targets in test_loader:

        images = images.to(device)
        targets = targets.to(device)

        # Forward propagation
        logits = model(images)

        # Calcular pérdida
        loss = criterion(logits, targets)

        test_loss += loss.item()

        # Predicciones
        predictions = logits.argmax(dim=1)

        correct += (predictions == targets).sum().item()
        total += targets.size(0)

test_loss /= len(test_loader)
test_accuracy = 100 * correct / total

print(f"Test Loss: {test_loss:.4f} Test Accuracy: {test_accuracy:.2f}%")

# Graficar pérdidas y precisiones
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
plt.savefig('loss_accuracy_plot.png')
