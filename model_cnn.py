import torch
import torch.nn as nn

class CNNModel(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()

        # Bloque 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)

        # Bloque 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Bloque 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.dropout_conv = nn.Dropout2d(0.2)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Clasificador
        self.fc1        = nn.Linear(128 * 4 * 4, 256)
        self.dropout    = nn.Dropout(0.5)
        self.fc2        = nn.Linear(256, num_classes)

        self.relu       = nn.ReLU()


    def forward(self, x):

        # [B, 3, 32, 32] -> [B, 32, 16, 16]
        out = self.conv1(x)
        out = self.relu(out)
        out = self.pool(out)

        # -> [B, 64, 8, 8]
        out = self.conv2(out)
        out = self.relu(out)
        out = self.pool(out)
        out = self.dropout_conv(out)
       
        # -> [B, 128, 4, 4]
        out = self.conv3(out)
        out = self.relu(out)
        out = self.pool(out)
        out = self.dropout_conv(out)
        

        # Flatten
        out = torch.flatten(out, 1)

        # Clasificador
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out


if __name__ == "__main__":

    # Crear un modelo CNN
    model = CNNModel(num_classes=10)

    image = torch.randn(1, 3, 32, 32)  # Imagen de ejemplo
    output = model(image)  # Forward pass

    # Imprimir la arquitectura del modelo
    print(model)
    print("Output shape:", output.shape)

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"Parámetros entrenables: {trainable_params:,}")