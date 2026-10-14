import torch
import torch.nn as nn


def conv_output_size(input_size, kernel_size, padding=0, stride=1):
    """Output Shape = floor((input - kernel + 2 * padding) / stride) + 1"""

    return (input_size - kernel_size + 2 * padding) // stride + 1


def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


class CNNModel(nn.Module):

    def __init__(self, num_classes=6, in_channels=3, img_size=128):
        super().__init__()

        # Bloque 1
        self.conv1  = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)

        # Bloque 2
        self.conv2  = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # Bloque 3
        self.conv3  = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        # Bloque 4
        self.conv4  = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

        self.pool           = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout_conv   = nn.Dropout2d(0.2)

        # Tamaño espacial después de los 4 bloques (conv "same" + pool 2x2): 128 -> 64 -> 32 -> 16 -> 8
        size = img_size
        for _ in range(4):
            size = conv_output_size(size, kernel_size=3, padding=1, stride=1)
            size = conv_output_size(size, kernel_size=2, padding=0, stride=2)

        self.flatten_size = 256 * size * size

        # Clasificador
        self.fc1        = nn.Linear(self.flatten_size, 128)
        self.dropout    = nn.Dropout(0.5)
        self.fc2        = nn.Linear(128, num_classes)

        self.relu       = nn.ReLU()

    def features(self, x):
        """Etapa de extracción de características; retorna los mapas de cada bloque."""

        maps = []

        # [B, 3, 128, 128] -> [B, 32, 64, 64]
        out = self.pool(self.relu(self.conv1(x)))
        maps.append(out)

        # -> [B, 64, 32, 32]
        out = self.pool(self.relu(self.conv2(out)))
        maps.append(out)

        # -> [B, 128, 16, 16]
        out = self.dropout_conv(self.pool(self.relu(self.conv3(out))))
        maps.append(out)

        # -> [B, 256, 8, 8]
        out = self.dropout_conv(self.pool(self.relu(self.conv4(out))))
        maps.append(out)

        return maps

    def forward(self, x):

        out = self.features(x)[-1]

        # Flatten: [B, 256, 8, 8] -> [B, 16384]
        out = torch.flatten(out, 1)

        # Clasificador
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out


if __name__ == "__main__":

    # Crear un modelo CNN
    model = CNNModel(num_classes=6, img_size=128)

    image = torch.randn(1, 3, 128, 128)  # Imagen de ejemplo
    output = model(image)  # Forward pass

    # Imprimir la arquitectura del modelo
    print(model)
    print("Output shape:", output.shape)
    print(f"Parámetros entrenables: {count_parameters(model):,}")
