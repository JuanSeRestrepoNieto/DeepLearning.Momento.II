import torch


class MLPModel(torch.nn.Module):
    def __init__(self, input_size, output_size):

        super(MLPModel, self).__init__()

        self.fc1 = torch.nn.Linear(input_size, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, output_size)

        self.relu = torch.nn.ReLU()


    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out


if __name__ == "__main__":

    # Crear un modelo MLP
    img_size = 128
    input_size = img_size * img_size * 3  # Tamaño de la imagen
    output_size = 6  # Número de clases
    model = MLPModel(input_size, output_size)

    image = torch.randn(1, 3, img_size, img_size)  # Imagen de ejemplo
    output = model(torch.flatten(image, 1))  # Forward pass

    # Imprimir la arquitectura del modelo
    print(model)
    print("Output shape:", output.shape)

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"Parámetros entrenables: {trainable_params:,}")