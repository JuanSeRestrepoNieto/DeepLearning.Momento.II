# Momento Evaluativo II — CNN para clasificación de aves

Clasificación de 6 especies de aves con una Red Neuronal Convolucional en PyTorch.

- **Dataset:** [The Birds (Kaggle, figolm10)](https://www.kaggle.com/datasets/figolm10/the-birds) — 811 imágenes RGB de 224×224, 6 clases.
- **Entregable principal:** [`MomentoEvaluativo_CNN_Birds.ipynb`](MomentoEvaluativo_CNN_Birds.ipynb) (ejecutado completo).

## Estructura

| Archivo | Contenido |
|---|---|
| `loader.py` | `BirdsDataset`, `Loader` (split estratificado 70/15/15, Resize 128 + ToTensor + Normalize), descarga del dataset |
| `model_cnn.py` | `CNNModel`: 4 bloques Conv-BN-ReLU-MaxPool + 2 capas densas |
| `model_mlp.py` | `MLPModel` (comparación de parámetros) |
| `train_model.py` | `fit`, `evaluate`, `predict`, `plot_history`; ejecutable por CLI |
| `metrics.py` | Matriz de confusión, precision/recall/F1, especificidad, AUC-ROC, visualización de predicciones |
| `figures/` | Gráficas generadas por el notebook |

## Instalación

PyTorch 2.2.2 es la última versión con soporte para Mac Intel (x86_64), por eso se usa Python 3.12 y `numpy<2`.

```bash
python3.12 -m venv .venv
```

```bash
.venv/bin/pip install -r requirements.txt
```

## Datos

`get_data_dir()` usa `data/the-birds/` si existe; si no, descarga el dataset con `kagglehub`. Descarga manual:

```bash
mkdir -p data && curl -L -o data/the-birds.zip https://www.kaggle.com/api/v1/datasets/download/figolm10/the-birds && unzip -q data/the-birds.zip -d data/the-birds && rm data/the-birds.zip
```

## Ejecución

Notebook completo:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace MomentoEvaluativo_CNN_Birds.ipynb
```

Solo entrenamiento por línea de comandos:

```bash
.venv/bin/python train_model.py --epochs 40
```
