import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


def plot_confusion_matrix(y_true, y_pred, classes, path=None, title="Matriz de confusión"):

    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))

    fig = plt.figure(figsize=(8, 6.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel("Predicción")
    plt.ylabel("Etiqueta")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if path:
        fig.savefig(path, dpi=120)
    plt.show()

    return cm


def per_class_metrics(cm, classes):
    """TP, FP, FN, TN por clase (esa clase como positiva y el resto como negativas) y métricas derivadas."""

    total   = cm.sum()
    rows    = []

    for i, clase in enumerate(classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp

        rows.append({
            "Clase":                    clase,
            "TP":                       tp,
            "FP":                       fp,
            "FN":                       fn,
            "TN":                       tn,
            "Precision":                tp / (tp + fp) if tp + fp else 0.0,
            "Recall (Sensibilidad)":    tp / (tp + fn) if tp + fn else 0.0,
            "Especificidad":            tn / (tn + fp) if tn + fp else 0.0,
            "Tasa FP":                  fp / (fp + tn) if fp + tn else 0.0,
            "Tasa FN":                  fn / (fn + tp) if fn + tp else 0.0,
        })

    df = pd.DataFrame(rows).set_index("Clase")
    df["F1-Score"] = (
        2 * df["Precision"] * df["Recall (Sensibilidad)"]
        / (df["Precision"] + df["Recall (Sensibilidad)"]).replace(0, np.nan)
    ).fillna(0.0)

    return df


def summary_metrics(y_true, y_pred, y_prob, classes):
    """Accuracy global y promedio entre clases de Precision, Recall, F1-Score y AUC-ROC."""

    per_class   = per_class_metrics(confusion_matrix(y_true, y_pred, labels=range(len(classes))), classes)
    aucs        = [roc_auc_score((y_true == i).astype(int), y_prob[:, i]) for i in range(len(classes))]

    return pd.Series({
        "Accuracy":                 accuracy_score(y_true, y_pred),
        "Precision (promedio)":     per_class["Precision"].mean(),
        "Recall (promedio)":        per_class["Recall (Sensibilidad)"].mean(),
        "Especificidad (promedio)": per_class["Especificidad"].mean(),
        "F1-Score (promedio)":      per_class["F1-Score"].mean(),
        "AUC-ROC (promedio)":       np.mean(aucs),
    })


def plot_roc_curves(y_true, y_prob, classes, path=None):
    """Curva ROC de cada clase (esa clase como positiva y el resto como negativas) con su AUC."""

    fig = plt.figure(figsize=(7, 6))

    for i, clase in enumerate(classes):
        fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_prob[:, i])
        auc         = roc_auc_score((y_true == i).astype(int), y_prob[:, i])
        plt.plot(fpr, tpr, label=f"{clase} (AUC = {auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", label="Clasificador aleatorio")
    plt.xlabel("Tasa de falsos positivos (1 - Especificidad)")
    plt.ylabel("Tasa de verdaderos positivos (Sensibilidad)")
    plt.title("Curvas ROC por clase")
    plt.legend(fontsize=8)
    plt.tight_layout()

    if path:
        fig.savefig(path, dpi=120)
    plt.show()


def top_confusions(cm, classes, k=5):
    """Pares (real -> predicha) con más errores."""

    pairs = [
        (classes[i], classes[j], int(cm[i, j]))
        for i in range(len(classes))
        for j in range(len(classes))
        if i != j and cm[i, j] > 0
    ]
    pairs.sort(key=lambda p: p[2], reverse=True)

    return pd.DataFrame(pairs[:k], columns=["Clase real", "Clase predicha", "Errores"])


def denormalize(image, mean, std):
    """Revierte Normalize para poder visualizar: tensor [C, H, W] -> array [H, W, C] en [0, 1]."""

    image = image.cpu().numpy().transpose(1, 2, 0)
    image = image * np.array(std) + np.array(mean)

    return np.clip(image, 0, 1)


def show_predictions(dataset, positions, y_true, y_pred, y_prob, classes, mean, std, title, path=None):
    """Muestra las imágenes del dataset (Subset de prueba) en las posiciones indicadas."""

    n   = len(positions)
    fig = plt.figure(figsize=(3.2 * n, 3.8))

    for k, pos in enumerate(positions):
        image, _ = dataset[pos]
        ok       = y_true[pos] == y_pred[pos]

        plt.subplot(1, n, k + 1)
        plt.imshow(denormalize(image, mean, std))
        plt.title(
            f"Real: {classes[y_true[pos]]}\nPred: {classes[y_pred[pos]]} ({y_prob[pos].max():.0%})",
            fontsize=8,
            color="green" if ok else "red",
        )
        plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()

    if path:
        fig.savefig(path, dpi=120)
    plt.show()
