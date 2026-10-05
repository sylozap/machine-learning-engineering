"""Оценка качества модели на валидационной и тестовой выборках."""
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from common import FEATURES_DIR, METRICS_DIR, MODELS_DIR, PLOTS_DIR, PREPARED_DIR, save_json


def scores(y_true, y_pred):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro"), 4),
        "precision_macro": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "recall_macro": round(recall_score(y_true, y_pred, average="macro"), 4),
    }


def plot_confusion(y_true, y_pred, n_classes):
    cm = confusion_matrix(y_true, y_pred, labels=range(n_classes), normalize="true")
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")
    ax.set_title("Нормированная матрица ошибок (test)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "confusion_matrix.png", dpi=120)
    plt.close(fig)


def main():
    model = joblib.load(MODELS_DIR / "model.joblib")
    classes = json.loads((PREPARED_DIR / "classes.json").read_text(encoding="utf-8"))

    metrics = {}
    for split in ["val", "test"]:
        data = np.load(FEATURES_DIR / f"{split}.npz")
        y_pred = model.predict(data["X"])
        metrics[split] = scores(data["y"], y_pred)
    save_json(metrics, METRICS_DIR / "eval.json")

    PLOTS_DIR.mkdir(exist_ok=True)
    y_true = data["y"]
    pd.DataFrame({"actual": [classes[i] for i in y_true], "predicted": [classes[i] for i in y_pred]}).to_csv(
        PLOTS_DIR / "confusion.csv", index=False
    )
    plot_confusion(y_true, y_pred, len(classes))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
