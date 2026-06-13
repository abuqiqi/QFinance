import time
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
)


def metric_row(
    model: str, risk: str, y_true, scores, elapsed: float = 0.0,
    threshold: float = 0.5,
) -> Dict:
    predictions = (np.asarray(scores) >= threshold).astype(int)
    return {
        "model": model,
        "risk": risk,
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, scores) if len(np.unique(y_true)) > 1 else float("nan"),
        "elapsed_seconds": elapsed,
        "threshold": threshold,
    }


def save_confusion(y_true, scores, title: str, path: Path) -> None:
    predictions = (np.asarray(scores) >= 0.5).astype(int)
    display = ConfusionMatrixDisplay.from_predictions(y_true, predictions)
    display.ax_.set_title(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
