"""Metrici + matrice de confuzie + raport text."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from . import config as C

def evaluate_model(model, X_test, y_test, reports_dir: Path):
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    y_test = np.asarray(y_test).astype(str)
    y_pred = np.asarray(model.predict(X_test)).astype(str)
    acc = accuracy_score(y_test, y_pred)
    labels = sorted(set(y_test) | set(y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    report = classification_report(y_test, y_pred, labels=labels,
                                   zero_division=0)
    # Acuratețea globală include clasa NONE_LABEL (eliberări), care e ușoară;
    # cifra relevantă e acuratețea pe apăsările reale.
    keys = y_test != C.NONE_LABEL
    summary = f"Accuracy (toate ferestrele): {acc:.4f}\n"
    if keys.any():
        key_acc = float(np.mean(y_pred[keys] == y_test[keys]))
        majority = Counter(y_test[keys]).most_common(1)[0][1] / keys.sum()
        summary += (f"Accuracy doar pe taste:     {key_acc:.4f} "
                    f"(n={int(keys.sum())}, baseline clasa majoritară {majority:.4f})\n")
    report = summary + "\n" + report
    (reports_dir / "classification_report.txt").write_text(report + "\n")
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix (acc={acc:.3f})")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(reports_dir / "confusion_matrix.png", dpi=120)
    plt.close(fig)
    return acc, report, cm, labels
