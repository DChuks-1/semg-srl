from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def save_confusion_matrix(cm: np.ndarray, labels: list[str], title: str, outpath: str):
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(6, 5))
    ax = plt.gca()
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8)
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
