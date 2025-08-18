from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from textwrap import wrap

def _prettify_labels(labels, max_chars_per_line: int = 16) -> list[str]:
    """Replace underscores, wrap long labels onto multiple lines."""
    pretty = []
    for s in labels:
        s = str(s).replace("_", " ")
        if len(s) <= max_chars_per_line:
            pretty.append(s)
        else:
            pretty.append("\n".join(wrap(s, max_chars_per_line)))
    return pretty

def save_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],
    title: str,
    outpath: str,
    normalize: bool = True,
    dpi: int = 400,
):
    """
    Save a readable confusion matrix:
    - row-normalised (optional) with counts + percent annotations
    - dynamic figsize based on number of classes
    - exports PNG + PDF + SVG
    """
    labels = _prettify_labels(labels, max_chars_per_line=16)
    cm = np.array(cm, dtype=float)

    # Row normalisation (safer: avoid div-by-zero)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm_norm = cm / row_sums
    else:
        cm_norm = cm.copy()

    n = len(labels)
    # Dynamic size: ~0.8" per class, bounded by sensible minimums
    w = max(8, 0.8 * n)
    h = max(7, 0.8 * n)

    fig, ax = plt.subplots(figsize=(w, h))
    im = ax.imshow(cm_norm, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Row-normalised fraction" if normalize else "Count", rotation=90, va="bottom")

    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=12); ax.set_ylabel("True", fontsize=12)

    # Annotate each cell with count + percent
    for i in range(n):
        for j in range(n):
            cnt = cm[i, j]
            pct = cm_norm[i, j] * 100
            txt = f"{int(cnt)}\n{pct:.1f}%"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9)

    fig.tight_layout()

    outpath = Path(outpath)
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    # Also save vector formats for crystal-clear print
    try:
        fig.savefig(outpath.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(outpath.with_suffix(".svg"), bbox_inches="tight")
    except Exception:
        pass
    plt.close(fig)
