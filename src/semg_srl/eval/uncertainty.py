from __future__ import annotations
import numpy as np
from typing import Callable, Tuple

def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
    random_state: int = 42,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """
    Nonparametric bootstrap CI over indices of concatenated test predictions.
    """
    rng = np.random.default_rng(random_state)
    n = y_true.shape[0]
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats.append(metric_fn(y_true[idx], y_pred[idx]))
    lo, hi = np.percentile(stats, [100*alpha/2, 100*(1-alpha/2)])
    return float(lo), float(hi)
