from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
    balanced_accuracy_score, cohen_kappa_score, matthews_corrcoef,
    top_k_accuracy_score, log_loss
)

def basic_metrics(y_true, y_pred, labels_order=None) -> dict:
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm  = confusion_matrix(y_true, y_pred, labels=labels_order)
    report = classification_report(y_true, y_pred, labels=labels_order, output_dict=True, zero_division=0)
    return {"accuracy": acc, "f1_macro": f1m, "cm": cm, "report": pd.DataFrame(report).T}

def extended_metrics(y_true, y_pred, y_proba=None, labels_order=None, topk=(2,3)) -> dict:
    out = {}
    out["accuracy"] = accuracy_score(y_true, y_pred)
    out["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    out["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    out["kappa"] = cohen_kappa_score(y_true, y_pred)
    out["mcc"] = matthews_corrcoef(y_true, y_pred)
    for k in topk:
        try:
            out[f"top{k}_acc"] = top_k_accuracy_score(y_true, y_proba, k=k) if y_proba is not None else np.nan
        except Exception:
            out[f"top{k}_acc"] = np.nan
    try:
        out["log_loss"] = log_loss(y_true, y_proba, labels=labels_order) if y_proba is not None else np.nan
    except Exception:
        out["log_loss"] = np.nan
    return out
