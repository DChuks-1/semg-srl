from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

def basic_metrics(y_true, y_pred, labels_order=None) -> dict:
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm  = confusion_matrix(y_true, y_pred, labels=labels_order)
    report = classification_report(y_true, y_pred, labels=labels_order, output_dict=True, zero_division=0)
    return {"accuracy": acc, "f1_macro": f1m, "cm": cm, "report": pd.DataFrame(report).T}
