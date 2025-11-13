from __future__ import annotations
from typing import Dict
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


def classification_metrics(y_true, y_pred) -> Dict[str, float]:
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    return {"accuracy": float(acc), "f1_macro": float(f1), "precision_macro": float(p), "recall_macro": float(r)}