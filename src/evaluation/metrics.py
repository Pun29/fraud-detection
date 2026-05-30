# src/evaluation/metrics.py
import json
import os
from dataclasses import dataclass, asdict
from typing import Optional
import numpy as np
from sklearn.metrics import (
    precision_recall_curve, roc_auc_score, auc,
    f1_score, recall_score, confusion_matrix
)

@dataclass
class EvalResult:
    model_name: str
    pr_auc: float
    roc_auc: float
    f1: float
    recall: float
    threshold: float
    confusion_matrix: list  # [[TN, FP], [FN, TP]]

def _best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray):
    """Find threshold maximizing F1 on the given data (must be val set)."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1s[:-1])  # last threshold is excluded
    return thresholds[best_idx]

def evaluate(
    y_true: np.ndarray,
    y_score: np.ndarray,
    model_name: str,
    threshold: Optional[float] = None,
) -> EvalResult:
    """
    Evaluate a model given true labels and anomaly scores.
    If threshold is None, selects best F1 threshold from y_score (use val data).
    """
    if threshold is None:
        threshold = _best_f1_threshold(y_true, y_score)

    y_pred = (y_score >= threshold).astype(int)

    precisions, recalls, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recalls, precisions)
    roc_auc = roc_auc_score(y_true, y_score)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()

    result = EvalResult(
        model_name=model_name,
        pr_auc=round(pr_auc, 4),
        roc_auc=round(roc_auc, 4),
        f1=round(f1, 4),
        recall=round(recall, 4),
        threshold=round(float(threshold), 4),
        confusion_matrix=cm,
    )
    return result

def log_result(result: EvalResult, log_path: str = "experiments/experiment_log.json") -> None:
    """Append result to the central experiment log."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    existing = []
    if os.path.exists(log_path):
        with open(log_path) as f:
            existing = json.load(f)
    existing.append(asdict(result))
    with open(log_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"[{result.model_name}] PR-AUC={result.pr_auc} | ROC-AUC={result.roc_auc} | F1={result.f1} | Recall={result.recall}")
