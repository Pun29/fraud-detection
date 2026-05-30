# src/models/knn_model.py
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from src.evaluation.metrics import evaluate, log_result
from src.visualization.plots import save_all

RANDOM_SEED = 42

class KNNAnomalyDetector:
    def __init__(self, k: int = 5):
        self.k = k
        self.nn = None

    def fit(self, X: pd.DataFrame) -> None:
        self.nn = NearestNeighbors(n_neighbors=self.k + 1, algorithm="ball_tree", n_jobs=-1)
        self.nn.fit(X)

    def anomaly_scores(self, X: pd.DataFrame) -> np.ndarray:
        """Anomaly score = mean distance to k nearest neighbors (excluding self)."""
        distances, _ = self.nn.kneighbors(X)
        return distances[:, 1:].mean(axis=1)  # exclude the point itself (distance=0)

    def tune(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        k_values=None,
    ) -> int:
        """Select k maximizing ROC-AUC on validation set."""
        if k_values is None:
            k_values = [3, 5, 10, 20, 50]
        best_auc, best_k = -1, k_values[0]
        for k in k_values:
            tmp = KNNAnomalyDetector(k=k)
            tmp.fit(X_val)
            scores = tmp.anomaly_scores(X_val)
            try:
                auc = roc_auc_score(y_val, scores)
            except Exception:
                auc = 0.0
            print(f"  KNN k={k}: ROC-AUC={auc:.4f}")
            if auc > best_auc:
                best_auc = auc
                best_k = k
        print(f"Best KNN k={best_k} (ROC-AUC={best_auc:.4f})")
        return best_k

def run_knn(split, tune: bool = True):
    model = KNNAnomalyDetector(k=5)
    model.fit(split.X_train)

    if tune:
        best_k = model.tune(split.X_val, split.y_val)
        model = KNNAnomalyDetector(k=best_k)
        model.fit(split.X_train)

    y_score_val = model.anomaly_scores(split.X_val)
    y_score_test = model.anomaly_scores(split.X_test)

    val_result = evaluate(split.y_val.values, y_score_val, model_name="knn_val")
    result = evaluate(split.y_test.values, y_score_test,
                      model_name="knn", threshold=val_result.threshold)
    log_result(result)
    save_all(split.y_test.values, y_score_test, result.confusion_matrix, "knn")
    return result
