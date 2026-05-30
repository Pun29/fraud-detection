# src/models/dbscan_model.py
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import f1_score
from src.evaluation.metrics import evaluate, log_result
from src.visualization.plots import save_all

RANDOM_SEED = 42

class DBSCANAnomalyDetector:
    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        self.eps = eps
        self.min_samples = min_samples
        self.model = None

    def fit(self, X: pd.DataFrame) -> None:
        self.model = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            algorithm="ball_tree",
            n_jobs=-1,
        )
        self.model.fit(X)
        from sklearn.neighbors import BallTree
        # Store core samples for scoring new points; fall back to all points if none found
        core_idx = self.model.core_sample_indices_
        ref = X.values[core_idx] if len(core_idx) > 0 else X.values
        self._ball_tree = BallTree(ref)
        self._has_cores = len(core_idx) > 0

    def anomaly_scores(self, X: pd.DataFrame) -> np.ndarray:
        """Returns 1 if farther than eps from any core sample (anomaly), else 0."""
        data = X.values if hasattr(X, "values") else X
        dist, _ = self._ball_tree.query(data, k=1)
        return (dist[:, 0] > self.eps).astype(int)

    def tune(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eps_values=None,
        min_samples_values=None,
        subsample: float = 0.1,
    ) -> dict:
        """Grid search eps and min_samples on a subsample, maximizing F1."""
        if eps_values is None:
            eps_values = [0.3, 0.5, 0.8, 1.0, 1.5]
        if min_samples_values is None:
            min_samples_values = [5, 10, 20]

        # Subsample for speed (use all data if small enough)
        n = max(int(len(X) * subsample), min(len(X), 500))
        idx = np.random.RandomState(RANDOM_SEED).choice(len(X), size=n, replace=False)
        X_sub = X.iloc[idx]
        y_sub = y.iloc[idx]

        best_f1, best_params = -1, {"eps": eps_values[0], "min_samples": min_samples_values[0]}
        for eps in eps_values:
            for ms in min_samples_values:
                db = DBSCAN(eps=eps, min_samples=ms, algorithm="ball_tree", n_jobs=-1)
                db.fit(X_sub)
                # Use training labels directly for tuning (fit and score same subsample)
                scores = (db.labels_ == -1).astype(int)
                if len(db.core_sample_indices_) == 0:
                    print(f"  DBSCAN eps={eps} min_samples={ms}: no core samples, skipping")
                    continue
                f1 = f1_score(y_sub, scores, zero_division=0)
                print(f"  DBSCAN eps={eps} min_samples={ms}: F1={f1:.4f}, noise_ratio={scores.mean():.4f}")
                if f1 > best_f1:
                    best_f1 = f1
                    best_params = {"eps": eps, "min_samples": ms}

        print(f"Best DBSCAN params: {best_params} (F1={best_f1:.4f})")
        return best_params

def run_dbscan(split, tune: bool = True):
    model = DBSCANAnomalyDetector()
    if tune:
        best = model.tune(split.X_train, split.y_train)
        model = DBSCANAnomalyDetector(**best)

    model.fit(split.X_train)
    y_score_val = model.anomaly_scores(split.X_val)
    y_score_test = model.anomaly_scores(split.X_test)

    val_result = evaluate(split.y_val.values, y_score_val.astype(float), model_name="dbscan_val")
    result = evaluate(split.y_test.values, y_score_test.astype(float),
                      model_name="dbscan", threshold=0.5)
    log_result(result)
    save_all(split.y_test.values, y_score_test.astype(float), result.confusion_matrix, "dbscan")
    return result
