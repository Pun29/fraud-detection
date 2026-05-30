# src/models/logistic_model.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from src.evaluation.metrics import evaluate, log_result
from src.visualization.plots import save_all

RANDOM_SEED = 42

class LogisticRegressionModel:
    def __init__(self, C: float = 1.0):
        self.C = C
        self.model = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model = LogisticRegression(
            C=self.C,
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_SEED,
            solver="lbfgs",
        )
        self.model.fit(X, y)

    def anomaly_scores(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def get_coefficients(self) -> pd.Series:
        return pd.Series(self.model.coef_[0], index=self.model.feature_names_in_)

    def tune(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        C_values=None,
    ) -> float:
        if C_values is None:
            C_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        best_auc, best_C = -1, C_values[0]
        for C in C_values:
            m = LogisticRegressionModel(C=C)
            m.fit(X_train, y_train)
            try:
                auc = roc_auc_score(y_val, m.anomaly_scores(X_val))
            except Exception:
                auc = 0.0
            print(f"  LR C={C}: ROC-AUC={auc:.4f}")
            if auc > best_auc:
                best_auc = auc
                best_C = C
        print(f"Best LR C={best_C} (ROC-AUC={best_auc:.4f})")
        return best_C

def run_logistic(split, tune: bool = True):
    model = LogisticRegressionModel()
    if tune:
        best_C = model.tune(split.X_train, split.y_train, split.X_val, split.y_val)
        model = LogisticRegressionModel(C=best_C)

    model.fit(split.X_train, split.y_train)
    y_score_val = model.anomaly_scores(split.X_val)
    y_score_test = model.anomaly_scores(split.X_test)

    val_result = evaluate(split.y_val.values, y_score_val, model_name="logistic_val")
    result = evaluate(split.y_test.values, y_score_test,
                      model_name="logistic_regression", threshold=val_result.threshold)
    log_result(result)
    save_all(split.y_test.values, y_score_test, result.confusion_matrix, "logistic_regression")

    print("\nTop 10 features by |coefficient|:")
    print(model.get_coefficients().abs().sort_values(ascending=False).head(10))
    return result
