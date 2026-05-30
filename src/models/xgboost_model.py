# src/models/xgboost_model.py
import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score
from src.evaluation.metrics import evaluate, log_result
from src.visualization.plots import save_all

RANDOM_SEED = 42

class XGBoostModel:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        use_smote: bool = True,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.use_smote = use_smote
        self.model = None
        self.feature_names = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.feature_names = list(X.columns)
        X_fit, y_fit = X.copy(), y.copy()

        if self.use_smote:
            sm = SMOTE(random_state=RANDOM_SEED)
            X_fit, y_fit = sm.fit_resample(X_fit, y_fit)
            print(f"  SMOTE applied: {len(y_fit)} samples ({y_fit.sum()} fraud)")

        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            eval_metric="aucpr",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        self.model.fit(X_fit, y_fit)

    def anomaly_scores(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importances(self) -> pd.Series:
        return pd.Series(
            self.model.feature_importances_, index=self.feature_names
        ).sort_values(ascending=False)

    def tune(self, X_train, y_train, X_val, y_val,
             n_estimators_values=None, max_depth_values=None):
        if n_estimators_values is None:
            n_estimators_values = [100, 200]
        if max_depth_values is None:
            max_depth_values = [4, 6, 8]
        best_auc, best_params = -1, {}
        for n in n_estimators_values:
            for d in max_depth_values:
                m = XGBoostModel(n_estimators=n, max_depth=d, use_smote=self.use_smote)
                m.fit(X_train, y_train)
                try:
                    auc = roc_auc_score(y_val, m.anomaly_scores(X_val))
                except Exception:
                    auc = 0.0
                print(f"  XGB n_est={n} depth={d}: ROC-AUC={auc:.4f}")
                if auc > best_auc:
                    best_auc = auc
                    best_params = {"n_estimators": n, "max_depth": d}
        print(f"Best XGB params: {best_params}")
        return best_params


def run_xgboost(split, tune: bool = True):
    model = XGBoostModel(use_smote=True)
    if tune:
        best = model.tune(split.X_train, split.y_train, split.X_val, split.y_val)
        model = XGBoostModel(use_smote=True, **best)

    model.fit(split.X_train, split.y_train)
    y_score_val = model.anomaly_scores(split.X_val)
    y_score_test = model.anomaly_scores(split.X_test)

    val_result = evaluate(split.y_val.values, y_score_val, model_name="xgboost_val")
    result = evaluate(split.y_test.values, y_score_test,
                      model_name="xgboost_smote", threshold=val_result.threshold)
    log_result(result)
    save_all(split.y_test.values, y_score_test, result.confusion_matrix, "xgboost_smote")

    print("\nTop 10 features by importance:")
    print(model.get_feature_importances().head(10))
    return result
