# tests/test_xgboost.py
import numpy as np
import pytest
from src.models.xgboost_model import XGBoostModel

def test_xgboost_fit_predict(tiny_X_y):
    X, y = tiny_X_y
    model = XGBoostModel()
    model.fit(X, y)
    scores = model.anomaly_scores(X)
    assert len(scores) == len(X)
    assert ((scores >= 0) & (scores <= 1)).all()

def test_smote_only_applied_to_train(tiny_X_y):
    X, y = tiny_X_y
    original_len = len(X)
    model = XGBoostModel(use_smote=True)
    model.fit(X, y)
    # SMOTE happens internally; X should be unchanged externally
    assert len(X) == original_len

def test_feature_importances_shape(tiny_X_y):
    X, y = tiny_X_y
    model = XGBoostModel()
    model.fit(X, y)
    importances = model.get_feature_importances()
    assert len(importances) == X.shape[1]
