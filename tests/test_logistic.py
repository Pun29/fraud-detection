# tests/test_logistic.py
import numpy as np
import pytest
from src.models.logistic_model import LogisticRegressionModel

def test_lr_fit_predict(tiny_X_y):
    X, y = tiny_X_y
    model = LogisticRegressionModel()
    model.fit(X, y)
    scores = model.anomaly_scores(X)
    assert len(scores) == len(X)
    assert ((scores >= 0) & (scores <= 1)).all()

def test_lr_coefficients_shape(tiny_X_y):
    X, y = tiny_X_y
    model = LogisticRegressionModel()
    model.fit(X, y)
    coefs = model.get_coefficients()
    assert len(coefs) == X.shape[1]

def test_lr_tune_returns_best_C(tiny_X_y):
    X, y = tiny_X_y
    model = LogisticRegressionModel()
    best_C = model.tune(X, y, X, y, C_values=[0.01, 0.1])
    assert best_C in [0.01, 0.1]
