# tests/test_dbscan.py
import numpy as np
import pytest
from src.models.dbscan_model import DBSCANAnomalyDetector

def test_dbscan_predict_returns_binary(tiny_X_y):
    X, y = tiny_X_y
    model = DBSCANAnomalyDetector(eps=1.5, min_samples=5)
    model.fit(X)
    scores = model.anomaly_scores(X)
    assert set(scores).issubset({0, 1})

def test_dbscan_score_shape(tiny_X_y):
    X, y = tiny_X_y
    model = DBSCANAnomalyDetector(eps=1.5, min_samples=5)
    model.fit(X)
    scores = model.anomaly_scores(X)
    assert len(scores) == len(X)

def test_dbscan_tune_returns_best_params(tiny_X_y):
    X, y = tiny_X_y
    model = DBSCANAnomalyDetector()
    best = model.tune(X, y, eps_values=[0.5, 1.0], min_samples_values=[3, 5])
    assert "eps" in best
    assert "min_samples" in best
