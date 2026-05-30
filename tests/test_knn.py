# tests/test_knn.py
import numpy as np
import pytest
from src.models.knn_model import KNNAnomalyDetector

def test_knn_score_shape(tiny_X_y):
    X, y = tiny_X_y
    model = KNNAnomalyDetector(k=5)
    model.fit(X)
    scores = model.anomaly_scores(X)
    assert len(scores) == len(X)

def test_knn_scores_non_negative(tiny_X_y):
    X, y = tiny_X_y
    model = KNNAnomalyDetector(k=5)
    model.fit(X)
    scores = model.anomaly_scores(X)
    assert (scores >= 0).all()

def test_knn_tune_returns_best_k(tiny_X_y):
    X, y = tiny_X_y
    model = KNNAnomalyDetector()
    best_k = model.tune(X, y, k_values=[3, 5, 7])
    assert isinstance(best_k, int)
    assert best_k in [3, 5, 7]
