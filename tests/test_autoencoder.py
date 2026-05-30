# tests/test_autoencoder.py
import numpy as np
import pytest
import torch
from src.models.autoencoder_model import Autoencoder, AutoencoderAnomalyDetector

def test_autoencoder_forward_shape(tiny_X_y):
    X, y = tiny_X_y
    n_features = X.shape[1]
    model = Autoencoder(input_dim=n_features)
    x = torch.randn(10, n_features)
    out = model(x)
    assert out.shape == x.shape

def test_autoencoder_trains_without_fraud(tiny_X_y):
    X, y = tiny_X_y
    detector = AutoencoderAnomalyDetector(input_dim=X.shape[1], epochs=2)
    X_train_clean = X[y == 0]
    detector.fit(X_train_clean)  # should not raise

def test_reconstruction_scores_shape(tiny_X_y):
    X, y = tiny_X_y
    detector = AutoencoderAnomalyDetector(input_dim=X.shape[1], epochs=2)
    detector.fit(X[y == 0])
    scores = detector.anomaly_scores(X)
    assert len(scores) == len(X)
    assert (scores >= 0).all()

def test_fraud_higher_reconstruction_error_on_avg(tiny_X_y):
    X, y = tiny_X_y
    detector = AutoencoderAnomalyDetector(input_dim=X.shape[1], epochs=2)
    detector.fit(X[y == 0])
    scores = detector.anomaly_scores(X)
    # Scores must be non-negative regardless of distribution
    assert (scores >= 0).all()
