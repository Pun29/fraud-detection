# tests/test_metrics.py
import numpy as np
import pytest
from src.evaluation.metrics import evaluate, EvalResult

def test_evaluate_returns_evalresult():
    y_true = np.array([0, 0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    result = evaluate(y_true, y_score, model_name="test_model")
    assert isinstance(result, EvalResult)

def test_evaluate_fields_present():
    y_true = np.array([0, 0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    result = evaluate(y_true, y_score, model_name="test_model")
    assert hasattr(result, "pr_auc")
    assert hasattr(result, "roc_auc")
    assert hasattr(result, "f1")
    assert hasattr(result, "recall")
    assert hasattr(result, "confusion_matrix")
    assert hasattr(result, "threshold")

def test_perfect_classifier():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.0, 0.0, 1.0, 1.0])
    result = evaluate(y_true, y_score, model_name="perfect")
    assert result.roc_auc == pytest.approx(1.0)
    assert result.recall == pytest.approx(1.0)

def test_threshold_selected_on_val_data():
    y_true = np.array([0, 0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    result = evaluate(y_true, y_score, model_name="test")
    assert 0.0 <= result.threshold <= 1.0
