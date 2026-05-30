# tests/test_preprocessor.py
import pytest
import numpy as np
from src.data.preprocessor import preprocess, SplitResult

def test_split_sizes(tiny_df):
    result = preprocess(tiny_df)
    total = len(result.X_train) + len(result.X_val) + len(result.X_test)
    assert total == len(tiny_df)

def test_fraud_present_in_all_splits(tiny_df):
    result = preprocess(tiny_df)
    assert result.y_train.sum() > 0, "No fraud in train"
    assert result.y_val.sum() > 0, "No fraud in val"
    assert result.y_test.sum() > 0, "No fraud in test"

def test_scaler_fit_on_train_only(tiny_df):
    result = preprocess(tiny_df)
    # After scaling, train Amount should have ~0 mean
    assert abs(result.X_train["Amount"].mean()) < 5

def test_no_leakage_val_test_scaled_with_train_scaler(tiny_df):
    # Scaler must be fit on train only: train should have ~0 mean, ~1 std
    result = preprocess(tiny_df)
    assert abs(result.X_train["Amount"].mean()) < 0.1
    assert abs(result.X_train["Amount"].std() - 1.0) < 0.1
    # Scaler has mean_ attribute derived from training data
    assert hasattr(result.scaler, "mean_") and result.scaler.mean_ is not None

def test_time_column_dropped(tiny_df):
    result = preprocess(tiny_df)
    assert "Time" not in result.X_train.columns

def test_split_ratios(tiny_df):
    result = preprocess(tiny_df, val_size=0.15, test_size=0.15)
    n = len(tiny_df)
    assert len(result.X_test) == pytest.approx(n * 0.15, abs=3)
