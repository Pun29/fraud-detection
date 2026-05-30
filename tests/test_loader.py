# tests/test_loader.py
import pytest
import pandas as pd
from src.data.loader import load_dataset, validate_dataset

def test_load_returns_dataframe(tmp_path, tiny_df):
    path = tmp_path / "creditcard.csv"
    tiny_df.to_csv(path, index=False)
    df = load_dataset(str(path))
    assert isinstance(df, pd.DataFrame)

def test_load_has_expected_columns(tmp_path, tiny_df):
    path = tmp_path / "creditcard.csv"
    tiny_df.to_csv(path, index=False)
    df = load_dataset(str(path))
    assert "Class" in df.columns
    assert "Amount" in df.columns
    assert "Time" in df.columns
    assert all(f"V{i}" in df.columns for i in range(1, 29))

def test_validate_raises_on_missing_column(tiny_df):
    bad_df = tiny_df.drop(columns=["Class"])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_dataset(bad_df)

def test_validate_reports_fraud_count(tiny_df, capsys):
    validate_dataset(tiny_df)
    captured = capsys.readouterr()
    assert "fraud" in captured.out.lower()

def test_no_null_values(tmp_path, tiny_df):
    path = tmp_path / "creditcard.csv"
    tiny_df.to_csv(path, index=False)
    df = load_dataset(str(path))
    assert df.isnull().sum().sum() == 0
