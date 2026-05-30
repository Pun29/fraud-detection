# tests/conftest.py
import pytest
import numpy as np
import pandas as pd

RANDOM_SEED = 42

@pytest.fixture
def tiny_df():
    """Synthetic dataset mimicking creditcard.csv structure: 200 legit, 10 fraud."""
    np.random.seed(RANDOM_SEED)
    n_legit, n_fraud = 200, 10
    n = n_legit + n_fraud
    v_cols = {f"V{i}": np.random.randn(n) for i in range(1, 29)}
    df = pd.DataFrame({
        "Time": np.linspace(0, 172800, n),
        "Amount": np.abs(np.random.randn(n)) * 50,
        **v_cols,
        "Class": [0] * n_legit + [1] * n_fraud,
    })
    return df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

@pytest.fixture
def tiny_X_y(tiny_df):
    X = tiny_df.drop(columns=["Class"])
    y = tiny_df["Class"]
    return X, y
