# src/data/preprocessor.py
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42

@dataclass
class SplitResult:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    scaler: StandardScaler

def preprocess(
    df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = RANDOM_SEED,
) -> SplitResult:
    # Drop Time (temporal leakage risk; not useful after PCA features exist)
    df = df.drop(columns=["Time"])

    X = df.drop(columns=["Class"])
    y = df["Class"]

    # First split off test set (stratified)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    # Then split val from remaining
    val_relative = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_relative, stratify=y_temp, random_state=random_state
    )

    # Fit scaler ONLY on training data
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_val = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns, index=X_val.index)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    print(f"Train: {len(X_train)} ({y_train.sum()} fraud) | Val: {len(X_val)} ({y_val.sum()} fraud) | Test: {len(X_test)} ({y_test.sum()} fraud)")
    return SplitResult(X_train, X_val, X_test, y_train, y_val, y_test, scaler)
