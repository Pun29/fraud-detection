# src/data/loader.py
import pandas as pd

REQUIRED_COLUMNS = {"Time", "Amount", "Class"} | {f"V{i}" for i in range(1, 29)}

def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    validate_dataset(df)
    return df

def validate_dataset(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    n_fraud = df["Class"].sum()
    n_total = len(df)
    print(f"Dataset loaded: {n_total} transactions, {n_fraud} fraud ({100*n_fraud/n_total:.3f}%)")
    if df.isnull().sum().sum() > 0:
        print("WARNING: null values detected — dropping rows with nulls")
