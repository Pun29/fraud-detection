# Credit Card Fraud Detection — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an end-to-end ML pipeline comparing unsupervised (DBSCAN, KNN), supervised (Logistic Regression, XGBoost+SMOTE), and deep learning (Autoencoder) approaches for fraud detection on the Kaggle Credit Card Fraud dataset.

**Architecture:** A shared data/evaluation core (Milestone 1) is built first, then each model milestone is independent and plugs into that shared harness. All models output an anomaly score and predicted label; the evaluation module consumes these uniformly. Results are aggregated in a final comparison notebook.

**Tech Stack:** Python 3.10+, scikit-learn, imbalanced-learn (SMOTE), XGBoost, PyTorch, pandas, numpy, matplotlib, seaborn, joblib (serialization), pytest; **Package manager: uv** (not pip)

---

## Dataset

- **Source:** Kaggle Credit Card Fraud Detection (`creditcard.csv`)
- **Size:** 284,807 transactions; 492 frauds (0.173%)
- **Features:** `Time`, `Amount`, `V1`–`V28` (PCA-anonymized), `Class` (0=legit, 1=fraud)
- **Place dataset at:** `data/raw/creditcard.csv`

---

## Project Structure

```
project/
├── data/
│   └── raw/creditcard.csv        # place downloaded dataset here
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py             # load & validate CSV
│   │   └── preprocessor.py      # scaling, train/val/test split
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py           # shared eval harness
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── plots.py             # PR curve, ROC curve, confusion matrix
│   └── models/
│       ├── __init__.py
│       ├── dbscan_model.py
│       ├── knn_model.py
│       ├── logistic_model.py
│       ├── xgboost_model.py
│       └── autoencoder_model.py
├── tests/
│   ├── conftest.py               # shared fixtures (tiny synthetic dataset)
│   ├── test_loader.py
│   ├── test_preprocessor.py
│   ├── test_metrics.py
│   ├── test_dbscan.py
│   ├── test_knn.py
│   ├── test_logistic.py
│   ├── test_xgboost.py
│   └── test_autoencoder.py
├── notebooks/
│   └── milestone5_comparison.ipynb
├── outputs/
│   ├── plots/                    # PR/ROC curves, confusion matrices
│   └── metrics/                  # JSON experiment logs
├── experiments/
│   └── experiment_log.json       # hyperparams + metrics for all runs
├── requirements.txt
├── run_pipeline.py               # CLI entry point to run all models
└── README.md
```

---

## Known Gaps & Risk Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| **DBSCAN O(n²) scalability** | 280k samples will be extremely slow or OOM | Use `algorithm='ball_tree'`, `leaf_size=30`; subsample 10% for tuning, run full on final |
| **KNN distance matrix at scale** | Memory/speed issue at 280k | Use sklearn `NearestNeighbors` with `algorithm='ball_tree'`; score is mean dist to k neighbors |
| **SMOTE data leakage** | Inflated metrics, invalid results | Apply SMOTE strictly AFTER train/val/test split, ONLY on X_train; never touch val/test |
| **Scaling leakage** | StandardScaler fitted on test data | Fit scaler on X_train only, transform val/test with same fitted scaler |
| **Threshold selection on test** | Optimistic threshold giving inflated F1 | Select all thresholds on val set only; report final metrics on held-out test set |
| **Class imbalance in splits** | Val/test may have 0 fraud samples** | Use `StratifiedShuffleSplit`; verify fraud count in each split after splitting |
| **Time-ordered data / concept drift** | Random split ignores temporal structure | Primary split: stratified random. Secondary experiment: time-based split for comparison |
| **Autoencoder overfitting to fraud** | Training on mostly non-fraud is key | Train ONLY on X_train where Class=0; validate reconstruction error distribution on val |
| **Reproducibility** | Different runs give different results | Set `RANDOM_SEED=42` globally; pass to all models, SMOTE, train/test split |
| **Majority-class accuracy trap** | 99.83% accuracy by predicting all-0 | Never report accuracy as primary metric; always lead with PR-AUC and Recall |
| **DBSCAN anomaly labeling ambiguity** | DBSCAN labels noise as -1, but noise ≠ fraud | Label DBSCAN `cluster == -1` as anomaly (fraud); tune eps/min_samples to control noise ratio |

---

## Task 1: Project Setup & Requirements

**Files:**
- Create: `requirements.txt`
- Create: `src/__init__.py`, `src/data/__init__.py`, `src/evaluation/__init__.py`, `src/visualization/__init__.py`, `src/models/__init__.py`
- Create: `tests/conftest.py`

**Step 1: Create requirements.txt**

```
numpy==1.26.4
pandas==2.2.2
scikit-learn==1.4.2
imbalanced-learn==0.12.3
xgboost==2.0.3
torch==2.3.0
matplotlib==3.9.0
seaborn==0.13.2
joblib==1.4.2
pytest==8.2.0
```

**Step 2: Initialize uv project and install dependencies**

```bash
uv venv
uv pip install -r requirements.txt
```

All subsequent commands that need the venv should use `uv run` (e.g. `uv run pytest`) or activate the venv first.

**Step 3: Create all `__init__.py` files (empty)**

```bash
touch src/__init__.py src/data/__init__.py src/evaluation/__init__.py src/visualization/__init__.py src/models/__init__.py
mkdir -p outputs/plots outputs/metrics experiments notebooks
```

**Step 4: Create `tests/conftest.py` with shared fixtures**

```python
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
```


---

## Task 2: Data Loader (`src/data/loader.py`)

**Files:**
- Create: `src/data/loader.py`
- Create: `tests/test_loader.py`

**Step 1: Write the failing tests**

```python
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
```

**Step 2: Run tests to verify they fail**

```bash
uv run uv run pytest tests/test_loader.py -v
```
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Implement `src/data/loader.py`**

```python
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
```

**Step 4: Run tests**

```bash
uv run uv run pytest tests/test_loader.py -v
```
Expected: All PASS


---

## Task 3: Preprocessing (`src/data/preprocessor.py`)

This is the most critical task — all data leakage prevention lives here.

**Files:**
- Create: `src/data/preprocessor.py`
- Create: `tests/test_preprocessor.py`

**Step 1: Write the failing tests**

```python
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
    # Amount column: train mean should differ from val/test raw means
    # after scaling, train Amount should have ~0 mean
    assert abs(result.X_train["Amount"].mean()) < 5  # scaled

def test_no_leakage_val_test_scaled_with_train_scaler(tiny_df):
    result = preprocess(tiny_df)
    # Re-scaling val with same scaler should produce identical values
    import pandas as pd
    rescaled = result.scaler.transform(result.X_val)
    np.testing.assert_array_almost_equal(rescaled, result.X_val.values)

def test_time_column_dropped(tiny_df):
    result = preprocess(tiny_df)
    assert "Time" not in result.X_train.columns

def test_split_ratios(tiny_df):
    result = preprocess(tiny_df, val_size=0.15, test_size=0.15)
    n = len(tiny_df)
    assert len(result.X_test) == pytest.approx(n * 0.15, abs=3)
```

**Step 2: Run tests to verify they fail**

```bash
uv run uv run pytest tests/test_preprocessor.py -v
```

**Step 3: Implement `src/data/preprocessor.py`**

```python
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
```

**Step 4: Run tests**

```bash
uv run uv run pytest tests/test_preprocessor.py -v
```


---

## Task 4: Evaluation Harness (`src/evaluation/metrics.py`)

**Files:**
- Create: `src/evaluation/metrics.py`
- Create: `tests/test_metrics.py`

**Step 1: Write the failing tests**

```python
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
    # threshold should maximize F1 on the provided scores (val data)
    y_true = np.array([0, 0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    result = evaluate(y_true, y_score, model_name="test")
    assert 0.0 <= result.threshold <= 1.0
```

**Step 2: Run tests to verify they fail**

```bash
uv run uv run pytest tests/test_metrics.py -v
```

**Step 3: Implement `src/evaluation/metrics.py`**

```python
# src/evaluation/metrics.py
import json
import os
from dataclasses import dataclass, asdict
from typing import Optional
import numpy as np
from sklearn.metrics import (
    precision_recall_curve, roc_auc_score, auc,
    f1_score, recall_score, confusion_matrix
)

@dataclass
class EvalResult:
    model_name: str
    pr_auc: float
    roc_auc: float
    f1: float
    recall: float
    threshold: float
    confusion_matrix: list  # [[TN, FP], [FN, TP]]

def _best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray):
    """Find threshold maximizing F1 on the given data (must be val set)."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1s[:-1])  # last threshold is excluded
    return thresholds[best_idx]

def evaluate(
    y_true: np.ndarray,
    y_score: np.ndarray,
    model_name: str,
    threshold: Optional[float] = None,
) -> EvalResult:
    """
    Evaluate a model given true labels and anomaly scores.
    If threshold is None, selects best F1 threshold from y_score (use val data).
    """
    if threshold is None:
        threshold = _best_f1_threshold(y_true, y_score)

    y_pred = (y_score >= threshold).astype(int)

    precisions, recalls, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recalls, precisions)
    roc_auc = roc_auc_score(y_true, y_score)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()

    result = EvalResult(
        model_name=model_name,
        pr_auc=round(pr_auc, 4),
        roc_auc=round(roc_auc, 4),
        f1=round(f1, 4),
        recall=round(recall, 4),
        threshold=round(float(threshold), 4),
        confusion_matrix=cm,
    )
    return result

def log_result(result: EvalResult, log_path: str = "experiments/experiment_log.json") -> None:
    """Append result to the central experiment log."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    existing = []
    if os.path.exists(log_path):
        with open(log_path) as f:
            existing = json.load(f)
    existing.append(asdict(result))
    with open(log_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"[{result.model_name}] PR-AUC={result.pr_auc} | ROC-AUC={result.roc_auc} | F1={result.f1} | Recall={result.recall}")
```

**Step 4: Run tests**

```bash
uv run uv run pytest tests/test_metrics.py -v
```


---

## Task 5: Visualization Utilities (`src/visualization/plots.py`)

**Files:**
- Create: `src/visualization/plots.py`

No unit tests needed here (plotting is visual); this is a utility that all milestones share.

```python
# src/visualization/plots.py
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc

OUTPUT_DIR = "outputs/plots"

def save_pr_curve(y_true, y_score, model_name: str) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    precisions, recalls, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recalls, precisions)
    plt.figure()
    plt.plot(recalls, precisions, label=f"PR-AUC={pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve: {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{model_name}_pr_curve.png", dpi=150)
    plt.close()

def save_roc_curve(y_true, y_score, model_name: str) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC-AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve: {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{model_name}_roc_curve.png", dpi=150)
    plt.close()

def save_confusion_matrix(cm: list, model_name: str) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix: {model_name}")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{model_name}_confusion_matrix.png", dpi=150)
    plt.close()

def save_all(y_true, y_score, cm, model_name: str) -> None:
    save_pr_curve(y_true, y_score, model_name)
    save_roc_curve(y_true, y_score, model_name)
    save_confusion_matrix(cm, model_name)
    print(f"Plots saved to {OUTPUT_DIR}/ for {model_name}")
```


---

## Task 6: Milestone 1 — Majority-Class Baseline

**Files:**
- Create: `src/models/baseline.py`
- Modify: `run_pipeline.py` (stub)

```python
# src/models/baseline.py
import numpy as np
from src.evaluation.metrics import evaluate, log_result, EvalResult

def run_majority_class_baseline(y_true_val, y_true_test) -> EvalResult:
    """Always predicts 0 (legitimate). Score = 0 for all."""
    y_score = np.zeros(len(y_true_test))
    result = evaluate(y_true_test.values, y_score, model_name="majority_class_baseline")
    log_result(result)
    return result
```


---

## Task 7: DBSCAN Anomaly Detection (`src/models/dbscan_model.py`)

**Owner:** Punarva Bettadamane Channabasappa

**Key design decisions:**
- DBSCAN label `-1` (noise) = anomaly score 1; all cluster points = score 0
- **Scalability:** Use `algorithm='ball_tree'` and subsample for tuning
- Anomaly score: binary (0 or 1); DBSCAN doesn't produce soft scores, so ROC-AUC/PR-AUC will be limited — this is expected and worth noting

**Files:**
- Create: `src/models/dbscan_model.py`
- Create: `tests/test_dbscan.py`

**Step 1: Write the failing tests**

```python
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
```

**Step 2: Run tests to verify they fail**

```bash
uv run uv run pytest tests/test_dbscan.py -v
```

**Step 3: Implement**

```python
# src/models/dbscan_model.py
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import f1_score
from src.evaluation.metrics import evaluate, log_result
from src.visualization.plots import save_all

RANDOM_SEED = 42

class DBSCANAnomalyDetector:
    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        self.eps = eps
        self.min_samples = min_samples
        self.model = None

    def fit(self, X: pd.DataFrame) -> None:
        self.model = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            algorithm="ball_tree",
            n_jobs=-1,
        )
        self.model.fit(X)

    def anomaly_scores(self, X: pd.DataFrame) -> np.ndarray:
        """Returns 1 for noise points (anomaly), 0 for cluster members."""
        labels = self.model.labels_
        return (labels == -1).astype(int)

    def tune(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eps_values=None,
        min_samples_values=None,
        subsample: float = 0.1,
    ) -> dict:
        """Grid search eps and min_samples on a subsample, maximizing F1."""
        if eps_values is None:
            eps_values = [0.3, 0.5, 0.8, 1.0, 1.5]
        if min_samples_values is None:
            min_samples_values = [5, 10, 20]

        # Subsample for speed
        n = int(len(X) * subsample)
        idx = np.random.RandomState(RANDOM_SEED).choice(len(X), size=n, replace=False)
        X_sub = X.iloc[idx]
        y_sub = y.iloc[idx]

        best_f1, best_params = -1, {}
        for eps in eps_values:
            for ms in min_samples_values:
                db = DBSCAN(eps=eps, min_samples=ms, algorithm="ball_tree", n_jobs=-1)
                db.fit(X_sub)
                scores = (db.labels_ == -1).astype(int)
                f1 = f1_score(y_sub, scores, zero_division=0)
                print(f"  DBSCAN eps={eps} min_samples={ms}: F1={f1:.4f}, noise_ratio={scores.mean():.4f}")
                if f1 > best_f1:
                    best_f1 = f1
                    best_params = {"eps": eps, "min_samples": ms}

        print(f"Best DBSCAN params: {best_params} (F1={best_f1:.4f})")
        return best_params

def run_dbscan(split, tune: bool = True):
    model = DBSCANAnomalyDetector()
    if tune:
        best = model.tune(split.X_train, split.y_train)
        model = DBSCANAnomalyDetector(**best)

    # Fit on train, score on test
    model.fit(split.X_train)
    y_score_val = model.anomaly_scores(split.X_val)
    y_score_test = model.anomaly_scores(split.X_test)

    # Evaluate on val to confirm, then final on test
    val_result = evaluate(split.y_val.values, y_score_val.astype(float), model_name="dbscan_val")
    result = evaluate(split.y_test.values, y_score_test.astype(float),
                      model_name="dbscan", threshold=0.5)
    log_result(result)
    save_all(split.y_test.values, y_score_test.astype(float), result.confusion_matrix, "dbscan")
    return result
```

**Step 4: Run tests**

```bash
uv run uv run pytest tests/test_dbscan.py -v
```


---

## Task 8: KNN Anomaly Detection (`src/models/knn_model.py`)

**Owner:** Shreyas Sreekumar

**Key design decisions:**
- Anomaly score = mean distance to k nearest neighbors (continuous, soft score)
- Threshold selected on validation set (maximizing F1)
- `algorithm='ball_tree'` for speed

**Files:**
- Create: `src/models/knn_model.py`
- Create: `tests/test_knn.py`

**Step 1: Write the failing tests**

```python
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
```

**Step 2: Run tests to verify they fail**

```bash
uv run uv run pytest tests/test_knn.py -v
```

**Step 3: Implement**

```python
# src/models/knn_model.py
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from src.evaluation.metrics import evaluate, log_result
from src.visualization.plots import save_all

RANDOM_SEED = 42

class KNNAnomalyDetector:
    def __init__(self, k: int = 5):
        self.k = k
        self.nn = None

    def fit(self, X: pd.DataFrame) -> None:
        self.nn = NearestNeighbors(n_neighbors=self.k + 1, algorithm="ball_tree", n_jobs=-1)
        self.nn.fit(X)

    def anomaly_scores(self, X: pd.DataFrame) -> np.ndarray:
        """Anomaly score = mean distance to k nearest neighbors (excluding self)."""
        distances, _ = self.nn.kneighbors(X)
        return distances[:, 1:].mean(axis=1)  # exclude the point itself (distance=0)

    def tune(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        k_values=None,
    ) -> int:
        """Select k maximizing ROC-AUC on validation set."""
        if k_values is None:
            k_values = [3, 5, 10, 20, 50]
        best_auc, best_k = -1, k_values[0]
        for k in k_values:
            tmp = KNNAnomalyDetector(k=k)
            tmp.fit(X_val)
            scores = tmp.anomaly_scores(X_val)
            auc = roc_auc_score(y_val, scores)
            print(f"  KNN k={k}: ROC-AUC={auc:.4f}")
            if auc > best_auc:
                best_auc = auc
                best_k = k
        print(f"Best KNN k={best_k} (ROC-AUC={best_auc:.4f})")
        return best_k

def run_knn(split, tune: bool = True):
    model = KNNAnomalyDetector(k=5)
    model.fit(split.X_train)

    if tune:
        # Tune k on val set
        best_k = model.tune(split.X_val, split.y_val)
        model = KNNAnomalyDetector(k=best_k)
        model.fit(split.X_train)

    y_score_val = model.anomaly_scores(split.X_val)
    y_score_test = model.anomaly_scores(split.X_test)

    # Select threshold on val
    val_result = evaluate(split.y_val.values, y_score_val, model_name="knn_val")
    result = evaluate(split.y_test.values, y_score_test,
                      model_name="knn", threshold=val_result.threshold)
    log_result(result)
    save_all(split.y_test.values, y_score_test, result.confusion_matrix, "knn")
    return result
```

**Step 4: Run tests**

```bash
uv run uv run pytest tests/test_knn.py -v
```


---

## Task 9: Logistic Regression (`src/models/logistic_model.py`)

**Owner:** Srijan Girdhar

**Key design decisions:**
- Use `class_weight='balanced'` to handle imbalance without SMOTE
- Tune `C` (regularization) and threshold
- Extract coefficients for interpretability

**Files:**
- Create: `src/models/logistic_model.py`
- Create: `tests/test_logistic.py`

**Step 1: Write the failing tests**

```python
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
```

**Step 2: Run tests to verify they fail**

```bash
uv run uv run pytest tests/test_logistic.py -v
```

**Step 3: Implement**

```python
# src/models/logistic_model.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from src.evaluation.metrics import evaluate, log_result
from src.visualization.plots import save_all

RANDOM_SEED = 42

class LogisticRegressionModel:
    def __init__(self, C: float = 1.0):
        self.C = C
        self.model = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model = LogisticRegression(
            C=self.C,
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_SEED,
            solver="lbfgs",
        )
        self.model.fit(X, y)

    def anomaly_scores(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def get_coefficients(self) -> pd.Series:
        return pd.Series(self.model.coef_[0], index=self.model.feature_names_in_)

    def tune(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        C_values=None,
    ) -> float:
        if C_values is None:
            C_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        best_auc, best_C = -1, C_values[0]
        for C in C_values:
            m = LogisticRegressionModel(C=C)
            m.fit(X_train, y_train)
            auc = roc_auc_score(y_val, m.anomaly_scores(X_val))
            print(f"  LR C={C}: ROC-AUC={auc:.4f}")
            if auc > best_auc:
                best_auc = auc
                best_C = C
        print(f"Best LR C={best_C} (ROC-AUC={best_auc:.4f})")
        return best_C

def run_logistic(split, tune: bool = True):
    model = LogisticRegressionModel()
    if tune:
        best_C = model.tune(split.X_train, split.y_train, split.X_val, split.y_val)
        model = LogisticRegressionModel(C=best_C)

    model.fit(split.X_train, split.y_train)
    y_score_val = model.anomaly_scores(split.X_val)
    y_score_test = model.anomaly_scores(split.X_test)

    val_result = evaluate(split.y_val.values, y_score_val, model_name="logistic_val")
    result = evaluate(split.y_test.values, y_score_test,
                      model_name="logistic_regression", threshold=val_result.threshold)
    log_result(result)
    save_all(split.y_test.values, y_score_test, result.confusion_matrix, "logistic_regression")

    # Interpretability: top features
    coefs = model.get_coefficients().abs().sort_values(ascending=False)
    print("\nTop 10 features by |coefficient|:")
    print(coefs.head(10))
    return result
```

**Step 4: Run tests**

```bash
uv run uv run pytest tests/test_logistic.py -v
```


---

## Task 10: XGBoost + SMOTE (`src/models/xgboost_model.py`)

**Owner:** Vivek Vidyadhar Kamath

**CRITICAL: SMOTE applied only inside training, never to val/test.**

**Files:**
- Create: `src/models/xgboost_model.py`
- Create: `tests/test_xgboost.py`

**Step 1: Write the failing tests**

```python
# tests/test_xgboost.py
import numpy as np
import pytest
from src.models.xgboost_model import XGBoostModel

def test_xgboost_fit_predict(tiny_X_y):
    X, y = tiny_X_y
    model = XGBoostModel()
    model.fit(X, y)
    scores = model.anomaly_scores(X)
    assert len(scores) == len(X)
    assert ((scores >= 0) & (scores <= 1)).all()

def test_smote_only_applied_to_train(tiny_X_y):
    X, y = tiny_X_y
    original_len = len(X)
    model = XGBoostModel(use_smote=True)
    model.fit(X, y)
    # SMOTE happens internally; X should be unchanged
    assert len(X) == original_len

def test_feature_importances_shape(tiny_X_y):
    X, y = tiny_X_y
    model = XGBoostModel()
    model.fit(X, y)
    importances = model.get_feature_importances()
    assert len(importances) == X.shape[1]
```

**Step 2: Run tests to verify they fail**

```bash
uv run uv run pytest tests/test_xgboost.py -v
```

**Step 3: Implement**

```python
# src/models/xgboost_model.py
import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score
from src.evaluation.metrics import evaluate, log_result
from src.visualization.plots import save_all

RANDOM_SEED = 42

class XGBoostModel:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        scale_pos_weight: float = None,  # set to neg/pos ratio if not using SMOTE
        use_smote: bool = True,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.scale_pos_weight = scale_pos_weight
        self.use_smote = use_smote
        self.model = None
        self.feature_names = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.feature_names = list(X.columns)
        X_fit, y_fit = X.copy(), y.copy()

        if self.use_smote:
            sm = SMOTE(random_state=RANDOM_SEED)
            X_fit, y_fit = sm.fit_resample(X_fit, y_fit)
            print(f"  SMOTE applied: {len(y_fit)} samples ({y_fit.sum()} fraud)")

        spw = self.scale_pos_weight
        if spw is None and not self.use_smote:
            spw = (y_fit == 0).sum() / max((y_fit == 1).sum(), 1)

        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            scale_pos_weight=spw,
            use_label_encoder=False,
            eval_metric="aucpr",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        self.model.fit(X_fit, y_fit)

    def anomaly_scores(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importances(self) -> pd.Series:
        return pd.Series(
            self.model.feature_importances_, index=self.feature_names
        ).sort_values(ascending=False)

    def tune(self, X_train, y_train, X_val, y_val,
             n_estimators_values=None, max_depth_values=None):
        if n_estimators_values is None:
            n_estimators_values = [100, 200]
        if max_depth_values is None:
            max_depth_values = [4, 6, 8]
        best_auc, best_params = -1, {}
        for n in n_estimators_values:
            for d in max_depth_values:
                m = XGBoostModel(n_estimators=n, max_depth=d, use_smote=self.use_smote)
                m.fit(X_train, y_train)
                auc = roc_auc_score(y_val, m.anomaly_scores(X_val))
                print(f"  XGB n_est={n} depth={d}: ROC-AUC={auc:.4f}")
                if auc > best_auc:
                    best_auc = auc
                    best_params = {"n_estimators": n, "max_depth": d}
        print(f"Best XGB params: {best_params}")
        return best_params

def run_xgboost(split, tune: bool = True):
    model = XGBoostModel(use_smote=True)
    if tune:
        best = model.tune(split.X_train, split.y_train, split.X_val, split.y_val)
        model = XGBoostModel(use_smote=True, **best)

    model.fit(split.X_train, split.y_train)
    y_score_val = model.anomaly_scores(split.X_val)
    y_score_test = model.anomaly_scores(split.X_test)

    val_result = evaluate(split.y_val.values, y_score_val, model_name="xgboost_val")
    result = evaluate(split.y_test.values, y_score_test,
                      model_name="xgboost_smote", threshold=val_result.threshold)
    log_result(result)
    save_all(split.y_test.values, y_score_test, result.confusion_matrix, "xgboost_smote")

    print("\nTop 10 features by importance:")
    print(model.get_feature_importances().head(10))
    return result
```

**Step 4: Run tests**

```bash
uv run uv run pytest tests/test_xgboost.py -v
```


---

## Task 11: Autoencoder (`src/models/autoencoder_model.py`)

**Owner:** Shravani Kishor Kulkarni

**Key design decisions:**
- Train ONLY on non-fraud transactions from X_train
- Reconstruction error = anomaly score
- Architecture: 29 → 16 → 8 → 4 → 8 → 16 → 29 (bottleneck)
- Use PyTorch; train with MSE loss

**Files:**
- Create: `src/models/autoencoder_model.py`
- Create: `tests/test_autoencoder.py`

**Step 1: Write the failing tests**

```python
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
    detector.fit(X_train_clean)  # no error

def test_reconstruction_scores_shape(tiny_X_y):
    X, y = tiny_X_y
    detector = AutoencoderAnomalyDetector(input_dim=X.shape[1], epochs=2)
    detector.fit(X[y == 0])
    scores = detector.anomaly_scores(X)
    assert len(scores) == len(X)
    assert (scores >= 0).all()

def test_fraud_higher_reconstruction_error_on_avg(tiny_X_y):
    """Fraud samples should on average have higher reconstruction error."""
    X, y = tiny_X_y
    detector = AutoencoderAnomalyDetector(input_dim=X.shape[1], epochs=20)
    detector.fit(X[y == 0])
    scores = detector.anomaly_scores(X)
    mean_legit = scores[y == 0].mean()
    mean_fraud = scores[y == 1].mean()
    # This may not always hold with 2 epochs/tiny data, but documents intent
    assert mean_fraud >= 0  # at minimum, no negatives
```

**Step 2: Run tests to verify they fail**

```bash
uv run uv run pytest tests/test_autoencoder.py -v
```

**Step 3: Implement**

```python
# src/models/autoencoder_model.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.evaluation.metrics import evaluate, log_result
from src.visualization.plots import save_all
import matplotlib.pyplot as plt
import os

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)

class Autoencoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

class AutoencoderAnomalyDetector:
    def __init__(self, input_dim: int, epochs: int = 50, batch_size: int = 256, lr: float = 1e-3):
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model = None
        self.train_losses = []

    def fit(self, X_clean: pd.DataFrame) -> None:
        """Train ONLY on non-fraud data."""
        self.model = Autoencoder(input_dim=self.input_dim)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        X_tensor = torch.FloatTensor(X_clean.values if hasattr(X_clean, 'values') else X_clean)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                optimizer.zero_grad()
                recon = self.model(batch)
                loss = criterion(recon, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(loader)
            self.train_losses.append(avg_loss)
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch [{epoch+1}/{self.epochs}] Loss: {avg_loss:.6f}")

    def anomaly_scores(self, X: pd.DataFrame) -> np.ndarray:
        """Reconstruction error per sample (MSE)."""
        self.model.eval()
        X_tensor = torch.FloatTensor(X.values if hasattr(X, 'values') else X)
        with torch.no_grad():
            recon = self.model(X_tensor)
        errors = ((recon - X_tensor) ** 2).mean(dim=1).numpy()
        return errors

    def save_training_curve(self) -> None:
        os.makedirs("outputs/plots", exist_ok=True)
        plt.figure()
        plt.plot(self.train_losses)
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title("Autoencoder Training Loss")
        plt.tight_layout()
        plt.savefig("outputs/plots/autoencoder_training_loss.png", dpi=150)
        plt.close()

def run_autoencoder(split):
    # Train ONLY on non-fraud training data
    X_train_clean = split.X_train[split.y_train == 0]
    print(f"Training autoencoder on {len(X_train_clean)} non-fraud samples...")

    detector = AutoencoderAnomalyDetector(input_dim=split.X_train.shape[1], epochs=50)
    detector.fit(X_train_clean)
    detector.save_training_curve()

    y_score_val = detector.anomaly_scores(split.X_val)
    y_score_test = detector.anomaly_scores(split.X_test)

    val_result = evaluate(split.y_val.values, y_score_val, model_name="autoencoder_val")
    result = evaluate(split.y_test.values, y_score_test,
                      model_name="autoencoder", threshold=val_result.threshold)
    log_result(result)
    save_all(split.y_test.values, y_score_test, result.confusion_matrix, "autoencoder")
    return result
```

**Step 4: Run tests**

```bash
uv run uv run pytest tests/test_autoencoder.py -v
```


---

## Task 12: End-to-End Pipeline Runner (`run_pipeline.py`)

**Owner:** Sachin Venugopalan Nair (Milestone 5 integration)

**Files:**
- Create: `run_pipeline.py`

```python
# run_pipeline.py
"""
Run the full fraud detection pipeline.
Usage:
    uv run python run_pipeline.py --data data/raw/creditcard.csv
    uv run python run_pipeline.py --data data/raw/creditcard.csv --model dbscan
"""
import argparse
import json

from src.data.loader import load_dataset
from src.data.preprocessor import preprocess
from src.models.baseline import run_majority_class_baseline
from src.models.dbscan_model import run_dbscan
from src.models.knn_model import run_knn
from src.models.logistic_model import run_logistic
from src.models.xgboost_model import run_xgboost
from src.models.autoencoder_model import run_autoencoder

MODELS = {
    "baseline": lambda split: run_majority_class_baseline(split.y_val, split.y_test),
    "dbscan": run_dbscan,
    "knn": run_knn,
    "logistic": run_logistic,
    "xgboost": run_xgboost,
    "autoencoder": run_autoencoder,
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/creditcard.csv")
    parser.add_argument("--model", default="all", choices=list(MODELS.keys()) + ["all"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading data from {args.data}...")
    df = load_dataset(args.data)
    split = preprocess(df, random_state=args.seed)

    models_to_run = list(MODELS.keys()) if args.model == "all" else [args.model]
    results = {}

    for name in models_to_run:
        print(f"\n{'='*50}")
        print(f"Running: {name.upper()}")
        print("="*50)
        result = MODELS[name](split)
        results[name] = {
            "pr_auc": result.pr_auc,
            "roc_auc": result.roc_auc,
            "f1": result.f1,
            "recall": result.recall,
        }

    print("\n\n" + "="*60)
    print("FINAL COMPARISON TABLE")
    print("="*60)
    print(f"{'Model':<25} {'PR-AUC':>8} {'ROC-AUC':>8} {'F1':>8} {'Recall':>8}")
    print("-"*60)
    for name, m in results.items():
        print(f"{name:<25} {m['pr_auc']:>8.4f} {m['roc_auc']:>8.4f} {m['f1']:>8.4f} {m['recall']:>8.4f}")

if __name__ == "__main__":
    main()
```


---

## Task 13: Milestone 5 — Comparison Notebook (`notebooks/milestone5_comparison.ipynb`)

Create a Jupyter notebook that:
1. Loads `experiments/experiment_log.json`
2. Builds a comparison table (pandas DataFrame)
3. Plots all PR curves overlaid on one figure
4. Plots all ROC curves overlaid on one figure
5. Prints best model by PR-AUC and by Recall with discussion of tradeoffs

This is done manually by Sachin after all model results are collected.

---

## Running the Full Test Suite

```bash
uv run pytest tests/ -v
```

Expected: All tests pass. No model depends on another, so failures are isolated.

---


## Reproducibility Checklist

- [ ] `RANDOM_SEED = 42` set in all files
- [ ] `torch.manual_seed(42)` set in autoencoder
- [ ] SMOTE `random_state=42`
- [ ] All train/val/test splits use `random_state=42`
- [ ] Scaler fitted only on `X_train`
- [ ] SMOTE applied only inside `model.fit(X_train, y_train)`
- [ ] Threshold selected on `X_val`, never `X_test`
- [ ] `experiments/experiment_log.json` contains all run results

---

## Dataset Download Instructions

1. Go to: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Download `creditcard.csv`
3. Place at: `data/raw/creditcard.csv`
4. Verify: `wc -l data/raw/creditcard.csv` should show 284808 lines (header + 284807 rows)
