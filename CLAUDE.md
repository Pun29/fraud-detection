# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package Management

Always use `uv` (never `pip`) for all Python operations:
- Install deps: `uv pip install -r requirements.txt`
- Run scripts: `uv run python <script>`
- Run tests: `uv run pytest`

## Common Commands

```bash
# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_xgboost.py -v

# Run the full pipeline (requires data/raw/creditcard.csv)
uv run python run_pipeline.py --data data/raw/creditcard.csv --model all

# Run a single model
uv run python run_pipeline.py --data data/raw/creditcard.csv --model xgboost

# Skip hyperparameter tuning (faster)
uv run python run_pipeline.py --data data/raw/creditcard.csv --model logistic --no-tune

# Generate comparison dashboard from logged results
uv run python visualize_results.py
```

## Architecture

The pipeline is:

1. **Load** (`src/data/loader.py`) → validate CSV has V1–V28, Time, Amount, Class columns
2. **Preprocess** (`src/data/preprocessor.py`) → drop Time, stratified 70/15/15 split, fit StandardScaler on train only
3. **Train & tune** (each model in `src/models/`) → grid search on val set, final eval on test set
4. **Evaluate** (`src/evaluation/metrics.py`) → PR-AUC (primary), ROC-AUC, threshold chosen on val to maximize F1
5. **Plot** (`src/visualization/plots.py`) + log to `experiments/experiment_log.json`

### Models

| Model | Type | Key detail |
|---|---|---|
| `baseline.py` | Supervised | Predicts all-zeros; sanity check |
| `logistic_model.py` | Supervised | `class_weight="balanced"`, tuned C |
| `xgboost_model.py` | Supervised | SMOTE applied **after** split, tuned depth/estimators |
| `dbscan_model.py` | Unsupervised | BallTree; tunes on 10% subsample for speed |
| `knn_model.py` | Unsupervised | Anomaly score = mean distance to k nearest neighbors |
| `autoencoder_model.py` | Deep learning | Trained on **non-fraud samples only**; anomaly score = reconstruction error |

### Leakage prevention rules

- StandardScaler is fit **only on train**, then applied to val/test
- SMOTE is applied **only to train** after the split
- Threshold selection uses val set; final metrics are reported on test set

### Key data classes

- `SplitResult` (preprocessor.py): `X_train, X_val, X_test, y_train, y_val, y_test, scaler`
- `EvalResult` (metrics.py): `model_name, pr_auc, roc_auc, f1, recall, threshold, confusion_matrix`

### Dataset

Place `creditcard.csv` at `data/raw/creditcard.csv`. It is the Kaggle Credit Card Fraud dataset (284,807 rows, 0.173% fraud rate, features: Time, Amount, V1–V28, Class).

### Outputs

- `outputs/plots/` — per-model PR curves, ROC curves, confusion matrices
- `experiments/experiment_log.json` — append-only results log (used by `visualize_results.py`)

## Tests

Tests use a synthetic 210-row fixture (200 legit, 10 fraud) defined in `tests/conftest.py`. No real dataset required for tests.
