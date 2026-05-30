# Anomaly Scope — Credit Card Fraud Detection

CSE 575 Statistical Machine Learning — Project 12

Compares six fraud detection approaches on the Kaggle Credit Card Fraud dataset: a majority-class baseline, DBSCAN, KNN anomaly detection, Logistic Regression, XGBoost + SMOTE, and a PyTorch Autoencoder.

## Results

| Model | PR-AUC | ROC-AUC | F1 | Recall |
|---|---|---|---|---|
| Majority Class Baseline | 0.5009 | 0.5000 | 0.0035 | 1.0000 |
| DBSCAN | 0.4820 | 0.7951 | 0.0089 | 0.9595 |
| KNN | 0.1034 | 0.9512 | 0.2192 | 0.4324 |
| Logistic Regression | 0.7212 | 0.9626 | 0.6230 | 0.5135 |
| **XGBoost + SMOTE** | **0.7791** | **0.9703** | **0.7692** | **0.7432** |
| Autoencoder | 0.2986 | 0.9367 | 0.3671 | 0.3919 |

---

## Setup

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) for package management

### Install dependencies

```bash
uv pip install -r requirements.txt
```

### Get the dataset

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it at:

```
data/raw/creditcard.csv
```

---

## Running the Pipeline

### Run all models

```bash
uv run python run_pipeline.py --data data/raw/creditcard.csv --model all
```

### Run a single model

```bash
uv run python run_pipeline.py --data data/raw/creditcard.csv --model xgboost
```

Available model names: `baseline`, `dbscan`, `knn`, `logistic`, `xgboost`, `autoencoder`

### Skip hyperparameter tuning (faster)

```bash
uv run python run_pipeline.py --data data/raw/creditcard.csv --model all --no-tune
```

### Set a custom random seed

```bash
uv run python run_pipeline.py --data data/raw/creditcard.csv --seed 123
```

### Generate the comparison dashboard

After running the pipeline, generate a visual comparison of all models:

```bash
uv run python visualize_results.py
```

---

## Outputs

| Path | Contents |
|---|---|
| `outputs/plots/` | PR curves, ROC curves, confusion matrices per model |
| `outputs/plots/model_comparison.png` | Side-by-side dashboard of all models |
| `outputs/plots/autoencoder_training_loss.png` | Autoencoder training loss curve |
| `experiments/experiment_log.json` | Append-only log of all run results |

---

## Running Tests

```bash
# Run all tests
uv run pytest

# Run a specific model's tests
uv run pytest tests/test_xgboost.py -v

# Run with coverage
uv run pytest --cov=src tests/
```

Tests use a synthetic 210-row fixture (no real dataset required).

---

## Project Structure

```
├── src/
│   ├── data/
│   │   ├── loader.py          # Dataset loading and validation
│   │   └── preprocessor.py    # Stratified split + StandardScaler
│   ├── models/
│   │   ├── baseline.py        # Majority class baseline
│   │   ├── dbscan_model.py    # DBSCAN anomaly detector
│   │   ├── knn_model.py       # KNN distance-based anomaly detector
│   │   ├── logistic_model.py  # Logistic Regression with class weighting
│   │   ├── xgboost_model.py   # XGBoost with SMOTE
│   │   └── autoencoder_model.py # PyTorch Autoencoder
│   ├── evaluation/
│   │   └── metrics.py         # Unified evaluation harness
│   └── visualization/
│       └── plots.py           # PR/ROC curves, confusion matrices
├── tests/                     # pytest suite
├── data/raw/                  # Place creditcard.csv here
├── experiments/               # Experiment logs
├── outputs/                   # Generated plots
├── run_pipeline.py            # Main entry point
└── visualize_results.py       # Dashboard generator
```

---

## Team

| Name | Role |
|---|---|
| Sachin Venugopalan Nair | Data pipeline, evaluation harness, pipeline integration |
| Punarva Bettadamane Channabasappa | DBSCAN |
| Shreyas Sreekumar | KNN anomaly detection |
| Srijan Girdhar | Logistic Regression |
| Vivek Vidyadhar Kamath | XGBoost + SMOTE |
| Shravani Kishor Kulkarni | Autoencoder |
