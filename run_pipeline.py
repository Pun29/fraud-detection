# run_pipeline.py
"""
End-to-end fraud detection pipeline.

Usage:
    uv run python run_pipeline.py --data data/raw/creditcard.csv
    uv run python run_pipeline.py --data data/raw/creditcard.csv --model dbscan
    uv run python run_pipeline.py --data data/raw/creditcard.csv --model all --no-tune
"""
import argparse
import os

from src.data.loader import load_dataset
from src.data.preprocessor import preprocess
from src.models.baseline import run_majority_class_baseline
from src.models.dbscan_model import run_dbscan
from src.models.knn_model import run_knn
from src.models.logistic_model import run_logistic
from src.models.xgboost_model import run_xgboost
from src.models.autoencoder_model import run_autoencoder

MODEL_NAMES = ["baseline", "dbscan", "knn", "logistic", "xgboost", "autoencoder"]

def run_model(name: str, split, tune: bool):
    if name == "baseline":
        return run_majority_class_baseline(split.y_val, split.y_test)
    elif name == "dbscan":
        return run_dbscan(split, tune=tune)
    elif name == "knn":
        return run_knn(split, tune=tune)
    elif name == "logistic":
        return run_logistic(split, tune=tune)
    elif name == "xgboost":
        return run_xgboost(split, tune=tune)
    elif name == "autoencoder":
        return run_autoencoder(split)
    else:
        raise ValueError(f"Unknown model: {name}")

def main():
    parser = argparse.ArgumentParser(description="Credit Card Fraud Detection Pipeline")
    parser.add_argument("--data", default="data/raw/creditcard.csv",
                        help="Path to creditcard.csv")
    parser.add_argument("--model", default="all",
                        choices=MODEL_NAMES + ["all"],
                        help="Which model to run (default: all)")
    parser.add_argument("--no-tune", action="store_true",
                        help="Skip hyperparameter tuning")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"ERROR: Dataset not found at {args.data}")
        print("Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print("Place at: data/raw/creditcard.csv")
        return

    tune = not args.no_tune
    models_to_run = MODEL_NAMES if args.model == "all" else [args.model]

    print(f"Loading data from {args.data}...")
    df = load_dataset(args.data)
    split = preprocess(df, random_state=args.seed)

    results = {}
    for name in models_to_run:
        print(f"\n{'='*55}")
        print(f"  Running: {name.upper()}")
        print("="*55)
        try:
            result = run_model(name, split, tune=tune)
            results[name] = result
        except Exception as e:
            print(f"ERROR running {name}: {e}")

    if results:
        print("\n\n" + "="*65)
        print("  FINAL COMPARISON TABLE")
        print("="*65)
        print(f"{'Model':<25} {'PR-AUC':>8} {'ROC-AUC':>8} {'F1':>8} {'Recall':>8}")
        print("-"*65)
        for name, r in results.items():
            print(f"{name:<25} {r.pr_auc:>8.4f} {r.roc_auc:>8.4f} {r.f1:>8.4f} {r.recall:>8.4f}")
        print("\nPlots saved to: outputs/plots/")
        print("Metrics logged to: experiments/experiment_log.json")

if __name__ == "__main__":
    main()
