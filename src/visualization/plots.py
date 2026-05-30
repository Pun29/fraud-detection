# src/visualization/plots.py
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
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
