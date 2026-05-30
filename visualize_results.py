"""
Visualize and compare all model results from experiment_log.json.
Usage: uv run python visualize_results.py
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import os

# ── Load & deduplicate results ────────────────────────────────────────────────
with open("experiments/experiment_log.json") as f:
    raw = json.load(f)

# Keep last entry per model (exclude _val variants)
seen, results = {}, []
for r in raw:
    name = r["model_name"]
    if name.endswith("_val"):
        continue
    seen[name] = r
for r in seen.values():
    results.append(r)

# Display order
ORDER = ["majority_class_baseline", "dbscan", "knn",
         "logistic_regression", "xgboost_smote", "autoencoder"]
LABELS = ["Baseline", "DBSCAN", "KNN", "Logistic\nRegression", "XGBoost\n+SMOTE", "Autoencoder"]
results = sorted(results, key=lambda r: ORDER.index(r["model_name"]) if r["model_name"] in ORDER else 99)

models  = [r["model_name"] for r in results]
display = [LABELS[ORDER.index(m)] if m in ORDER else m for m in models]
pr_auc  = [r["pr_auc"]  for r in results]
roc_auc = [r["roc_auc"] for r in results]
f1      = [r["f1"]      for r in results]
recall  = [r["recall"]  for r in results]
cms     = [r["confusion_matrix"] for r in results]

n = len(results)
best_idx = pr_auc.index(max(pr_auc))  # primary metric for fraud detection

# ── Colors ────────────────────────────────────────────────────────────────────
COLORS = ["#b0b8c1", "#e07b54", "#f2c45a", "#5b8dd9", "#2ecc71", "#9b59b6"]
HIGHLIGHT = "#2ecc71"  # XGBoost green

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 22))
fig.patch.set_facecolor("#0f1117")
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35,
                       left=0.07, right=0.97, top=0.93, bottom=0.05)

TICK_COLOR = "#c9d1d9"
GRID_COLOR = "#21262d"
TITLE_COLOR = "#ffffff"
LABEL_COLOR = "#8b949e"

def style_ax(ax, title):
    ax.set_facecolor("#161b22")
    ax.tick_params(colors=TICK_COLOR, labelsize=9)
    ax.xaxis.label.set_color(LABEL_COLOR)
    ax.yaxis.label.set_color(LABEL_COLOR)
    ax.title.set_color(TITLE_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.7)
    ax.set_axisbelow(True)

# ── 1. Grouped metric bars ────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
style_ax(ax1, "Model Performance Comparison — All Metrics")

x = np.arange(n)
width = 0.2
metrics_data = [pr_auc, roc_auc, f1, recall]
metric_labels = ["PR-AUC", "ROC-AUC", "F1", "Recall"]
metric_colors = ["#5b8dd9", "#e07b54", "#2ecc71", "#f2c45a"]
offsets = [-1.5, -0.5, 0.5, 1.5]

for i, (data, label, color, offset) in enumerate(zip(metrics_data, metric_labels, metric_colors, offsets)):
    bars = ax1.bar(x + offset * width, data, width, label=label,
                   color=color, alpha=0.85, zorder=3)
    for j, (bar, val) in enumerate(zip(bars, data)):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=7,
                 color=TICK_COLOR, fontweight="bold" if j == best_idx else "normal")

# Highlight best model column
ax1.axvspan(best_idx - 0.55, best_idx + 0.55, color=HIGHLIGHT, alpha=0.06, zorder=0)
ax1.text(best_idx, 1.05, "★ BEST", ha="center", va="bottom", fontsize=9,
         color=HIGHLIGHT, fontweight="bold")

ax1.set_xticks(x)
ax1.set_xticklabels(display, fontsize=10)
ax1.set_ylim(0, 1.12)
ax1.set_ylabel("Score", color=LABEL_COLOR)
ax1.legend(loc="upper left", framealpha=0.2, labelcolor=TICK_COLOR,
           facecolor="#161b22", edgecolor=GRID_COLOR, fontsize=9)

# ── 2. PR-AUC bar (primary metric) ───────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
style_ax(ax2, "PR-AUC  (Primary Metric for Imbalanced Data)")

bar_colors = [HIGHLIGHT if i == best_idx else COLORS[i] for i in range(n)]
bars = ax2.barh(display, pr_auc, color=bar_colors, alpha=0.88, zorder=3)
for bar, val in zip(bars, pr_auc):
    ax2.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
             f"{val:.4f}", va="center", fontsize=9, color=TICK_COLOR, fontweight="bold")
ax2.set_xlim(0, 1.0)
ax2.set_xlabel("PR-AUC", color=LABEL_COLOR)
ax2.invert_yaxis()

# ── 3. ROC-AUC bar ───────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
style_ax(ax3, "ROC-AUC")

bar_colors3 = [HIGHLIGHT if i == best_idx else COLORS[i] for i in range(n)]
bars3 = ax3.barh(display, roc_auc, color=bar_colors3, alpha=0.88, zorder=3)
for bar, val in zip(bars3, roc_auc):
    ax3.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
             f"{val:.4f}", va="center", fontsize=9, color=TICK_COLOR, fontweight="bold")
ax3.set_xlim(0, 1.05)
ax3.set_xlabel("ROC-AUC", color=LABEL_COLOR)
ax3.invert_yaxis()

# ── 4. Confusion matrices ─────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[2, 0])
style_ax(ax4, "F1 Score vs Recall  (Threshold Trade-off)")

scatter_colors = [HIGHLIGHT if i == best_idx else COLORS[i] for i in range(n)]
for i, (f, r, label, color) in enumerate(zip(f1, recall, display, scatter_colors)):
    ax4.scatter(r, f, s=220, color=color, zorder=5, edgecolors="white", linewidths=0.8)
    ax4.annotate(label.replace("\n", " "), (r, f),
                 textcoords="offset points", xytext=(8, 4),
                 fontsize=8, color=TICK_COLOR)
ax4.set_xlabel("Recall", color=LABEL_COLOR)
ax4.set_ylabel("F1 Score", color=LABEL_COLOR)
ax4.set_xlim(-0.05, 1.15)
ax4.set_ylim(-0.05, 1.0)
ax4.axhline(0, color=GRID_COLOR, linewidth=0.5)
ax4.axvline(0, color=GRID_COLOR, linewidth=0.5)

# ── 5. Summary table ─────────────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 1])
ax5.set_facecolor("#161b22")
ax5.set_xlim(0, 1)
ax5.set_ylim(0, 1)
ax5.axis("off")
ax5.set_title("Results Summary", fontsize=12, fontweight="bold",
              color=TITLE_COLOR, pad=10)

headers = ["Model", "PR-AUC", "ROC-AUC", "F1", "Recall"]
col_x   = [0.0, 0.38, 0.55, 0.70, 0.84]
row_h   = 0.083
y_start = 0.93

# Header row
for hdr, cx in zip(headers, col_x):
    ax5.text(cx, y_start, hdr, fontsize=8.5, fontweight="bold",
             color="#ffffff", va="top", transform=ax5.transAxes)
ax5.plot([0, 1], [y_start - 0.03, y_start - 0.03], color=GRID_COLOR,
         linewidth=0.8, transform=ax5.transAxes)

for i, r in enumerate(results):
    y = y_start - row_h * (i + 1) - 0.01
    is_best = i == best_idx
    row_color = "#1a2e1a" if is_best else "#161b22"
    ax5.add_patch(FancyBboxPatch((0, y - 0.01), 1, row_h - 0.005,
                                  boxstyle="round,pad=0.005",
                                  facecolor=row_color, edgecolor="none",
                                  transform=ax5.transAxes, clip_on=False))
    label = display[i].replace("\n", " ")
    if is_best:
        label = "★ " + label
    vals = [label, f"{r['pr_auc']:.4f}", f"{r['roc_auc']:.4f}",
            f"{r['f1']:.4f}", f"{r['recall']:.4f}"]
    for val, cx in zip(vals, col_x):
        ax5.text(cx, y + row_h / 2 - 0.015, val, fontsize=8,
                 color=HIGHLIGHT if is_best else TICK_COLOR,
                 fontweight="bold" if is_best else "normal",
                 va="center", transform=ax5.transAxes)

# ── Title ─────────────────────────────────────────────────────────────────────
fig.text(0.5, 0.965, "Credit Card Fraud Detection — Model Comparison",
         ha="center", fontsize=16, fontweight="bold", color=TITLE_COLOR)
fig.text(0.5, 0.952, "Kaggle Credit Card Fraud Dataset  |  284,807 transactions  |  492 fraud (0.173%)",
         ha="center", fontsize=10, color=LABEL_COLOR)

os.makedirs("outputs/plots", exist_ok=True)
out = "outputs/plots/model_comparison.png"
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved: {out}")

# ── Print recommendation ──────────────────────────────────────────────────────
best = results[best_idx]
print(f"""
╔══════════════════════════════════════════════════════╗
║           RECOMMENDED MODEL: XGBoost + SMOTE         ║
╠══════════════════════════════════════════════════════╣
║  PR-AUC  : {best['pr_auc']:.4f}  (best — primary metric)         ║
║  ROC-AUC : {best['roc_auc']:.4f}  (best)                         ║
║  F1      : {best['f1']:.4f}  (best)                         ║
║  Recall  : {best['recall']:.4f}  (best)                         ║
║  FP      : {best['confusion_matrix'][0][1]:<5}   FN: {best['confusion_matrix'][1][0]:<5}                    ║
╚══════════════════════════════════════════════════════╝
""")

print("Model ranking by PR-AUC (most important for imbalanced fraud detection):")
ranked = sorted(results, key=lambda r: r["pr_auc"], reverse=True)
for rank, r in enumerate(ranked, 1):
    label = display[ORDER.index(r["model_name"])].replace("\n", " ") if r["model_name"] in ORDER else r["model_name"]
    print(f"  {rank}. {label:<25}  PR-AUC={r['pr_auc']:.4f}  ROC-AUC={r['roc_auc']:.4f}  F1={r['f1']:.4f}  Recall={r['recall']:.4f}")
