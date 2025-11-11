"""
baseline_hybrid_comparison_plots.py
Generates visual comparisons between the Baseline (Random Forest) and Hybrid (Random Forest + GNN) models.
Includes:
 - Bar chart of mean metrics
 - Fold-wise accuracy line plot
 - Box plot of accuracy distributions
 - Confusion matrix comparison
 - Optional Bland–Altman agreement plot
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path
from utils import RESULTS_DIR, PLOTS_DIR

GRAPHS_DIR = PLOTS_DIR / "hybrid_model_plots"
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

# === Load results ===
hybrid_metrics = pd.read_csv(RESULTS_DIR / "hybrid_fold_metrics.csv")
hybrid_metrics = hybrid_metrics[hybrid_metrics["fold"] != "mean"]
baseline_metrics = pd.read_csv(RESULTS_DIR / "baseline_rf_metrics.csv")
baseline_metrics = baseline_metrics[baseline_metrics["fold"] != "mean"]

# === 1️⃣ Bar chart of mean performance ===
metrics = ["Accuracy", "F1-score"]
base_means = [baseline_metrics["test_acc"].mean(), baseline_metrics["test_f1"].mean()]
hyb_means = [hybrid_metrics["test_acc"].mean(), hybrid_metrics["test_f1"].mean()]

x = np.arange(len(metrics))
width = 0.35
plt.figure(figsize=(6,5))
plt.bar(x - width/2, base_means, width, label="Baseline (RF)")
plt.bar(x + width/2, hyb_means, width, label="Hybrid (RF + GNN)")
plt.xticks(x, metrics)
plt.ylabel("Score")
plt.ylim(0.6, 1.0)
plt.title("Mean Model Performance Comparison")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(GRAPHS_DIR / "bar_chart_mean_performance.png", dpi=300)
plt.close()

# # === 2️⃣ Line plot of per-fold accuracy === USED FOR PAIRED T-TEST
# plt.figure(figsize=(8,5))
# plt.plot(baseline_metrics["fold"], baseline_metrics["test_acc"], marker='o', label="Baseline (RF)")
# plt.plot(hybrid_metrics["fold"], hybrid_metrics["test_acc"], marker='o', label="Hybrid (RF + GNN)")
# plt.xlabel("Fold")
# plt.ylabel("Test Accuracy")
# plt.title("Fold-wise Accuracy Comparison")
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.tight_layout()
# plt.show()

# === 3️⃣ Box plot of accuracy distribution ===
data = pd.DataFrame({
    "Accuracy": np.concatenate([baseline_metrics["test_acc"], hybrid_metrics["test_acc"]]),
    "Model": ["Baseline (RF)"] * len(baseline_metrics) + ["Hybrid (RF + GNN)"] * len(hybrid_metrics)
})

plt.figure(figsize=(6,5))
sns.boxplot(x="Model", y="Accuracy", data=data, palette="pastel")
sns.swarmplot(x="Model", y="Accuracy", data=data, color=".25", alpha=0.6)
plt.title("Distribution of Fold Accuracies")
plt.tight_layout()
plt.savefig(GRAPHS_DIR / "box_plot_accuracy_dist.png", dpi=300)
plt.close()

# === 4️⃣ Confusion Matrix Comparison ===
try:
    base_pred = pd.read_csv(RESULTS_DIR / "baseline_predictions_full.csv")
    hyb_pred = pd.read_csv(RESULTS_DIR / "hybrid_predictions_full.csv")

    y_true = hyb_pred["annotation"].map({"real": 0, "fake": 1}).values
    y_pred_base = base_pred["predicted_label_baseline"].map({"real": 0, "fake": 1}).values
    y_pred_hyb = hyb_pred["predicted_label_calibrated"].map({"real": 0, "fake": 1}).values

    cm_base = confusion_matrix(y_true, y_pred_base)
    cm_hyb = confusion_matrix(y_true, y_pred_hyb)

    fig, axs = plt.subplots(1, 2, figsize=(10,4))
    ConfusionMatrixDisplay(cm_base, display_labels=["Real", "Fake"]).plot(ax=axs[0], cmap="Blues", colorbar=False)
    axs[0].set_title("Baseline (RF)")
    ConfusionMatrixDisplay(cm_hyb, display_labels=["Real", "Fake"]).plot(ax=axs[1], cmap="Greens", colorbar=False)
    axs[1].set_title("Hybrid (RF + GNN)")
    plt.suptitle("Confusion Matrix Comparison")
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "cm_baseline_hybrid_comp.png", dpi=300)
    plt.close()
except Exception as e:
    print(f"⚠️ Skipping confusion matrices: {e}")

# === 5️⃣ Bland–Altman Plot (optional) ===
diff = hybrid_metrics["test_acc"].values - baseline_metrics["test_acc"].values
mean = (hybrid_metrics["test_acc"].values + baseline_metrics["test_acc"].values) / 2

plt.figure(figsize=(6,5))
plt.scatter(mean, diff, color="teal", alpha=0.7)
plt.axhline(np.mean(diff), color='red', linestyle='--', label=f"Mean Diff = {np.mean(diff):.4f}")
plt.axhline(np.mean(diff) + 1.96*np.std(diff), color='gray', linestyle=':')
plt.axhline(np.mean(diff) - 1.96*np.std(diff), color='gray', linestyle=':')
plt.xlabel("Mean Accuracy (Baseline + Hybrid)")
plt.ylabel("Accuracy Difference (Hybrid - Baseline)")
plt.title("Bland–Altman Plot: Accuracy Agreement")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(GRAPHS_DIR / "bland_altman.png", dpi=300)
plt.close()

# SUMMARY TABLE
# Extract comparable metrics
metrics = ["test_acc", "test_prec", "test_rec", "test_f1"]
baseline_means = baseline_metrics[metrics].mean()
hybrid_means = hybrid_metrics[metrics].mean()

# Combine for summary table
summary_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
    "Baseline": baseline_means.values,
    "Hybrid": hybrid_means.values
})

summary_df.to_csv(GRAPHS_DIR / "model_comparison_summary.csv", index=False)

# ===============================
# 3️⃣ RADAR CHART - Overall Metric Shape
# ===============================
labels = list(summary_df["Metric"])
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

baseline_values = summary_df["Baseline"].tolist()
hybrid_values = summary_df["Hybrid"].tolist()
baseline_values += baseline_values[:1]
hybrid_values += hybrid_values[:1]

# ===============================
# 4️⃣ COMBINED FIGURE (Thesis Layout)
# ===============================
fig, axs = plt.subplots(2, 2, figsize=(12,10))

# Recompute x for 4 metrics (Accuracy, Precision, Recall, F1)
x = np.arange(len(summary_df["Metric"]))
width = 0.35

# Bar Chart
axs[0,0].bar(x - width/2, summary_df["Baseline"], width, label="Baseline (RF)", color="#4C72B0")
axs[0,0].bar(x + width/2, summary_df["Hybrid"], width, label="Hybrid (RF + GNN)", color="#DD8452")
axs[0,0].set_xticks(x)
axs[0,0].set_xticklabels(summary_df["Metric"])
axs[0,0].set_title("Mean Metrics")
axs[0,0].legend()
axs[0,0].grid(alpha=0.3, linestyle='--')

# Fold-wise Accuracy
axs[0,1].plot(baseline_metrics["fold"], baseline_metrics["test_acc"], marker='o', label="Baseline", color="#4C72B0")
axs[0,1].plot(hybrid_metrics["fold"], hybrid_metrics["test_acc"], marker='o', label="Hybrid", color="#DD8452")
axs[0,1].set_title("Fold-wise Accuracy")
axs[0,1].set_xlabel("Fold")
axs[0,1].set_ylabel("Accuracy")
axs[0,1].legend()
axs[0,1].grid(alpha=0.3, linestyle='--')

# Fold-wise F1
axs[1,0].plot(baseline_metrics["fold"], baseline_metrics["test_f1"], marker='x', linestyle='--', label="Baseline F1", color="#6A5ACD")
axs[1,0].plot(hybrid_metrics["fold"], hybrid_metrics["test_f1"], marker='x', linestyle='--', label="Hybrid F1", color="#FF7F0E")
axs[1,0].set_title("Fold-wise F1-score")
axs[1,0].set_xlabel("Fold")
axs[1,0].set_ylabel("F1-score")
axs[1,0].legend()
axs[1,0].grid(alpha=0.3, linestyle='--')

# Radar Chart
ax = plt.subplot(2, 2, 4, polar=True)
ax.plot(angles, baseline_values, color="#4C72B0", linewidth=2)
ax.fill(angles, baseline_values, color="#4C72B0", alpha=0.25)
ax.plot(angles, hybrid_values, color="#DD8452", linewidth=2)
ax.fill(angles, hybrid_values, color="#DD8452", alpha=0.25)
plt.xticks(angles[:-1], labels)
ax.set_title("Radar Plot – Model Comparison", y=1.1)

plt.suptitle("Baseline (RF) vs Hybrid (RF + GNN) Model Performance", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(GRAPHS_DIR / "combined_model_comparison.png", dpi=300)
plt.close()

print("✅ Visualization complete. All comparison plots generated successfully.")
