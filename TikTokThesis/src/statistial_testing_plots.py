"""
statistical_testing_plots.py
Visualizes the results of statistical hypothesis testing between the Baseline (Random Forest)
and Hybrid (Random Forest + GNN) models.

Includes:
 - Q–Q plot and histogram for Kolmogorov–Smirnov normality test
 - Paired line plot and boxplot for Paired T-test
 - Bar chart of n01 vs n10 for McNemar’s test
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from pathlib import Path
from utils import RESULTS_DIR, PLOTS_DIR

GRAPHS_DIR = PLOTS_DIR / "hypothesis_testing_plots"
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

# Load metrics and predictions
baseline_metrics = pd.read_csv(RESULTS_DIR / "baseline_rf_metrics.csv")
baseline_metrics = baseline_metrics[baseline_metrics["fold"] != "mean"]

hybrid_metrics = pd.read_csv(RESULTS_DIR / "hybrid_fold_metrics_calibrated.csv")
hybrid_metrics = hybrid_metrics[hybrid_metrics["fold"] != "mean"]
base_pred = pd.read_csv(RESULTS_DIR / "baseline_predictions_full.csv")
hyb_pred = pd.read_csv(RESULTS_DIR / "hybrid_predictions_full.csv")


# Ensure numerical fold metrics only
baseline = baseline_metrics[baseline_metrics["fold"].astype(str).str.isnumeric()]
hybrid = hybrid_metrics[hybrid_metrics["fold"].astype(str).str.isnumeric()]

# === 1️⃣ K-S Test Visualizations ===

diff = hybrid["test_acc"].values - baseline["test_acc"].values

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
sns.histplot(diff, kde=True, color="skyblue", bins=10)
x = np.linspace(min(diff), max(diff), 100)
plt.title("Histogram of Accuracy Differences (Hybrid - Baseline)")
plt.xlabel("Accuracy Difference")

plt.subplot(1,2,2)
stats.probplot(diff, dist="norm", plot=plt)
plt.title("Q–Q Plot for Normality (K-S Test)")
plt.tight_layout()
plt.savefig(GRAPHS_DIR / "ks_test_hist_qq.png", dpi=300)
plt.close()

# === 2️⃣ Paired T-test Visualization ===

plt.figure(figsize=(8,5))
plt.plot(baseline_metrics["fold"], baseline_metrics["test_acc"], marker='o', label="Baseline (RF)")
plt.plot(hybrid_metrics["fold"], hybrid_metrics["test_acc"], marker='o', label="Hybrid (RF + GNN)")
plt.xlabel("Fold")
plt.ylabel("Test Accuracy")
plt.title("Fold-wise Accuracy Comparison (Paired t-test)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(GRAPHS_DIR / "t_test_paired_line.png", dpi=300)
plt.close()

# # Boxplot of accuracy differences
# plt.figure(figsize=(5,5))
# sns.boxplot(y=diff, color="lightgreen")
# plt.axhline(0, color='red', linestyle='--', label="No Difference")
# plt.title("Distribution of Accuracy Differences (Hybrid - Baseline)")
# plt.ylabel("Accuracy Difference")
# plt.legend()
# plt.tight_layout()
# plt.show()

# print("✅ Displayed paired accuracy comparisons and difference distribution.")

# === 3️⃣ McNemar’s Test Visualization ===

y_true = hyb_pred["annotation"].map({"real": 0, "fake": 1}).values
y_pred_h = hyb_pred["predicted_label_calibrated"].map({"real": 0, "fake": 1}).values
y_pred_b = base_pred["predicted_label_baseline"].map({"real": 0, "fake": 1}).values

correct_baseline = (y_true == y_pred_b).astype(int)
correct_hybrid = (y_true == y_pred_h).astype(int)

n01 = ((correct_baseline == 1) & (correct_hybrid == 0)).sum()
n10 = ((correct_baseline == 0) & (correct_hybrid == 1)).sum()

plt.figure(figsize=(6,5))
sns.barplot(x=["Baseline Correct / Hybrid Wrong", "Baseline Wrong / Hybrid Correct"], y=[n01, n10], palette="pastel")
plt.title("McNemar’s Test Discordant Cases")
plt.ylabel("Count of Instances")
plt.tight_layout()
plt.savefig(GRAPHS_DIR / "mcnemar_barplot.png", dpi=300)
plt.close()

print(f"✅ Displayed McNemar’s discordant counts (n01={n01}, n10={n10}).")
print("Visualization complete for all hypothesis and assumption tests.")
