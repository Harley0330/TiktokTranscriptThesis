"""
Statistical Tests for Model Comparison
--------------------------------------
Performs:
1. Kolmogorov-Smirnov Test (normality)
2. Paired t-test or Wilcoxon signed-rank test (fold-level performance)
3. McNemar's test (sample-level predictions)
"""

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, ttest_rel, wilcoxon
from statsmodels.stats.contingency_tables import mcnemar
from pathlib import Path
from utils import RESULTS_DIR

def run_tests():
    print("Comparing Models")

    # Load metrics files

    hybrid_metrics = pd.read_csv(RESULTS_DIR / "hybrid_fold_metrics.csv")
    hybrid_metrics = hybrid_metrics[hybrid_metrics["fold"] != "mean"]

    baseline_metrics = pd.read_csv(RESULTS_DIR / "baseline_rf_metrics.csv")
    baseline_metrics = baseline_metrics[baseline_metrics["fold"] != "mean"]

    baseline_acc = baseline_metrics["test_acc"].values
    hybrid_acc = hybrid_metrics["test_acc"].values

    # Kolmogorov-Smirnov Test for normality
    print("\nKolmogorov-Smirnov Test")
    ks_stat, ks_p = ks_2samp(hybrid_acc,baseline_acc)
    normality = "Normal" if ks_p > 0.05 else "Non-Normal"
    print(f"K-S Statistic = {ks_stat:.4f}, p = {ks_p:.4f} -> {normality}")

    # Paired t-test
    print("\nPaired t-test for Normally Distributed Data")
    t_stat, t_p_two = ttest_rel(hybrid_acc,baseline_acc)
    test_used = "Paired t-test"
    print(f"{test_used} -> t = {t_stat:.4f}, p = {t_p_two:.10f}")

    print("\nOne-Tailed Paired t-test (Hybrid > Baseline)")

    # Convert to one-tailed p-value
    if t_stat > 0:
        t_p_one = t_p_two / 2
    else:
        t_p_one = 1 - (t_p_two / 2)

    print(f"t = {t_stat:.4f}, p(one-tailed) = {t_p_one:.10f}")

    print("\nMcNemar's Test (Prediction Agreement)")

    hyb_pred = pd.read_csv(RESULTS_DIR / "hybrid_predictions_full.csv")
    base_pred = pd.read_csv(RESULTS_DIR / "baseline_predictions_full.csv")

    # --- Extract true labels (annotation = ground truth)
    if "annotation" in hyb_pred.columns:
        y_true = hyb_pred["annotation"].map({"real": 0, "fake": 1}).values
    else:
        raise ValueError("No 'annotation' column found in hybrid_predictions_full.csv")

    y_pred_h = hyb_pred["predicted_label"].map({"real": 0, "fake": 1}).values
    y_pred_b = base_pred["predicted_label_baseline"].map({"real": 0, "fake": 1}).values

    # --- Compute correctness for each model
    correct_baseline = (y_pred_b == y_true).astype(int)
    correct_hybrid   = (y_pred_h == y_true).astype(int)

    # --- Disagreement counts
    n01 = ((correct_baseline == 1) & (correct_hybrid == 0)).sum()  # baseline correct, hybrid wrong
    n10 = ((correct_baseline == 0) & (correct_hybrid == 1)).sum()  # baseline wrong, hybrid correct

    print(f"McNemar counts: baseline correct / hybrid wrong (n01) = {n01}")
    print(f"                baseline wrong / hybrid correct (n10) = {n10}")

    # --- Correct contingency table for McNemar
    table = [[0, n01],
            [n10, 0]]

    result = mcnemar(table, exact=False, correction=True)  # chi-square approximation
    print(f"McNemar’s χ² = {result.statistic:.4f}, p = {result.pvalue:.10f}")


     # ----------------------------
    # Step 5 — Summary and save
    # ----------------------------
    summary = {
        "KS_statistic": ks_stat,
        "KS_pvalue": ks_p,
        "Normality": normality,
        "Test_used": test_used,
        "T_stat": t_stat,
        "T_p_two_tailed": t_p_two,
        "T_p_one_tailed": t_p_one,
        "McNemar_stat": result.statistic,
        "McNemar_p": result.pvalue,
        "Baseline_mean_acc": np.mean(baseline_acc),
        "Hybrid_mean_acc": np.mean(hybrid_acc),
        "Baseline_mean_prec" : np.mean(baseline_metrics["test_prec"]),
        "Hybrid_mean_prec" : np.mean(hybrid_metrics["test_prec"]),
        "Baseline_mean_rec" : np.mean(baseline_metrics["test_rec"]),
        "Hybrid_mean_rec" : np.mean(hybrid_metrics["test_rec"]),
        "Baseline_mean_f1": np.mean(baseline_metrics["test_f1"]),
        "Hybrid_mean_f1": np.mean(hybrid_metrics["test_f1"])
    }

    summary_df = pd.DataFrame([summary])
    summary_csv = RESULTS_DIR / "statistical_tests_summary.csv"
    summary_txt = RESULTS_DIR / "statistical_tests_summary.txt"
    summary_df.to_csv(summary_csv, index=False)

    with open(summary_txt, "w") as f:
        f.write("=== Statistical Tests Summary ===\n")
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    print(f"\n💾 Saved statistical test summary to:\n- {summary_csv}\n- {summary_txt}")

if __name__ == "__main__":
    run_tests()
