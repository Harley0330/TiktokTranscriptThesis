"""
Fold Diagnostics Script (All Folds)
Analyzes each fold's performance and extracts useful insights
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from pathlib import Path
from utils import PROCESSED_DIR, RESULTS_DIR

# Update this path if needed
DATA_PATH = Path(PROCESSED_DIR/"data_cleaned_formatted.csv")
BASELINE_PRED_PATH = Path(RESULTS_DIR/"baseline_predictions_full.csv")
HYBRID_PRED_PATH = Path(RESULTS_DIR/"hybrid_predictions_full.csv")
FOLD_METRICS_PATH = Path(RESULTS_DIR/"baseline_fold_metrics.csv")

# =============================================================
# 1. LOAD DATA
# =============================================================
df = pd.read_csv(DATA_PATH)
df_base = pd.read_csv(BASELINE_PRED_PATH)
df_hyb = pd.read_csv(HYBRID_PRED_PATH)
fold_metrics = pd.read_csv(FOLD_METRICS_PATH)

print("Loaded dataset:", df.shape)

# Create an empty list to store results for all folds
all_fold_diagnostics = []

# =============================================================
# 2. IDENTIFY FOLDS
# =============================================================
# Loop through all folds (1 to 15)
for fold in range(1, 16):
    print(f"\n=== Fold {fold} Metrics ===")
    
    # Filter for the specific fold
    fold_pred = fold_metrics[fold_metrics["fold"] == str(fold)]
    
    # =============================================================
    # 3. Get fold indices (reconstruct via alphabetical split)
    # Your CV split is reproducible, so we recompute it
    # =============================================================
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=15, shuffle=True, random_state=42)
    labels = df["annotation"].map({"real": 0, "fake": 1}).values

    fold_idx = None
    for fold_idx, (_, test_idx) in enumerate(skf.split(np.zeros(len(labels)), labels), start=1):
        if fold_idx == fold:
            fold_idx = test_idx
            break

    df_fold = df.iloc[fold_idx].copy()
    print(f"Fold {fold} samples:", len(df_fold))

    # =============================================================
    # 4. LABEL DISTRIBUTION
    # =============================================================
    label_dist = df_fold["annotation"].value_counts()

    # =============================================================
    # 5. TRANSCRIPT LENGTH ANALYSIS
    # =============================================================
    df_fold["token_count"] = df_fold["tokens"].apply(len)
    token_stats = df_fold["token_count"].describe()

    # =============================================================
    # 6. TOPIC DRIFT VIA TF-IDF SIMILARITY
    # =============================================================
    corpus = ["".join(tokens) for tokens in df["tokens"]]
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(corpus)

    X_fold = X[fold_idx]
    X_rest = X[[i for i in range(len(df)) if i not in fold_idx]]

    similarities = cosine_similarity(X_fold, X_rest).mean(axis=1)
    similarity_stats = np.mean(similarities)

    # =============================================================
    # 7. RARE WORD ANALYSIS
    # =============================================================
    all_words = [w for tokens in df["tokens"] for w in tokens]
    word_freq = Counter(all_words)

    df_fold["avg_rarity"] = df_fold["tokens"].apply(lambda toks: np.mean([word_freq[t] for t in toks]))
    rarity_stats = df_fold["avg_rarity"].describe()

    # =============================================================
    # 8. MISCLASSIFICATION ANALYSIS
    # =============================================================
    df_fold_base = df_base.iloc[fold_idx]
    df_fold_hyb = df_hyb.iloc[fold_idx]

    df_fold_base["true"] = df_fold["annotation"]
    df_fold_hyb["true"] = df_fold["annotation"]

    df_fold_base["correct"] = (df_fold_base["predicted_label_baseline"] == df_fold_base["true"])
    df_fold_hyb["correct"] = (df_fold_hyb["predicted_label_calibrated"] == df_fold_hyb["true"])

    baseline_misclassification_rate = df_fold_base["correct"].value_counts(normalize=True).get(True, 0)
    hybrid_misclassification_rate = df_fold_hyb["correct"].value_counts(normalize=True).get(True, 0)

    # Store the metrics for the current fold
    fold_diagnostics = {
        "fold": fold,
        "real_count": label_dist.get("real", 0),
        "fake_count": label_dist.get("fake", 0),
        "mean_token_count": token_stats["mean"],
        "std_token_count": token_stats["std"],
        "mean_similarity": similarity_stats,
        "mean_rarity": rarity_stats["mean"],
        "baseline_misclassification_rate": baseline_misclassification_rate,
        "hybrid_misclassification_rate": hybrid_misclassification_rate,
    }

    all_fold_diagnostics.append(fold_diagnostics)

# =============================================================
# 9. SAVE FULL DIAGNOSTIC REPORT FOR ALL FOLDS
# =============================================================
df_diagnostics = pd.DataFrame(all_fold_diagnostics)
df_diagnostics.to_csv(RESULTS_DIR/"all_folds_diagnostics_report.csv", index=False)

print("\nSaved: all_folds_diagnostics_report.csv")

print("\n=== DIAGNOSTICS COMPLETE ===")
