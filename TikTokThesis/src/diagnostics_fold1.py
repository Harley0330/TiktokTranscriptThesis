"""
Fold 1 Diagnostics Script
Analyzes why Fold 1 had the performance it did
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

# =============================================================
# 2. IDENTIFY FOLD 1
# =============================================================
fold_1_pred = fold_metrics[fold_metrics["fold"] == "1"]
print("\n=== Fold 1 Metrics ===")
print(fold_1_pred)

# =============================================================
# 3. Get fold indices (we reconstruct via alphabetical split)
#    Your CV split is reproducible, so we recompute it
# =============================================================
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=15, shuffle=True, random_state=42)
labels = df["annotation"].map({"real": 0, "fake": 1}).values

fold1_idx = None
for fold, (_, test_idx) in enumerate(skf.split(np.zeros(len(labels)), labels), start=1):
    if fold == 1:
        fold1_idx = test_idx
        break

df1 = df.iloc[fold1_idx].copy()
print("\nFold 1 samples:", len(df1))

# =============================================================
# 4. LABEL DISTRIBUTION
# =============================================================
print("\n=== Label Distribution in Fold 1 ===")
print(df1["annotation"].value_counts())

# =============================================================
# 5. TRANSCRIPT LENGTH ANALYSIS
# =============================================================
df1["token_count"] = df1["tokens"].apply(len)
print("\n=== Token Count Statistics (Fold 1) ===")
print(df1["token_count"].describe())

# Compare to whole dataset
df["token_count"] = df["tokens"].apply(len)
print("\n=== Token Count Statistics (FULL DATASET) ===")
print(df["token_count"].describe())

# =============================================================
# 6. TOPIC DRIFT VIA TF-IDF SIMILARITY
# =============================================================

corpus = ["".join(tokens) for tokens in df["tokens"]]
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(corpus)

X1 = X[fold1_idx]
X_rest = X[[i for i in range(len(df)) if i not in fold1_idx]]

similarities = cosine_similarity(X1, X_rest).mean(axis=1)

df1["similarity_to_rest"] = similarities

print("\n=== Topic Similarity (Fold 1 → Rest) ===")
print(df1["similarity_to_rest"].describe())

# =============================================================
# 7. RARE WORD ANALYSIS
# =============================================================
all_words = [w for tokens in df["tokens"] for w in tokens]
word_freq = Counter(all_words)

df1["avg_rarity"] = df1["tokens"].apply(lambda toks: np.mean([word_freq[t] for t in toks]))

print("\n=== Word Rarity (Fold 1) ===")
print(df1["avg_rarity"].describe())

# =============================================================
# 8. UNIQUE WORDS ONLY SEEN IN FOLD 1
# =============================================================
words_fold1 = set([w for tokens in df1["tokens"] for w in tokens])
words_rest = set([w for i,tokens in enumerate(df["tokens"]) if i not in fold1_idx for w in tokens])

unique_words_1 = words_fold1 - words_rest

print("\n=== Unique Words in Fold 1 (not seen elsewhere) ===")
print(list(unique_words_1)[:50], "...")
print(f"Total: {len(unique_words_1)} unique words")

# =============================================================
# 9. MISCLASSIFICATION ANALYSIS
# =============================================================
df1_base = df_base.iloc[fold1_idx]
df1_hyb = df_hyb.iloc[fold1_idx]

df1_base["true"] = df1["annotation"]
df1_hyb["true"] = df1["annotation"]

df1_base["correct"] = (df1_base["predicted_label_baseline"] == df1_base["true"])
df1_hyb["correct"] = (df1_hyb["predicted_label_calibrated"] == df1_hyb["true"])

print("\n=== Baseline Misclassification Rate (Fold 1) ===")
print(df1_base["correct"].value_counts(normalize=True))

print("\n=== Hybrid Misclassification Rate (Fold 1) ===")
print(df1_hyb["correct"].value_counts(normalize=True))

print("\n=== Hardest Misclassified Examples (Baseline) ===")
print(df1.loc[~df1_base["correct"]][["transcript"]].head())

# =============================================================
# 10. OPTIONAL: SAVE FULL DIAGNOSTIC REPORT
# =============================================================
df1.to_csv(RESULTS_DIR/"fold1_diagnostics_report.csv", index=False)
print("\nSaved: fold1_diagnostics_report.csv")

print("\n=== DIAGNOSTICS COMPLETE ===")
