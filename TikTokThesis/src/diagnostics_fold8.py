"""
Fold 8 Diagnostics Script
Analyzes why Fold 8 had the lowest accuracy
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from pathlib import Path
from utils import PROCESSED_DIR, RESULTS_DIR

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
# 2. IDENTIFY FOLD 8
# =============================================================
fold_8_pred = fold_metrics[fold_metrics["fold"] == "8"]
print("\n=== Fold 8 Metrics ===")
print(fold_8_pred)

# =============================================================
# 3. Get fold indices
# =============================================================
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=15, shuffle=True, random_state=42)
labels = df["annotation"].map({"real": 0, "fake": 1}).values

fold8_idx = None
for fold, (_, test_idx) in enumerate(skf.split(np.zeros(len(labels)), labels), start=1):
    if fold == 8:
        fold8_idx = test_idx
        break

df8 = df.iloc[fold8_idx].copy()
print("\nFold 8 samples:", len(df8))

# =============================================================
# 4. LABEL DISTRIBUTION
# =============================================================
print("\n=== Label Distribution in Fold 8 ===")
print(df8["annotation"].value_counts())

# =============================================================
# 5. TRANSCRIPT LENGTH ANALYSIS
# =============================================================
df8["token_count"] = df8["tokens"].apply(len)
print("\n=== Token Count Statistics (Fold 8) ===")
print(df8["token_count"].describe())

# Compare to whole dataset
df["token_count"] = df["tokens"].apply(len)
print("\n=== Token Count Statistics (FULL DATASET) ===")
print(df["token_count"].describe())

# =============================================================
# 6. TOPIC DRIFT VIA TF-IDF SIMILARITY
# =============================================================

corpus = ["".join(tokens) for tokens in df["tokens"]]
corpus = [doc for doc in corpus if doc.strip()]  # Remove empty documents

if not corpus:
    raise ValueError("All documents are empty after tokenization!")

print(f"Corpus length after cleaning: {len(corpus)}")
print(f"Sample corpus: {corpus[:3]}")  # Preview a few documents

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(corpus)

X8 = X[fold8_idx]
X_rest = X[[i for i in range(len(df)) if i not in fold8_idx]]

similarities = cosine_similarity(X8, X_rest).mean(axis=1)

df8["similarity_to_rest"] = similarities

print("\n=== Topic Similarity (Fold 8 → Rest) ===")
print(df8["similarity_to_rest"].describe())

# =============================================================
# 7. RARE WORD ANALYSIS
# =============================================================
all_words = [w for tokens in df["tokens"] for w in tokens]
word_freq = Counter(all_words)

df8["avg_rarity"] = df8["tokens"].apply(lambda toks: np.mean([word_freq[t] for t in toks]))

print("\n=== Word Rarity (Fold 8) ===")
print(df8["avg_rarity"].describe())

# =============================================================
# 8. UNIQUE WORDS ONLY SEEN IN FOLD 8
# =============================================================
words_fold8 = set([w for tokens in df8["tokens"] for w in tokens])
words_rest = set([w for i,tokens in enumerate(df["tokens"]) if i not in fold8_idx for w in tokens])

unique_words_8 = words_fold8 - words_rest

print("\n=== Unique Words in Fold 8 (not seen elsewhere) ===")
print(list(unique_words_8)[:50], "...")
print(f"Total: {len(unique_words_8)} unique words")

# =============================================================
# 9. MISCLASSIFICATION ANALYSIS
# =============================================================
df8_base = df_base.iloc[fold8_idx]
df8_hyb = df_hyb.iloc[fold8_idx]

df8_base["true"] = df8["annotation"]
df8_hyb["true"] = df8["annotation"]

df8_base["correct"] = (df8_base["predicted_label_baseline"] == df8_base["true"])
df8_hyb["correct"] = (df8_hyb["predicted_label_calibrated"] == df8_hyb["true"])

print("\n=== Baseline Misclassification Rate (Fold 8) ===")
print(df8_base["correct"].value_counts(normalize=True))

print("\n=== Hybrid Misclassification Rate (Fold 8) ===")
print(df8_hyb["correct"].value_counts(normalize=True))

print("\n=== Hardest Misclassified Examples (Baseline) ===")
print(df8.loc[~df8_base["correct"]][["transcript"]].head())

# =============================================================
# 10. OPTIONAL: SAVE FULL DIAGNOSTIC REPORT
# =============================================================
df8.to_csv(RESULTS_DIR/"fold8_diagnostics_report.csv", index=False)
print("\nSaved: fold8_diagnostics_report.csv")

print("\n=== DIAGNOSTICS COMPLETE ===")
