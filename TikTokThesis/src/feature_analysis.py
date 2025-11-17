import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import RESULTS_DIR, MODELS_DIR, PLOTS_DIR
from sklearn.metrics import confusion_matrix, classification_report

# ========================================================
# 1. LOAD MODELS & DATA
# ========================================================

rf = joblib.load(MODELS_DIR/"rf_hybrid_gnn.pkl")
vectorizer = joblib.load(MODELS_DIR/"tfidf_vectorizer_hybrid.pkl")
scalers = joblib.load(MODELS_DIR/"hybrid_scalers.pkl")

hybrid_df = pd.read_csv(RESULTS_DIR/"hybrid_predictions_full.csv")
fold_df = pd.read_csv(RESULTS_DIR/"hybrid_fold_metrics.csv")

print("Loaded RF model, vectorizer, scalers, and prediction logs.")

GRAPHS_DIR = PLOTS_DIR / "feature_importance_plots"
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

# ========================================================
# 2. FEATURE IMPORTANCE EXTRACTION
# ========================================================

feature_names = list(vectorizer.get_feature_names_out())
n_tfidf = len(feature_names)

# GNN contributes 1 feature → name it
feature_names += ["gnn_prob"]

# RF feature importances
importances = rf.feature_importances_

# Pack in a DataFrame
feat_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values("importance", ascending=False)

print("\nTop 10 Most Important Features:")
print(feat_df.head(10))

# ========================================================
# 3. PLOT: Feature Importance Bar Chart
# ========================================================

plt.figure(figsize=(10, 6))
sns.barplot(data=feat_df.head(20),
            x="importance", y="feature", palette="viridis")
plt.title("Top 20 Feature Importances (Proposed RF+GNN)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig(GRAPHS_DIR / "feature_importance.png", dpi=300)

# ========================================================
# 4. Accuracy Distribution Across Folds
# ========================================================

plt.figure(figsize=(8, 5))
sns.histplot(fold_df["test_acc"], kde=True, bins=10)
plt.title("Distribution of Fold Test Accuracy (Proposed Model)")
plt.xlabel("Accuracy")
plt.ylabel("Count")
plt.savefig(GRAPHS_DIR / "acc_dist.png", dpi=300)
# ========================================================
# 5. F1 Score Distribution Across Folds
# ========================================================

plt.figure(figsize=(8, 5))
sns.histplot(fold_df["test_f1"], kde=True, color="orange", bins=10)
plt.title("Distribution of Fold Test F1 Score (Proposed Model)")
plt.xlabel("F1 Score")
plt.ylabel("Count")
plt.savefig(GRAPHS_DIR / "f1_dist.png", dpi=300)
# ========================================================
# 6. Probability Histogram (Real vs Fake)
# ========================================================

plt.figure(figsize=(8, 5))
sns.histplot(hybrid_df[hybrid_df["label"] == 0]["predicted_proba"],
             color="green", kde=True, label="Real")
sns.histplot(hybrid_df[hybrid_df["label"] == 1]["predicted_proba"],
             color="red", kde=True, label="Fake")
plt.title("Probability Distribution of Proposed Model Outputs")
plt.xlabel("Predicted Fake Probability")
plt.legend()
plt.savefig(GRAPHS_DIR / "prob_hist.png", dpi=300)

# ========================================================
# 7. Print Classification Report
# ========================================================
print("\nClassification Report:")
print(classification_report(hybrid_df["annotation"], hybrid_df["predicted_label"]))
