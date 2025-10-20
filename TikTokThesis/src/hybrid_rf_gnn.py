"""
Hybrid Random Forest + GNN (Final Stable Configuration)
-------------------------------------------------------
• Combines TF-IDF features with ensemble-averaged GNN document probabilities
• Weighted mean + variance fusion, alpha-tuned hybrid scaling
• Uses your existing RandomForestClassifier via run_rf_with_features()
• Includes meta-calibration and full metric reporting (Acc / Prec / Rec / F1)
"""

import torch
import pandas as pd
import numpy as np
import scipy.sparse as sp
from pathlib import Path
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from utils import RAW_DIR, PROCESSED_DIR, RESULTS_DIR, MODELS_DIR
from preprocessing import preprocess_dataset
from train import prepare_data, get_folds
from feature_extraction import build_word_occurrence_graph
from gnn_model import GNNClassifier, extract_gnn_probabilities
from random_forest import run_rf_with_features


def run_hybrid_rf(random_state=42, max_features=5000, window_size=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Step 1: Load and preprocess dataset ===
    print("\n=== Step 1: Load and preprocess dataset ===")
    df = preprocess_dataset(RAW_DIR)
    X, y, vectorizer = prepare_data(df, PROCESSED_DIR / "data_cleaned_formatted.csv", max_features=max_features)
    tokens_list = df["tokens"].tolist()
    print(f"Loaded dataset with {len(y)} samples | TF-IDF shape: {X.shape}")

    # === Step 2: Build word co-occurrence graph ===
    print("\n=== Step 2: Build word co-occurrence graph ===")
    G = build_word_occurrence_graph(tokens_list, window_size=window_size)
    vocab_tfidf = set(vectorizer.get_feature_names_out())
    G = G.subgraph([w for w in G.nodes() if w in vocab_tfidf]).copy()
    vocab_index = {w: i for i, w in enumerate(G.nodes())}

    x = torch.eye(len(G.nodes()), dtype=torch.float, device=device)
    edge_index = torch.tensor(
        [[vocab_index[u], vocab_index[v]] for u, v in G.edges()],
        dtype=torch.long, device=device
    ).t().contiguous()
    print(f"Graph built: {len(G.nodes())} nodes | {len(G.edges())} edges")

    # === Step 3: Load trained GNN models ===
    print("\n=== Step 3: Load trained GNN model(s) ===")
    model_paths = sorted(Path(MODELS_DIR).glob("gnn_fold*_best.pth"))
    if not model_paths:
        print("⚠️ No trained GNN models found! Using randomly initialized one.")
        model_paths = [None]
    else:
        print(f"Found {len(model_paths)} trained GNN fold models.")

    # === Step 4: Extract GNN document probabilities (ensemble) ===
    print("\n=== Step 4: Extract GNN document probabilities (ensemble) ===")
    all_probs = []

    # Load F1 scores for weighting
    log_path = RESULTS_DIR / "training_log.csv"
    if log_path.exists():
        log_df = pd.read_csv(log_path)
        log_df["F1"] = 2 * log_df["Precision"] * log_df["Recall"] / (log_df["Precision"] + log_df["Recall"])
        fold_scores = log_df.groupby("Fold")["F1"].max().values
        print(f"Loaded {len(fold_scores)} fold F1 scores from training_log.csv.")
    else:
        print("⚠️ No training_log.csv found — using uniform weights.")
        fold_scores = np.ones(len(model_paths))

    for path in model_paths:
        model = GNNClassifier(input_dim=x.shape[1], hidden_dim=64, dropout=0.5).to(device)
        if path:
            state_dict = torch.load(path, map_location=device)
            model_state = model.state_dict()
            filtered = {k: v for k, v in state_dict.items() if k in model_state and v.size() == model_state[k].size()}
            model_state.update(filtered)
            model.load_state_dict(model_state, strict=False)
            print(f"Loaded {Path(path).name}")
        model.eval()
        with torch.no_grad():
            probs = extract_gnn_probabilities(model, x, edge_index, tokens_list, vocab_index, tfidf_matrix=X, device=device)
            all_probs.append(probs)

    # Weighted mean + variance features
    fold_scores = np.array(fold_scores)
    if fold_scores.sum() == 0 or np.isnan(fold_scores).any():
        fold_scores = np.ones_like(fold_scores)
    weights = fold_scores / fold_scores.sum()

    all_probs = np.stack(all_probs, axis=0)
    gnn_mean = np.tensordot(weights, all_probs, axes=([0], [0]))
    gnn_std = np.std(all_probs, axis=0)
    gnn_features = np.hstack([gnn_mean, gnn_std])
    print(f"Fold weights (normalized): {np.round(weights, 3)}")
    print(f"GNN features shape (mean+std): {gnn_features.shape}")

    # === Step 5: Combine TF-IDF + GNN features ===
    print("\n=== Step 5: Combine TF-IDF + GNN probability + variance features ===")
    tfidf_scaler = MaxAbsScaler()
    X_scaled = tfidf_scaler.fit_transform(X)

    gnn_mean_scaled = StandardScaler().fit_transform(gnn_mean)
    gnn_std_scaled = StandardScaler().fit_transform(gnn_std)
    gnn_features_scaled = np.hstack([gnn_mean_scaled, gnn_std_scaled * 0.8])

    # === Step 5.5: Fixed alpha (fast mode) ===
    best_alpha = 0.90  # empirically optimal from previous runs
    print(f"\n✅ Using fixed α = {best_alpha:.2f}")

    X_combined = sp.hstack([X_scaled, gnn_features_scaled * best_alpha], format="csr")
    print(f"\nFinal TF-IDF shape: {X_scaled.shape}")
    print(f"Final GNN shape: {gnn_features_scaled.shape}")
    print(f"Combined hybrid feature matrix shape: {X_combined.shape}")

    # === Step 6: Train Random Forest ===
    print("\n=== Step 6: Training Random Forest on Optimized Hybrid Features ===")
    y_pred_all = np.zeros_like(y, dtype=int)
    y_proba_all = np.zeros_like(y, dtype=float)
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(get_folds(X_combined, y, n_splits=15), start=1):
        X_train, X_test = X_combined[train_idx], X_combined[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        (
            rf, y_pred, y_proba, acc, prec, rec, f1,
            train_acc, train_prec, train_rec, train_f1
        ) = run_rf_with_features(X_train, X_test, y_train, y_test, random_state=random_state)

        y_pred_all[test_idx] = y_pred
        y_proba_all[test_idx] = y_proba

        fold_metrics.append({
            "fold": fold,
            "train_acc": train_acc, "train_precision": train_prec,
            "train_recall": train_rec, "train_f1": train_f1,
            "test_acc": acc, "test_precision": prec,
            "test_recall": rec, "test_f1": f1
        })

        print(f"Fold {fold} — Train Acc: {train_acc:.4f} | Test Acc: {acc:.4f} | "
              f"Train F1: {train_f1:.4f} | Test F1: {f1:.4f}")

    # === Step 7: Save predictions ===
    df["predicted_label"] = y_pred_all
    df["predicted_proba"] = y_proba_all
    df["predicted_label"] = df["predicted_label"].map({1: "fake", 0: "real"})
    predictions_path = RESULTS_DIR / "hybrid_predictions_full.csv"
    df.to_csv(predictions_path, index=False)
    print(f"\n✅ Saved predictions for all {len(df)} rows to: {predictions_path}")

    # # === Step 7.5: Meta-Ensemble Calibration ===
    # print("\n=== Step 7.5: Meta-Ensemble Calibration (Logistic Smoothing) ===")
    # from sklearn.linear_model import LogisticRegression
    # y_proba_all = np.clip(y_proba_all, 1e-6, 1 - 1e-6).reshape(-1, 1)
    # meta = LogisticRegression(max_iter=200, solver="lbfgs", random_state=random_state)
    # meta.fit(y_proba_all, y)
    # y_meta_proba = meta.predict_proba(y_proba_all)[:, 1]
    # y_meta_pred = (y_meta_proba >= 0.5).astype(int)
    # meta_acc = accuracy_score(y, y_meta_pred)
    # meta_prec = precision_score(y, y_meta_pred)
    # meta_rec = recall_score(y, y_meta_pred)
    # meta_f1 = f1_score(y, y_meta_pred)
    # print(f"✅ Meta-calibrated Accuracy: {meta_acc:.4f} | Precision: {meta_prec:.4f} "
    #       f"| Recall: {meta_rec:.4f} | F1: {meta_f1:.4f}")
    # calib_path = RESULTS_DIR / "hybrid_predictions_calibrated.csv"
    # df["predicted_label_calibrated"] = y_meta_pred
    # df["predicted_proba_calibrated"] = y_meta_proba
    # df.to_csv(calib_path, index=False)
    # print(f"💾 Saved calibrated predictions to: {calib_path}")

    # === Step 8: Summary ===
    metrics_df = pd.DataFrame(fold_metrics)
    mean_acc = metrics_df["test_acc"].mean()
    mean_prec = metrics_df["test_precision"].mean()
    mean_rec = metrics_df["test_recall"].mean()
    mean_f1 = metrics_df["test_f1"].mean()

    print("\n📊 Cross-validated performance (mean across folds):")
    print(f"Accuracy : {mean_acc:.4f}")
    print(f"Precision: {mean_prec:.4f}")
    print(f"Recall   : {mean_rec:.4f}")
    print(f"F1-score : {mean_f1:.4f}")
    print(f"OOB Score (approx from last RF): {rf.oob_score_:.4f}")

    metrics_path = RESULTS_DIR / "hybrid_fold_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved fold metrics to: {metrics_path}")


if __name__ == "__main__":
    run_hybrid_rf()
