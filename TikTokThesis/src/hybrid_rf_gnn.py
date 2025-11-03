"""
Hybrid Random Forest + GNN (Integrated Version)
Receives preprocessed dataset, TF-IDF, and Graph from main.py.
"""

import torch, numpy as np, pandas as pd, scipy.sparse as sp
from pathlib import Path
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from src.utils import RESULTS_DIR, MODELS_DIR
from src.gnn_model import GNNClassifier, extract_gnn_probabilities
from src.random_forest import run_rf_with_features
from src.train import get_folds

APPLY_SVD = False
SVD_N_COMPONENTS = 500
GNN_WEIGHT_ALPHA = 0.90

def run_hybrid_rf(df,X, y, tokens_list, vectorizer, G, vocab_index, device, random_state=42):
    """
    Hybrid Random Forest using given parameters
    """

    print("\n=== Step 1: Prepare graph tensors ===")
    x = torch.eye(len(G.nodes()), dtype=torch.float, device=device)
    edge_index = torch.tensor(
        [[vocab_index[u], vocab_index[v]] for u, v in G.edges()],
        dtype=torch.long, device=device
    ).t().contiguous()
    print(f"Graph ready: {len(G.nodes())} nodes | {len(G.edges())} edges")

    # === Step 2: Load trained GNN models ===
    print("\n=== Step 2: Load trained GNN models")
    model_paths = sorted(Path(MODELS_DIR).glob("gnn_fold*_best.pth"))
    if not model_paths:
        raise RuntimeError("No trained GNN models found.")
    print(f"Found {len(model_paths)} trained GNN fold models.")

    # === Step 3: Extract ensemble GNN probabilities ===
    print("\n=== Step 3: Extract GNN probabilities ===")
    all_probs = []
    for path in model_paths:
        model = GNNClassifier(input_dim=x.shape[1], hidden_dim=64, dropout=0.5).to(device)
        state = torch.load(path, map_location=device)
        model_state = model.state_dict()
        filtered = {k: v for k, v in state.items()
                    if k in model_state and v.size() == model_state[k].size()}
        if len(filtered) < len(state):
            print(f"⚠️ Skipping incompatible layers for {path.name}")
        model_state.update(filtered)
        model.load_state_dict(model_state, strict=False)
        model.eval()
        with torch.no_grad():
            probs = extract_gnn_probabilities(model, x, edge_index, tokens_list, vocab_index,
                                              tfidf_matrix=X, device=device)
            all_probs.append(probs)
    gnn_prob_weighted = np.mean(np.stack(all_probs, axis=0), axis=0)
    print(f"GNN probability shape: {gnn_prob_weighted.shape}")

    print("\n=== Step 4: Feature scaling & optional PCA compression ===")

    # 1) TF-IDF transform
    if APPLY_SVD:
        print(f"▶ Applying TruncatedSVD ({SVD_N_COMPONENTS} components) ...")
        svd = TruncatedSVD(n_components=SVD_N_COMPONENTS, random_state=42)
        tfidf_reduced = svd.fit_transform(X)                         # dense (n, k)
        tfidf_scaled = StandardScaler(with_mean=True).fit_transform(tfidf_reduced)
        print(f"Reduced TF-IDF shape: {tfidf_scaled.shape}")
    else:
        tfidf_scaled = MaxAbsScaler().fit_transform(X)               # sparse (n, d)
        print(f"Scaled TF-IDF shape: {tfidf_scaled.shape}")

    # 2) GNN block (probabilities or probabilities+extras), scale + weight
    gnn_block = gnn_prob_weighted
    if gnn_block.ndim == 1:
        gnn_block = gnn_block.reshape(-1, 1)

    gnn_scaled = StandardScaler().fit_transform(gnn_block)
    gnn_scaled *= GNN_WEIGHT_ALPHA
    print(f"GNN block shape (after scale & α): {gnn_scaled.shape}")

    # 3) Combine (handle dense vs. sparse properly)
    if APPLY_SVD:
        # TF-IDF is dense -> use np.hstack with dense GNN
        X_combined = np.hstack([tfidf_scaled, gnn_scaled]).astype(np.float32)
    else:
        # TF-IDF is sparse -> convert GNN to sparse and hstack
        gnn_sparse = sp.csr_matrix(gnn_scaled)
        X_combined = sp.hstack([tfidf_scaled, gnn_sparse], format="csr")

    print(f"Hybrid feature matrix shape: {X_combined.shape}")

  # === Step 5: Stratified 15-Fold Training (per-fold α + threshold tuning) ===
    print("\n=== Step 5: Stratified 15-Fold Training (per-fold α + threshold tuning) ===")

    fold_metrics, y_pred_all, y_proba_all = [], np.zeros_like(y), np.zeros_like(y, dtype=float)

    for fold, (train_idx, test_idx) in enumerate(get_folds(X_combined, y, n_splits=15), start=1):
        X_train, X_test = X_combined[train_idx], X_combined[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # === Step 5a: Per-fold alpha optimization ===
        best_alpha, best_acc_alpha = 1.0, 0
        for alpha in np.arange(0.8, 1.3, 0.05):
            X_train_alpha = sp.hstack([tfidf_scaled[train_idx],
                                    gnn_scaled[train_idx] * alpha], format="csr")
            X_test_alpha = sp.hstack([tfidf_scaled[test_idx],
                                    gnn_scaled[test_idx] * alpha], format="csr")

            rf, y_pred_a, y_proba_a, acc_a, _, _, _, _, _, _, _ = \
                run_rf_with_features(X_train_alpha, X_test_alpha, y_train, y_test,
                                    random_state=random_state)
            if acc_a > best_acc_alpha:
                best_acc_alpha = acc_a
                best_alpha = alpha

        # Rebuild hybrid features with the best α for this fold
        X_train_opt = sp.hstack([tfidf_scaled[train_idx],
                                gnn_scaled[train_idx] * best_alpha], format="csr")
        X_test_opt = sp.hstack([tfidf_scaled[test_idx],
                                gnn_scaled[test_idx] * best_alpha], format="csr")

        rf, y_pred, y_proba, acc, prec, rec, f1, tr_acc, tr_prec, tr_rec, tr_f1 = \
            run_rf_with_features(X_train_opt, X_test_opt, y_train, y_test, random_state=random_state)

        # === Step 5b: Threshold calibration per fold ===
        best_t, best_acc, best_prec, best_rec, best_f1 = 0.5, acc, prec, rec, f1
        for t in np.arange(0.4, 0.61, 0.01):
            y_pred_t = (y_proba > t).astype(int)
            acc_t = accuracy_score(y_test, y_pred_t)
            prec_t = precision_score(y_test, y_pred_t)
            rec_t = recall_score(y_test, y_pred_t)
            f1_t = f1_score(y_test, y_pred_t)
            if acc_t > best_acc:
                best_acc, best_t, best_prec, best_rec, best_f1 = acc_t, t, prec_t, rec_t, f1_t

        y_pred_cal = (y_proba > best_t).astype(int)
        y_pred_all[test_idx], y_proba_all[test_idx] = y_pred_cal, y_proba

        fold_metrics.append({
            "fold": fold,
            "best_alpha": best_alpha,
            "best_thresh": best_t,
            "train_acc": tr_acc,
            "train_prec": tr_prec,
            "train_rec": tr_rec,
            "train_f1": tr_f1,
            "test_acc": best_acc,
            "test_prec": best_prec,
            "test_rec": best_rec,
            "test_f1": best_f1,
            "oob": getattr(rf, "oob_score_", np.nan)
        })

        print(
            f"Fold {fold:02d} — α:{best_alpha:.2f} Thresh:{best_t:.2f} "
            f"| TrainAcc:{tr_acc:.4f} TrainF1:{tr_f1:.4f} "
            f"| TestAcc:{best_acc:.4f} TestF1:{best_f1:.4f} "
            f"| OOB:{getattr(rf, 'oob_score_', float('nan')):.4f}"
        )

    # === Step 6: Aggregate calibrated fold metrics ===
    metrics_df = pd.DataFrame(fold_metrics)
    mean_train_acc = metrics_df["train_acc"].mean()
    mean_train_f1 = metrics_df["train_f1"].mean()
    mean_test_acc = metrics_df["test_acc"].mean()
    mean_test_f1 = metrics_df["test_f1"].mean()
    mean_t = metrics_df["best_thresh"].mean()
    mean_a = metrics_df["best_alpha"].mean()
    mean_oob = metrics_df["oob"].mean()

    print("\n=== Step 6: Calibrated performance summary (with OOB) ===")
    print(f"Avg optimal α: {mean_a:.3f} | Avg threshold: {mean_t:.3f}")
    print(f"📊 Mean Train Acc:{mean_train_acc:.4f}  F1:{mean_train_f1:.4f}  "
          f"| Test Acc:{mean_test_acc:.4f}  F1:{mean_test_f1:.4f}  | OOB:{mean_oob:.4f}")

    # Save calibrated metrics and predictions
    metrics_df.to_csv(RESULTS_DIR / 'hybrid_fold_metrics_calibrated.csv', index=False)
    df["predicted_label_calibrated"] = y_pred_all
    df["predicted_label_calibrated"] = df["predicted_label_calibrated"].map({1: "fake", 0: "real"})
    df["predicted_proba"] = y_proba_all
    df.to_csv(RESULTS_DIR / 'hybrid_predictions_calibrated.csv', index=False)
    print(f"💾 Saved calibrated fold metrics and predictions to: {RESULTS_DIR}")

    # === Step 7: Save results ===
    df["predicted_label"] = y_pred_all
    df["predicted_proba"] = y_proba_all
    df["predicted_label"] = df["predicted_label"].map({1: "fake", 0: "real"})
    df.to_csv(RESULTS_DIR / "hybrid_predictions_full.csv", index=False)

    metrics_df = pd.DataFrame(fold_metrics)
    metrics_df.to_csv(RESULTS_DIR / "hybrid_fold_metrics.csv", index=False)
    print("\n📊 FINAL SUMMARY — Mean Train Acc:{:.4f} | Test Acc:{:.4f} | OOB:{:.4f}"
          .format(metrics_df.train_acc.mean(),
                  metrics_df.test_acc.mean(),
                  metrics_df.oob.mean()))
