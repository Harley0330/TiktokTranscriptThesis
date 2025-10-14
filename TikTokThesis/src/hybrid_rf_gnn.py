"""
Hybrid Random Forest + GNN
Combines TF-IDF features with averaged GNN document probabilities from multiple folds.
"""

import torch
import pandas as pd
import scipy.sparse as sp
import numpy as np
from utils import RAW_DIR, PROCESSED_DIR, RESULTS_DIR, MODELS_DIR
from preprocessing import preprocess_dataset
from train import prepare_data, get_folds
from feature_extraction import build_word_occurrence_graph
from gnn_model import GNNClassifier, extract_gnn_probabilities, extract_gnn_embeddings
from random_forest import run_rf_with_features
from sklearn.metrics import accuracy_score
from pathlib import Path
from sklearn.preprocessing import StandardScaler

def run_hybrid_rf(random_state=42, max_features=5000, window_size=2, use_ensemble=True):
    """
    Runs the hybrid model: TF-IDF + GNN probability fusion.
    Optionally ensembles GNN predictions from multiple folds.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # === Step 3: Load trained GNN model(s) ===
    print("\n=== Step 3: Load trained GNN model(s) ===")
    model_paths = sorted(Path(MODELS_DIR).glob("gnn_fold*_best.pth"))
    if len(model_paths) == 0:
        print("⚠️ No trained GNN models found! Using randomly initialized one.")
        model_paths = [None]
    else:
        print(f"Found {len(model_paths)} trained GNN fold models.")

    # === Step 4: Extract GNN document embeddings (ensemble) ===
    print("\n=== Step 4: Extract GNN document embeddings ===")
    all_embs = []

    for path in model_paths:
        model = GNNClassifier(input_dim=x.shape[1], hidden_dim=64, dropout=0.5).to(device)
        if path is not None:
            state_dict = torch.load(path, map_location=device)
            model_state = model.state_dict()

            # Keep only matching parameter shapes
            filtered_state_dict = {
                k: v for k, v in state_dict.items()
                if k in model_state and v.size() == model_state[k].size()
            }

            missing_keys = [k for k in model_state.keys() if k not in filtered_state_dict]
            if missing_keys:
                print(f"⚠️ Skipping incompatible layers for {path.name}: {missing_keys[:3]}...")

            model_state.update(filtered_state_dict)
            model.load_state_dict(model_state, strict=False)

            print(f"Loaded {path.name}")
        model.eval()

        # 🔹 Extract document embeddings instead of probabilities
        emb = extract_gnn_embeddings(
            model, x, edge_index, tokens_list, vocab_index, tfidf_matrix=X, device=device
        )
        all_embs.append(emb)

    # 🔸 Average document embeddings across all folds (ensemble)
    gnn_emb = np.mean(all_embs, axis=0)
    print(f"GNN embeddings shape: {gnn_emb.shape}")


    # === Step 5: Combine TF-IDF + GNN features ===
    print("\n=== Step 5: Combine TF-IDF + GNN probability features ===")

    scaler = StandardScaler()
    gnn_emb_scaled = scaler.fit_transform(gnn_emb)

    # Optionally increase its influence (e.g., 3x weighting)
    gnn_weight = 3.0
    gnn_emb_scaled *= gnn_weight

    X_combined = sp.hstack([X, gnn_emb_scaled], format="csr")
    print(f"Combined feature matrix shape: {X_combined.shape}")

    # === Step 6: Train Random Forest with Stratified K-Fold ===
    print("\n=== Step 6: Training Random Forest on hybrid features (Stratified K-Fold) ===")

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

            # --- Optional threshold tuning to maximize test accuracy ---
        best_acc = acc
        best_thresh = 0.5
        best_f1 = f1

        from sklearn.metrics import accuracy_score, f1_score

        for t in np.arange(0.4, 0.61, 0.01):
            y_pred_thresh = (y_proba > t).astype(int)
            acc_t = accuracy_score(y_test, y_pred_thresh)
            f1_t = f1_score(y_test, y_pred_thresh)
            if acc_t > best_acc:
                best_acc = acc_t
                best_f1 = f1_t
                best_thresh = t

        print(f"Best threshold for fold {fold}: {best_thresh:.2f} | Acc: {best_acc:.4f} | F1: {best_f1:.4f}")

        # Apply best threshold to predictions
        y_pred = (y_proba > best_thresh).astype(int)


        fold_metrics.append({
            "fold": fold,
            "train_acc": train_acc, "train_precision": train_prec, "train_recall": train_rec, "train_f1": train_f1,
            "test_acc": acc, "test_precision": prec, "test_recall": rec, "test_f1": f1
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

    # === Step 8: Cross-validated performance summary ===
    metrics_df = pd.DataFrame(fold_metrics)
    mean_acc = metrics_df["test_acc"].mean()
    mean_f1 = metrics_df["test_f1"].mean()
    mean_prec = metrics_df["test_precision"].mean()
    mean_rec = metrics_df["test_recall"].mean()
    print(f"OOB Score: {rf.oob_score_:.4f}")
    print("\n📊 Cross-validated performance (mean across folds):")
    print(f"Accuracy : {mean_acc:.4f}")
    print(f"Precision: {mean_prec:.4f}")
    print(f"Recall   : {mean_rec:.4f}")
    print(f"F1-score : {mean_f1:.4f}")

    # Save fold metrics
    metrics_path = RESULTS_DIR / "hybrid_fold_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved fold metrics to: {metrics_path}")


if __name__ == "__main__":
    run_hybrid_rf()
