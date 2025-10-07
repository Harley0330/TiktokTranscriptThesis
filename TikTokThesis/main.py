from src.preprocessing import preprocess_dataset, save_preprocessed_dataset
from src.feature_extraction import build_tfidf, build_word_occurrence_graph
from src.train import prepare_data, get_folds
from src.gnn_model import GNNClassifier, train_gnn, test_gnn
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch_geometric.data import Data
import torch.optim as optim
import pandas as pd
import networkx as nx
import random
import numpy as np
from itertools import product
from src.utils import RAW_DIR, PROCESSED_DIR, set_seed, SEED
from src.gnn_model import graph_to_pyg_data
import os
from sklearn.metrics import precision_score, recall_score
from sklearn.decomposition import TruncatedSVD
"""
PREPROCESSING PORTION
    - Takes in the original file including the manually encoded transcripts
    - Declares an output path for the formatted dataset
    - Calls methods from preprocessing.py, which cleans and formats the dataset
"""
if __name__ == "__main__":
    # ----------------- Setup -----------------
    set_seed = 42
    random.seed(set_seed)
    np.random.seed(set_seed)
    torch.manual_seed(set_seed)
    device = "cpu"

    # ----------------- Preprocess -----------------
    dataset_path = RAW_DIR
    output_path = PROCESSED_DIR / "data_cleaned_formatted.csv"

    df = preprocess_dataset(dataset_path)
    save_preprocessed_dataset(df, output_path)

    tokens_list = df["tokens"].tolist()

    # TF-IDF
    X, y, vectorizer = prepare_data(df, output_path, max_features=5000)
    y = np.array(y)
    print(f"TF-IDF shape: {X.shape}")

    # Vocabulary & graph
    vocab = vectorizer.get_feature_names_out()
    vocab_index = {w: i for i, w in enumerate(vocab)}
    vocab_set = set(vocab)

    G = build_word_occurrence_graph(tokens_list=tokens_list, window_size=2,vocab_set=vocab_set)
    print(f"Graph nodes: {len(G.nodes())}, edges: {len(G.edges())}")

    # Convert TF-IDF to dense if needed
    if hasattr(X, "toarray"):
        X_dense = X.toarray()
    else:
        X_dense = X

    num_nodes = len(vocab)
    num_docs = X_dense.shape[0]

    # Node features: TF-IDF values per word across all docs
    node_features = np.zeros((num_nodes, num_docs))  # [num_nodes, num_docs]

    for word, idx in vocab_index.items():
        node_features[idx] = X_dense[:, idx]  # TF-IDF of word across all documents

    # Optional: Reduce dimensionality to hidden_dim
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=64, random_state=42)
    node_features_reduced = svd.fit_transform(node_features)

    # Convert to torch
    x = torch.tensor(node_features_reduced, dtype=torch.float, device=device)

    # Edges
    edge_index = torch.tensor(
        [[vocab_index[u], vocab_index[v]] for u, v in G.edges() if u in vocab_index and v in vocab_index],
        dtype=torch.long, device=device
    ).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    # ----------------- Training -----------------
    all_logs = []

    for fold, (train_idx, test_idx) in enumerate(get_folds(X, y, n_splits=15), start=1):
        train_mask = torch.zeros(len(y), dtype=torch.bool, device=device)
        test_mask = torch.zeros(len(y), dtype=torch.bool, device=device)
        train_mask[train_idx] = True
        test_mask[test_idx] = True

        data = Data(x=x, edge_index=edge_index, y=torch.tensor(y, dtype=torch.long, device=device),
                    train_mask=train_mask, test_mask=test_mask)
        data.docs_tokens = tokens_list
        data.vocab_index = vocab_index
        data.tfidf_matrix = X

        # Model, optimizer, scheduler
        model = GNNClassifier(input_dim=data.x.shape[1], hidden_dim=64, dropout=0.6).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.005,weight_decay=2e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5,patience=5, min_lr=1e-6)

        # Class weights
        y_tensor = torch.tensor(y, dtype=torch.long, device=device)
        class_counts = torch.bincount(y_tensor)
        class_weights = (1.0 / class_counts.float())
        class_weights = class_weights / class_weights.sum() * 2.0
        class_weights = class_weights.to(torch.float)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        print(f"\nFold {fold}")
        best_test_acc = 0
        patience = 10
        wait = 0
        min_acc_threshold = 0.70
        l1_lambda = 1e-4

        for epoch in range(50):
            model.train()
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.docs_tokens, data.vocab_index, tfidf_matrix=data.tfidf_matrix)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            # L1 regularization
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss = loss + l1_lambda * l1_norm
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            # Training accuracy
            data_test_mask_backup = data.test_mask.clone()
            data.test_mask = data.train_mask
            train_acc, _, _, train_cm = test_gnn(model, data, device=device)
            data.test_mask = data_test_mask_backup

            # Test accuracy
            test_acc, preds, probs, test_cm = test_gnn(model, data, device=device)

            # Precision & Recall
            precision = precision_score(y[test_idx], preds[test_mask].cpu(), zero_division=0)
            recall = recall_score(y[test_idx], preds[test_mask].cpu(), zero_division=0)

            # Scheduler step
            scheduler.step(test_acc)

            # Threshold-based early stopping
            if best_test_acc >= min_acc_threshold:
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    wait = 0
                else:
                    wait += 1
                if wait >= patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break
            else:
                if test_acc > best_test_acc:
                    best_test_acc = test_acc

            # Logging
            logs = {
                "Fold": fold,
                "Epoch": epoch,
                "Loss": loss.item(),
                "Train_Acc": train_acc,
                "Test_Acc": test_acc,
                "Precision": precision,
                "Recall": recall,
                "Train_CM": train_cm.tolist(),
                "Test_CM": test_cm.tolist()
            }
            all_logs.append(logs)

            print(f"Epoch {epoch:02d} | Loss: {loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | "
                f"Precision: {precision:.4f} | Recall: {recall:.4f}")

    # ----------------- Save logs -----------------
    df_logs = pd.DataFrame(all_logs)
    df_logs.to_csv("training_log.csv", index=False)

    # #RANDOMIZED SEARCH

    # --- Set seed ---
    # set_seed = 42
    # random.seed(set_seed)
    # np.random.seed(set_seed)
    # torch.manual_seed(set_seed)
    # print("Performing Randomized Search")

    # # --- Hyperparameter grid ---
    # param_grid = {
    #     'hidden_dim': [64, 128, 256],
    #     'dropout': [0.2, 0.5, 0.7],
    #     'lr': [0.001, 0.003, 0.005, 0.01, 0.1],
    #     'window_size': [2, 3, 4]
    # }

    # n_iter = 30  # number of random combinations
    # all_params = list(product(
    #     param_grid['hidden_dim'],
    #     param_grid['dropout'],
    #     param_grid['lr'],
    #     param_grid['window_size']
    # ))
    # sampled_params = random.sample(all_params, n_iter)
    # results = []
    # results_file = "randomized_search_results.csv"
    # results = []

    # if os.path.exists(results_file):
    #     df_existing = pd.read_csv(results_file)
    #     results = df_existing.to_dict('records')

    # # Convert y once to tensor
    # y_tensor = torch.tensor(y, dtype=torch.long)

    # for params in sampled_params:
    #     hidden_dim, dropout, lr, window_size = params
    #     fold_accs = []

    #     for fold_idx, (train_idx, test_idx) in enumerate(get_folds(X, y, n_splits=5)):

    #         # --- Manual Data creation like your original loop ---
    #         train_mask = torch.zeros(len(y_tensor), dtype=torch.bool)
    #         test_mask = torch.zeros(len(y_tensor), dtype=torch.bool)
    #         train_mask[train_idx] = True
    #         test_mask[test_idx] = True

    #         # Node features: identity for vocab nodes
    #         x = torch.eye(len(vocab), dtype=torch.float)

    #         # Edge index: word co-occurrence edges
    #         edge_index = torch.tensor(
    #             [[vocab_index[u], vocab_index[v]] for u, v in G.edges() if u in vocab_index and v in vocab_index],
    #             dtype=torch.long
    #         ).t().contiguous()
    #         edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    #         data = Data(
    #             x=x,
    #             edge_index=edge_index,
    #             y=y_tensor,
    #             train_mask=train_mask,
    #             test_mask=test_mask
    #         )
    #         data.docs_tokens = tokens_list
    #         data.vocab_index = vocab_index
    #         data.tfidf_matrix = X

    #         # --- Initialize 4-layer model ---
    #         model = GNNClassifier(input_dim=data.x.shape[1],
    #                             hidden_dim=hidden_dim,
    #                             dropout=dropout)

    #         optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    #         # Class weights for imbalance
    #         class_counts = torch.bincount(y_tensor)
    #         class_weights = 1.0 / class_counts.float()
    #         class_weights = class_weights / class_weights.sum() * 2.0
    #         class_weights = class_weights.to(torch.float)
    #         criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    #         # --- Train for quick evaluation ---
    #         for epoch in range(20):
    #             train_gnn(model, data, optimizer, criterion)
    #             print("Epoch done")

    #         # --- Evaluate ---
    #         acc, _, _, _ = test_gnn(model, data)
    #         fold_accs.append(acc)

    #     avg_acc = np.mean(fold_accs)
    #     results.append({
    #         'hidden_dim': hidden_dim,
    #         'dropout': dropout,
    #         'lr': lr,
    #         'window_size': window_size,
    #         'avg_acc': avg_acc
    #     })

    #     # Save results to CSV after each combination
    #     df_results = pd.DataFrame(results)
    #     df_results.to_csv(results_file, index=False)
    #     print(f"Params: hidden={hidden_dim}, dropout={dropout}, lr={lr}, window={window_size} -> avg_acc={avg_acc:.4f}")

    # # --- Sort and show top 3 ---
    # results.sort(key=lambda x: x['avg_acc'], reverse=True)
    # print("\nTop 3 parameter combinations:")
    # for r in results[:3]:
    #     print(r)