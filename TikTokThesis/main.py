from src.preprocessing import preprocess_dataset, save_preprocessed_dataset
from src.feature_extraction import build_tfidf, build_word_occurrence_graph
from src.train import prepare_data, get_folds
from src.gnn_model import GNNClassifier, train_gnn, test_gnn, FocalLoss
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch_geometric.data import Data
import torch.optim as optim
import pandas as pd
import networkx as nx
import random

from src.utils import RAW_DIR, PROCESSED_DIR, set_seed, SEED
"""
PREPROCESSING PORTION
    - Takes in the original file including the manually encoded transcripts
    - Declares an output path for the formatted dataset
    - Calls methods from preprocessing.py, which cleans and formats the dataset
"""
if __name__ == "__main__":
    dataset_path = RAW_DIR  # original file
    output_path = PROCESSED_DIR / "data_cleaned_formatted.csv" #preprocessed path file
    
    df = preprocess_dataset(dataset_path)
    save_preprocessed_dataset(df,output_path)
    
    # Checking total number of tokens
    # Flatten all tokens into one big list
    all_tokens = [token for tokens in df["tokens"] for token in tokens]

    # Total number of tokens (all words across all transcripts)
    total_tokens = len(all_tokens)
    unique_tokens = len(set(all_tokens))
    print(f"Total : {total_tokens}")
    print(f"Unique : {unique_tokens}")

    # Convert tokens back to text for TF-IDF
    corpus = [" ".join(tokens) for tokens in df["tokens"]]
    
    # Prepare TF-IDF and labels
    X, y, vectorizer = prepare_data(df, output_path, max_features=5000)
    tokens_list = df["tokens"].tolist()
    print("TF-IDF shape:", X.shape)

    # --- Build vocab & graph aligned with TF-IDF ---
    vocab = vectorizer.get_feature_names_out()
    vocab_set = set(vocab)
    G = build_word_occurrence_graph(tokens_list=tokens_list, window_size=2, vocab_set=vocab_set)

    vocab_index = {w: i for i, w in enumerate(vocab)}

    print("Graph nodes:", len(G.nodes()))
    print("Graph edges:", len(G.edges()))

    sub_nodes = list(G.nodes())[:50]
    H = G.subgraph(sub_nodes)

    plt.figure(figsize=(12, 8))
    nx.draw(H, with_labels=True, node_size=500, font_size=8)

    # Node features: identity (1-hot for TF-IDF vocab)
    x = torch.eye(len(vocab), dtype=torch.float)
    y = torch.tensor(y, dtype=torch.long)

    # Edges
    edge_index = torch.tensor(
        [[vocab_index[u], vocab_index[v]] for u, v in G.edges() if u in vocab_index and v in vocab_index],
        dtype=torch.long
    ).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    
    all_logs = []

    # === Cross-validation folds ===
    for fold, (train_idx, test_idx) in enumerate(get_folds(X, y, n_splits=10), start=1):
        train_mask = torch.zeros(len(y), dtype=torch.bool)
        test_mask = torch.zeros(len(y), dtype=torch.bool)
        train_mask[train_idx] = True
        test_mask[test_idx] = True

        data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            train_mask=train_mask,
            test_mask=test_mask
        )
        data.docs_tokens = tokens_list
        data.vocab_index = vocab_index
        data.tfidf_matrix = X

        # === Init model ===
        model = GNNClassifier(input_dim=data.x.shape[1], hidden_dim=256, dropout=0.2)
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        # Class weights for imbalance
        class_counts = torch.bincount(y)          # counts [n_fake, n_real]
        class_weights = 1.0 / class_counts.float() 
        class_weights = class_weights / class_weights.sum() * 2.0  # normalize
        class_weights = class_weights.to(torch.float)

        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # === Train loop ===
        print(f"\nFold {fold}")
        for epoch in range(50):  # train longer (50–100 is reasonable)
            logs = []
            loss = train_gnn(model, data, optimizer, criterion, device="cpu")
            acc, preds, probs,cm = test_gnn(model, data, device="cpu")
            
            tn, fp, fn, tp = cm.ravel()

            pred_dist = torch.bincount(preds[data.test_mask], minlength=2).tolist()
            true_dist = torch.bincount(y[data.test_mask], minlength=2).tolist()

            logs.append({
                    "Fold": fold,
                    "Epoch": epoch,
                    "Loss": loss,
                    "Accuracy": acc,
                    "Pred_Real": pred_dist[0],
                    "Pred_Fake": pred_dist[1],
                    "True_Real": true_dist[0],
                    "True_Fake": true_dist[1],
                    "TN": tn, "FP": fp, "FN": fn, "TP": tp
                })

            all_logs.extend(logs)
            

            print(f"Epoch {epoch:02d} | Loss: {loss:.4f} | Acc: {acc:.4f}")
            print(f"  Confusion matrix:\n{cm}")
        df = pd.DataFrame(all_logs)
        df.to_csv("training_log.csv", index=False)
    plt.show()