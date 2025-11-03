import torch
from src.utils import RAW_DIR, MODELS_DIR, PROCESSED_DIR
from src.preprocessing import preprocess_dataset
from src.train import prepare_data
from src.feature_extraction import build_word_occurrence_graph
from src.gnn_model import train_gnn_cv
from sklearn.decomposition import TruncatedSVD
import numpy as np

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n=== Step 1: Load and preprocess dataset ===")
    dataset_path = RAW_DIR
    output_path = PROCESSED_DIR / "data_cleaned_formatted.csv"

    df = preprocess_dataset(dataset_path)

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

    # --- Train using CV ---
    out_path = MODELS_DIR / "gnn_training_log.csv"
    train_gnn_cv(
        X=X,
        y=y,
        tokens_list=tokens_list,
        vocab_index=vocab_index,
        x=x,
        edge_index=edge_index,
        device=device,
        n_splits=15,
        out_path=out_path
    )

    print("\n✅ GNN training complete! Check the log at:", out_path)

if __name__ == "__main__":
    main()
