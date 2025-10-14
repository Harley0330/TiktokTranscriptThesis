from src.preprocessing import preprocess_dataset, save_preprocessed_dataset
from src.feature_extraction import build_tfidf, build_word_occurrence_graph
from src.train import prepare_data, get_folds
from src.gnn_model import train_gnn_cv
import matplotlib.pyplot as plt
import torch
import random
import numpy as np
from src.utils import RAW_DIR, PROCESSED_DIR, set_seed, SEED
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
    train_gnn_cv(X, y, tokens_list, vocab_index, x, edge_index, device=device, n_splits=15)


    