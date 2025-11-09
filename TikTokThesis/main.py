from src.preprocessing import preprocess_dataset, save_preprocessed_dataset
from src.feature_extraction import build_tfidf, build_word_occurrence_graph
from src.train import prepare_data, get_folds
from src.gnn_model import train_gnn_cv
from src.hybrid_rf_gnn import run_hybrid_rf
from src.random_forest import run_rf, run_rf_baseline_cv
#import matplotlib.pyplot as plt
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

TEXT FEATURE EXTRACTION PORTION
    - Uses TF-IDF algorithm to create and assign values to words found in the corpus
    - Builds a word occurence graph as input for GNN

MODEL TRAINING PORTION
    Initial Random Forest Classifier
        - Achieved 0.8010 to 0.8037 accuracy metric

    Graph Neural Network
        - 4 Layer GraphSAGE network
        - Utilized GraphNorm, Dropout, Residual Skip Connections, Early Stopping, Scheduling and CrossEntropyLoss
        - Has a method in which the class probabilities per row can be extracted later
        - Saves the model to be used later

    Final Hybrid Random Forest Classifier
        - Implements Text Features such as TF-IDF and GNN Probabilities (extracted from Trained Model)
        - Optimizes alpha per fold (15)
        - Optimizes Threshold per fold (15)
        - Utilizes OOB to test performance on unseen data
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

    #Baseline (Random Forest only)
    #run_rf_baseline_cv(df)

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

    # Training GNN Model
    #train_gnn_cv(X, y, tokens_list, vocab_index, x, edge_index, device=device, n_splits=15)

    # Hybrid RF-GNN Model
    vocab_tfidf = set(vectorizer.get_feature_names_out())
    G = G.subgraph([w for w in G.nodes() if w in vocab_tfidf]).copy()
    vocab_index = {w: i for i, w in enumerate(G.nodes())}
    x = torch.eye(len(G.nodes()), dtype=torch.float, device=device)

    run_hybrid_rf(df, X, y, tokens_list, vectorizer, G, vocab_index, device, random_state=42)

    