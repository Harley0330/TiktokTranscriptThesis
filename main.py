from src.preprocessing import preprocess_dataset, save_preprocessed_dataset
from src.feature_extraction import build_tfidf, build_word_occurrence_graph
import matplotlib.pyplot as plt
import pandas
import networkx as nx
import random
"""
PREPROCESSING PORTION
    - Takes in the original file including the manually encoded transcripts
    - Declares an output path for the formatted dataset
    - Calls methods from preprocessing.py, which cleans and formats the dataset
"""
if __name__ == "__main__":
    dataset_path = "data/data_cleaned.csv"  # original file
    output_path = "data/data_cleaned_formatted.csv" #preprocessed path file
    df = preprocess_dataset(dataset_path)

    # for i, row in df.iterrows():
    #     print(f"Transcript {i+1}: {row['transcript']}")
    #     print(f"Tokens {i+1}: {row['tokens']}\n")
    
    save_preprocessed_dataset(df,output_path)
    
    # Checking total number of tokens
    # Flatten all tokens into one big list
    all_tokens = [token for tokens in df["tokens"] for token in tokens]

    # Total number of tokens (all words across all transcripts)
    total_tokens = len(all_tokens)

    # Unique vocabulary size
    unique_tokens = len(set(all_tokens))

    print(total_tokens)


    # Convert tokens back to text for TF-IDF
    corpus = [" ".join(tokens) for tokens in df["tokens"]]

    #TF-IDF algorithm
    X, vectorizer = build_tfidf(corpus, max_features = 5000)
    print("TF-IDF shape:", X.shape)

    #Word occurence Graph
    G = build_word_occurrence_graph(df["tokens"], window_size = 2)
    print("Graph nodes:", len(G.nodes()))
    print("Graph edges:", len(G.edges()))

    sub_nodes = list(G.nodes())[:50]
    H = G.subgraph(sub_nodes)

    plt.figure(figsize=(12, 8))
    nx.draw(H, with_labels=True, node_size=500, font_size=8)
    plt.show()