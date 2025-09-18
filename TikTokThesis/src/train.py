"""
Split the dataset
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from feature_extraction import build_tfidf
from preprocessing import preprocess_dataset

def prepare_data(csv_path, max_features=7500):
    """
    Load dataset, preprocess transcript, and generate TF-IDF features
    Return feature matrix(X), labels (y), and vectorizer
    """

    #Load and preprocess
    df = preprocess_dataset(csv_path)

    #Map labels
    df["label"] = df["annotation"].map({"fake": 1, "real": 0})

    # Print dataset distribution
    print("\n Dataset Annotation Distribtuion: ")
    print(df["label"].value_counts())
    print(df["label"].value_counts(normalize=True).round(3))
    #Reconvert tokens back to text
    corpus = [" ".join(tokens) for tokens in df["tokens"]]

    #Build TF-IDF
    X, vectorizer = build_tfidf(corpus, max_features=max_features)
    y = df["label"].values

    return X, y, vectorizer

# def stratified_kfold_split(X, y, n_splits = 5):
#     """
#     Splits dataset by using stratified K-Fold splitting
#     Prints fold sizes and label balances
#     """

#     kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

#     fold = 1
#     for train_index, test_index in kf.split(X,y):
#         print(f"\n Fold{fold}")
#         print("Train size: ", len(train_index), " Test size: ", len(test_index))
#         print("Fake ratio in train: ", np.mean(y[train_index]))
#         print("Fake ratio in test: ", np.mean(y[test_index]))

#         # Extract train test splits
#         X_train, X_test = X[train_index], X[test_index]
#         y_train, y_test = y[train_index], y[test_index]

#         fold += 1

def get_folds(X,y, n_splits=5, random_state=42):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train_idx, test_idx, in kf.split(X,y):
        yield train_idx, test_idx