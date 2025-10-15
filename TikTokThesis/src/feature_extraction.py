"""
--------
Convert preprocessed text tokens into numerical features:
- TF-IDF Vectors
- Word occurrence graph

"""

from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx

def build_tfidf(corpus, max_features = 10000): # Builds a TF-IDF representation of the corpus
    
    vectorizer = TfidfVectorizer(max_features=max_features, min_df=5, max_df=0.9, ngram_range=(1,2),norm="l2")
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer

def build_word_occurrence_graph(tokens_list, window_size, vocab_set=None): # Builds a word occurence graph from tokenized transcripts

    G = nx.Graph()

    for tokens in tokens_list:
        for i in range(len(tokens) - window_size + 1):
            window = tokens[i:i + window_size]

            if vocab_set is not None:
                window = [w for w in window if w in vocab_set]
            for w1 in window:
                for w2 in window:
                    if w1 != w2:
                        if G.has_edge(w1, w2):
                            G[w1][w2]["weight"] += 1
                        else:
                            G.add_edge(w1,w2, weight=1)
    return G
    