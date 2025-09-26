"""
gnn_model.py
Graph Neural Network (GraphSAGE) to help with fake news classification
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from sklearn.metrics import confusion_matrix
# from src.preprocessing import preprocess_dataset
# from src.feature_extraction import build_word_occurrence_graph
import torch.optim as optim
# from src.utils import RAW_DIR, PROCESSED_DIR, set_seed, SEED
# from src.train import prepare_data, get_folds

class GNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=2, dropout=0.5):
        super(GNNClassifier, self).__init__()

        #GraphSAGE layers tentative (add?)
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)

        #BatchNorm layers
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        #Fully connected output layer
        self.fc = nn.Linear(hidden_dim, num_classes)

        self.dropout = dropout

    def forward(self, x, edge_index, docs_tokens=None, vocab_index=None, tfidf_matrix=None, vectorizer=None):
        # --- First GraphSAGE layer ---
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # --- Second GraphSAGE layer ---
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        word_embeddings = F.relu(x)   # shape: [vocab_size, hidden_dim]
        
        # --- Pool word embeddings → doc embeddings ---
        doc_embeddings = []
        for doc_idx, tokens in enumerate(docs_tokens):
            word_ids = [vocab_index[w] for w in tokens if w in vocab_index]
            if word_ids:
                weights = tfidf_matrix[doc_idx, word_ids]

                # convert sparse to dense if needed
                if hasattr(weights, "toarray"):
                    weights = torch.tensor(weights.toarray().ravel(), device=x.device, dtype=torch.float)
                else:
                    weights = torch.tensor(weights, device=x.device, dtype=torch.float)

                we = word_embeddings[word_ids]  # [num_words_in_doc, hidden_dim]

                # --- TF-IDF weighted mean pooling ---
                if weights.sum() > 0:
                    emb = (we * weights.unsqueeze(1)).sum(dim=0) / (weights.sum() + 1e-8)
                else:
                    emb = we.mean(dim=0)
            else:
                # doc has no valid words in vocab
                emb = torch.zeros(word_embeddings.size(1), device=x.device)

            doc_embeddings.append(emb)

        doc_embeddings = torch.stack(doc_embeddings)  # [num_docs, hidden_dim]

        # --- Classification head ---
        out = self.fc(doc_embeddings)  # shape: [num_docs, num_classes]
        return out

    
def train_gnn(model, data, optimizer, criterion, device):
    model.train()
    optimizer.zero_grad()
    out = model(
        data.x, data.edge_index,
        data.docs_tokens, data.vocab_index,
        tfidf_matrix=data.tfidf_matrix
    )

    
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def test_gnn(model, data, device):
    model.eval()
    out = model(
        data.x, data.edge_index,
        data.docs_tokens, data.vocab_index,
        tfidf_matrix=data.tfidf_matrix
    )
    preds = out.argmax(dim=1)

    correct = (preds[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())

    cm = confusion_matrix(
        data.y[data.test_mask].cpu().numpy(),
        preds[data.test_mask].cpu().numpy(),
        labels=[0, 1]  # 0 = fake, 1 = real
    )

    return acc, preds, out, cm


def graph_to_pyg_data(G, X, y, train_idx, test_idx):
    """
    Convert NetworkXGraph + TF-IDF features + labels into PyTorch Geometric Data Object
    G : word co-occurrence graph
    X : TF-IDF features (docs x vocabulary)
    y : Labels
    train_idx : training indices
    test_idx : testing indices
    """
    # --- Edges ---
    edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
    if edge_index.numel() == 0:
        raise ValueError("Graph has no edges. Recheck graph building.")

    # Make edges bidirectional
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    # --- Features ---
    if hasattr(X, "toarray"):   # handle sparse TF-IDF
        X = X.toarray()
    x = torch.tensor(X, dtype=torch.float)

    y = torch.tensor(y, dtype=torch.long)

    # --- Masks ---
    train_mask = torch.zeros(y.size(0), dtype=torch.bool)
    test_mask = torch.zeros(y.size(0), dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        test_mask=test_mask
    )
    return data

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction =reduction

    def forward(self, logits, targets):
        # logits: [N, C], targets: [N]
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        pt = probs[torch.arange(targets.size(0)), targets]

        focal_term = (1 - pt).pow(self.gamma)
        loss = -focal_term * log_probs[torch.arange(targets.size(0)), targets]

        if self.alpha is not None:
            at = self.alpha[targets]
            loss = at * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss