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
from preprocessing import preprocess_dataset
from feature_extraction import build_word_occurrence_graph
import torch.optim as optim
from utils import RAW_DIR, PROCESSED_DIR, set_seed, SEED
from train import prepare_data, get_folds


class GNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=2, dropout=0.5):
        super(GNNClassifier, self).__init__()

        #GraphSAGE layers tentative
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)

        #BatchNorm layers
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        #Fully connected output layer
        self.fc = nn.Linear(hidden_dim, 1)

        self.dropout = dropout

    def forward(self, x, edge_index):
        #First GraphSAGE layer
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        #Second GraphSAGE layer
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)

        #Final classifier
        x = self.fc(x)
        return torch.sigmoid(x).squeeze()
    
def train_gnn(model, data, optimizer, criterion, device):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)

    loss = criterion(out[data.train_mask], data.y[data.train_mask].float())
    loss.backward()
    optimizer.step()
    return loss.item()

def test_gnn(model, data, device):
    model.eval()
    out = model(data.x, data.edge_index)
    preds = (out > 0.5).long()

    correct = (preds[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    return acc, preds, out

def graph_to_pyg_data(G, X, y, train_idx, test_idx):
    """
    Convert NetworkXGraph + TF-IDF features + labels into PyTorch Geometric Data Object
    G : word co-occurrence graph
    X : TF-IDF features (docx x vocabulary)
    y : Labels
    train_idx : training
    test_idx : testing
    """

    edge_index = torch.tensor(list(G.edges), dtype = torch.long).t().contiguous()
    if edge_index.numel() == 0:
        raise ValueError("Graph has no edges. Recheck graph building")
    edge_index = torch.cat([edge_index], edge_index.flip[0], dim=1)

    # ---- TF-IDF to tensor
    if hasattr(X, "toarray"):
        X = X.toarray()
    x= torch.tensor(X, dtype=torch.float)

    y = torch.tensor(y,dtype=torch.long)

    train_mask = torch.zeros(y.size(0), dtype=torch.bool)
    test_mask = torch.zeros(y.size(0),dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True

    data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, test_mask=test_mask)

    return data

if __name__ == "__main__":
    from train import prepare_data, get_folds

    # === Load dataset ===
    dataset_path = RAW_DIR
    df = preprocess_dataset(dataset_path)
    X, y, vectorizer = prepare_data(df, dataset_path, max_features=5000)

    # Convert TF-IDF to tensor
    if hasattr(X, "toarray"):
        X = X.toarray()
    x = torch.tensor(X, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.long)

    # === Build document similarity graph (optional step) ===
    # For now, let’s connect docs with cosine similarity > threshold
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(X)
    edge_index = torch.nonzero(torch.tensor(sim_matrix > 0.2), as_tuple=False).t().contiguous()

for fold, (train_idx, test_idx) in enumerate(get_folds(X,y,n_splits=5), start=1):
    #train_mask = torch.tensor([i in train_idx for i in range(len(y))]),
    #test_mask = torch.tensor([i in test_idx for i in range(len(y))])
    
    train_mask = torch.from_numpy(train_idx).long()
    test_mask = torch.from_numpy(test_idx).long()
    
    train_mask =  torch.zeros(len(y), dtype=torch.bool)
    test_mask = torch.zeros(len(y), dtype=torch.bool)

    train_mask[train_idx] = True
    test_mask[test_idx] = True

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask = train_mask,
        test_mask = test_mask
    )

    # === Init model ===
    model = GNNClassifier(input_dim=data.x.shape[1], hidden_dim=64, dropout=0.5)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    # === Train loop ===
    print(f"\nFold {fold}")
    for epoch in range(20):
        loss = train_gnn(model, data, optimizer, criterion, device="cpu")
        acc, preds, probs = test_gnn(model, data, device="cpu")
        print(f"Epoch {epoch} - Loss: {loss:.4f}, Acc: {acc:.4f}")
