"""
gnn_model.py
4-layer GraphSAGE with optional skip connections for fake news classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GraphNorm
from torch_geometric.data import Data
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import torch.optim as optim
import pandas as pd
from src.train import get_folds
from src.utils import MODELS_DIR, RESULTS_DIR

class GNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=2, dropout=0.5):
        super(GNNClassifier, self).__init__()

        # --- GraphSAGE layers ---
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, hidden_dim)
        self.conv4 = SAGEConv(hidden_dim, hidden_dim)

        # --- BatchNorm layers ---
        self.bn1 = GraphNorm(hidden_dim)
        self.bn2 = GraphNorm(hidden_dim)
        self.bn3 = GraphNorm(hidden_dim)
        self.bn4 = GraphNorm(hidden_dim)

        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, docs_tokens=None, vocab_index=None, tfidf_matrix=None):
        # --- First layer ---
        x1 = self.conv1(x, edge_index)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)

        # --- Second layer with skip connection ---
        x2 = self.conv2(x1, edge_index)
        x2 = self.bn2(x2)
        x2 = F.relu(x2 + x1)

        # --- Third layer with skip connection ---
        x3 = self.conv3(x2, edge_index)
        x3 = self.bn3(x3)
        x3 = F.relu(x3 + x2)

        # --- Fourth layer with skip connection ---
        x4 = self.conv4(x3, edge_index)
        x4 = self.bn4(x4)
        word_embeddings = F.relu(x4 + x3)

        # --- Document embeddings ---
        doc_embeddings = []
        for doc_idx, tokens in enumerate(docs_tokens):
            word_ids = [vocab_index[w] for w in tokens if w in vocab_index]
            if word_ids:
                weights = tfidf_matrix[doc_idx, word_ids]
                if hasattr(weights, "toarray"):
                    weights = torch.tensor(weights.toarray().ravel(), device=x.device, dtype=torch.float)
                else:
                    weights = torch.tensor(weights, device=x.device, dtype=torch.float)

                we = word_embeddings[word_ids]
                if weights.sum() > 0:
                    emb = (we * weights.unsqueeze(1)).sum(dim=0) / (weights.sum() + 1e-8)
                else:
                    emb = we.mean(dim=0)
            else:
                emb = torch.zeros(word_embeddings.size(1), device=x.device)
            doc_embeddings.append(emb)

        doc_embeddings = torch.stack(doc_embeddings)
        out = self.fc(doc_embeddings)
        return out


def train_gnn(model, data, optimizer, criterion, device="cpu"):
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


def test_gnn(model, data, device="cpu"):
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
        labels=[0, 1]
    )
    return acc, preds, out, cm


def graph_to_pyg_data(G, X, y, train_idx, test_idx):
    edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
    if edge_index.numel() == 0:
        raise ValueError("Graph has no edges. Recheck graph building.")

    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    if hasattr(X, "toarray"):
        X = X.toarray()

    x = torch.tensor(X, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.long)

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


def train_gnn_cv(X, y, tokens_list, vocab_index, x, edge_index, device="cpu", n_splits=15, out_path=RESULTS_DIR/"training_log.csv"):
    
    MODELS_DIR.mkdir(exist_ok=True, parents=True)
    all_logs = []

    for fold, (train_idx, test_idx) in enumerate(get_folds(X, y, n_splits=n_splits), start=1):
        print(f"\n=== Fold {fold}/{n_splits} ===")

        train_mask = torch.zeros(len(y), dtype=torch.bool, device=device)
        test_mask = torch.zeros(len(y), dtype=torch.bool, device=device)
        train_mask[train_idx] = True
        test_mask[test_idx] = True

        data = Data(
            x=x,
            edge_index=edge_index,
            y=torch.tensor(y, dtype=torch.long, device=device),
            train_mask=train_mask,
            test_mask=test_mask
        )
        data.docs_tokens = tokens_list
        data.vocab_index = vocab_index
        data.tfidf_matrix = X

        model = GNNClassifier(input_dim=data.x.shape[1], hidden_dim=64, dropout=0.2).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0015, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6)

        y_tensor = torch.tensor(y, dtype=torch.long, device=device)
        class_counts = torch.bincount(y_tensor)
        class_weights = (1.0 / class_counts.float())
        class_weights = class_weights / class_weights.sum() * 2.0
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(torch.float))

        best_model_state = None
        best_test_acc = 0.0
        patience, wait = 15, 0
        l1_lambda = 5e-6

        for epoch in range(70):
            # ---- Train ----
            model.train()
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.docs_tokens, data.vocab_index, tfidf_matrix=data.tfidf_matrix)
            train_loss = criterion(out[data.train_mask], data.y[data.train_mask])
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            train_loss = train_loss + l1_lambda * l1_norm
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            # ---- Evaluate ----
            data_test_mask_backup = data.test_mask.clone()
            data.test_mask = data.train_mask
            train_acc, _, _, train_cm = test_gnn(model, data, device=device)
            data.test_mask = data_test_mask_backup

            test_acc, preds, probs, test_cm = test_gnn(model, data, device=device)
            precision = precision_score(y[test_idx], preds[test_mask].cpu(), zero_division=0)
            recall = recall_score(y[test_idx], preds[test_mask].cpu(), zero_division=0)

            # --- Compute test loss ---
            with torch.no_grad():
                out_test = model(data.x, data.edge_index, data.docs_tokens, data.vocab_index, tfidf_matrix=data.tfidf_matrix)
                test_loss = criterion(out_test[data.test_mask], data.y[data.test_mask]).item()

            scheduler.step(test_acc)

            # ---- Early stopping ----
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_model_state = model.state_dict().copy()
                wait = 0
            else:
                wait += 1

            if wait >= patience:
                print(f"⏸ Early stopping triggered at epoch {epoch}")
                break

            # ---- Logging ----
            logs = {
                "Fold": fold,
                "Epoch": epoch,
                "Train_Loss": train_loss.item(),
                "Test_Loss": test_loss,
                "Train_Acc": train_acc,
                "Test_Acc": test_acc,
                "Precision": precision,
                "Recall": recall,
                "Train_CM": train_cm.tolist(),
                "Test_CM": test_cm.tolist(),
            }
            all_logs.append(logs)

            print(
                f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | "
                f"Prec: {precision:.4f} | Rec: {recall:.4f}"
            )

        # ---- Save best model ----
        if best_model_state is not None:
            model_path = MODELS_DIR / f"gnn_fold{fold}_best.pth"
            torch.save(best_model_state, model_path)
            print(f"✅ Saved best model for Fold {fold} to: {model_path} (Acc={best_test_acc:.4f})")

    df_logs = pd.DataFrame(all_logs)
    df_logs.to_csv(out_path, index=False)
    print(f"\n📊 Training logs saved to {out_path}")


@torch.no_grad()
def extract_gnn_probabilities(model, x, edge_index, docs_tokens, vocab_index, tfidf_matrix=None, device="cpu"):
    """
    Extract GNN probabilities per document (row)
    """
    model.eval()
    x = x.to(device)
    edge_index = edge_index.to(device)
    out = model(x, edge_index, docs_tokens, vocab_index, tfidf_matrix=tfidf_matrix)

    if out.ndim == 2 and out.shape[1] == 2:
        probs = F.softmax(out, dim=1)[:, 1]  
    else:
        probs = out.squeeze()

    return probs.detach().cpu().numpy().reshape(-1, 1)
