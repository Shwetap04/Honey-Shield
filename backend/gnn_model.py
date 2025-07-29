#HoneyShield\backend\gnn_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
import numpy as np

# --- GNN Model ---
class GNNModel(nn.Module):
    def __init__(self, in_features, hidden_dim=64, out_classes=2, gnn_type="gcn", num_layers=2):
        super(GNNModel, self).__init__()
        self.gnn_type = gnn_type.lower()
        self.num_layers = num_layers

        GNNLayer = {
            "gcn": GCNConv,
            "gat": GATConv,
            "sage": SAGEConv
        }[self.gnn_type]

        self.layers = nn.ModuleList()
        self.layers.append(GNNLayer(in_features, hidden_dim))

        for _ in range(num_layers - 2):
            self.layers.append(GNNLayer(hidden_dim, hidden_dim))

        self.layers.append(GNNLayer(hidden_dim, out_classes))

        self.dropout = nn.Dropout(p=0.3)
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers - 1)])

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.layers[:-1]):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        return self.layers[-1](x, edge_index)

# --- Feature Normalization ---
def normalize_node_features(x_np):
    scaler = StandardScaler()
    return torch.tensor(scaler.fit_transform(x_np), dtype=torch.float)

# --- Train the Model ---
def train_model(model, data, epochs=100, lr=0.005, weight_decay=5e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        _, pred = out.max(dim=1)
        correct = int((pred[data.val_mask] == data.y[data.val_mask]).sum())
        acc = correct / int(data.val_mask.sum())
        print(f"Epoch {epoch+1:03d} | Loss: {loss.item():.4f} | Val Acc: {acc:.4f}")

    return model

# --- Evaluate with Full Metrics ---
def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        preds = out.argmax(dim=1).cpu().numpy()
        probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()
        y_true = data.y.cpu().numpy()

        test_mask = data.test_mask.cpu().numpy() if hasattr(data, "test_mask") else np.ones_like(y_true, dtype=bool)

        y_pred = preds[test_mask]
        y_score = probs[test_mask]
        y_gold = y_true[test_mask]

        metrics = {
            "Accuracy": accuracy_score(y_gold, y_pred),
            "Precision": precision_score(y_gold, y_pred, zero_division=0),
            "Recall": recall_score(y_gold, y_pred, zero_division=0),
            "F1 Score": f1_score(y_gold, y_pred, zero_division=0),
            "AUC": roc_auc_score(y_gold, y_score)
        }

        print("\n--- Evaluation Metrics ---")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

        return preds, probs, metrics
