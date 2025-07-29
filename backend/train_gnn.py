#HoneyShield\backend\train_gnn.py

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support, roc_auc_score
import os

from gnn_model import GNNModel

# Simulated dummy graph data (replace with real graph from temporal_graph_builder)
x = torch.tensor([[0.2, 0.1, 0.4], [0.3, 0.8, 0.1], [0.6, 0.5, 0.9], [0.1, 0.2, 0.2]], dtype=torch.float)
edge_index = torch.tensor([[0, 1, 2, 3, 1, 2], [1, 0, 3, 2, 2, 1]], dtype=torch.long)  # bidirectional
y = torch.tensor([0, 1, 1, 0], dtype=torch.long)  # labels: 0-safe, 1-risky

data = Data(x=x, edge_index=edge_index, y=y)
data.train_mask = torch.tensor([True, True, False, False], dtype=torch.bool)
data.test_mask = ~data.train_mask

model = GNNModel(in_channels=3, hidden_channels=8, out_channels=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    model.eval()
    _, pred = out[data.test_mask].max(dim=1)
    correct = pred.eq(data.y[data.test_mask]).sum().item()
    total = data.test_mask.sum().item()
    acc = correct / total
    print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}, Test Accuracy: {acc:.4f}")

# Metrics
model.eval()
probs = F.softmax(model(data.x, data.edge_index), dim=1)[data.test_mask]
preds = probs.argmax(dim=1).cpu().numpy()
true = data.y[data.test_mask].cpu().numpy()

f1 = f1_score(true, preds)
precision, recall, _, _ = precision_recall_fscore_support(true, preds, average='binary')
try:
    auc = roc_auc_score(true, probs[:, 1].detach().numpy())
except ValueError:
    auc = 0.0  # if only one class in y_test

print(f"\nFinal Evaluation:\nF1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | AUC: {auc:.4f}")

# 🔒 Save model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/gnn_model.pth")
print("✅ Model saved to models/gnn_model.pth")
