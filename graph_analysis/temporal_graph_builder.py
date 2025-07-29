#HoneyShield\graph_analysis\temporal_graph_builder.py
import torch
from torch_geometric.data import Data
import networkx as nx
from sklearn.preprocessing import StandardScaler
from transformers import pipeline
from datetime import datetime
from collections import defaultdict
import numpy as np

# --- Transformer Sentiment Model (Phase 1 Replacement of TextBlob) ---
sentiment_pipeline = pipeline("sentiment-analysis")

def get_sentiment_score(text):
    result = sentiment_pipeline(text[:512])[0]
    return result['score'] if result['label'] == 'POSITIVE' else -result['score']

# --- Utility to Convert Timestamp to Hour Bucket ---
def to_hour_bucket(timestamp):
    return datetime.fromisoformat(timestamp).strftime("%Y-%m-%d-%H")

# --- Phase 1: Graph Construction ---
def build_temporal_graph(messages, min_messages_per_node=1):
    node_features = defaultdict(lambda: {'count': 0, 'sent_sum': 0, 'risk_sum': 0})
    edge_list = []
    edge_attrs = []
    timestamps = []

    for msg in messages:
        sender = msg['sender']
        receiver = msg['receiver']
        text = msg['text']
        time = msg['timestamp']

        sent_score = get_sentiment_score(text)
        risk_score = compute_risk_score(text)

        # Node Aggregation
        node_features[sender]['count'] += 1
        node_features[sender]['sent_sum'] += sent_score
        node_features[sender]['risk_sum'] += risk_score

        # Bidirectional Edges with Attributes
        for direction in [(sender, receiver), (receiver, sender)]:
            edge_list.append(direction)
            edge_attrs.append([sent_score, risk_score])
            timestamps.append(datetime.fromisoformat(time).timestamp())

    # Filter out nodes with too few messages
    filtered_nodes = {k for k, v in node_features.items() if v['count'] >= min_messages_per_node}
    node_map = {node: i for i, node in enumerate(filtered_nodes)}

    # Build x (node features)
    x_raw = []
    for node in filtered_nodes:
        v = node_features[node]
        avg_sent = v['sent_sum'] / v['count']
        avg_risk = v['risk_sum'] / v['count']
        x_raw.append([v['count'], avg_sent, avg_risk])

    x_np = StandardScaler().fit_transform(np.array(x_raw))
    x = torch.tensor(x_np, dtype=torch.float)

    # Build edge_index and edge_attr
    edges = [(src, tgt) for (src, tgt), ts in zip(edge_list, timestamps)
             if src in filtered_nodes and tgt in filtered_nodes]

    edge_index = torch.tensor([[node_map[src], node_map[tgt]] for src, tgt in edges], dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs[:len(edge_index[0])], dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# --- Risk Heuristic for Now (Placeholder) ---
def compute_risk_score(text):
    keywords = ["urgent", "click", "verify", "account", "password", "transfer"]
    text_l = text.lower()
    score = sum(1 for kw in keywords if kw in text_l)
    return min(score / len(keywords), 1.0)

# --- Phase 2: Graph Snapshots ---
def build_time_windowed_graphs(messages, window='hour'):
    buckets = defaultdict(list)
    for msg in messages:
        if window == 'hour':
            key = to_hour_bucket(msg['timestamp'])
        else:
            key = msg['timestamp'].split('T')[0]  # Daily bucket
        buckets[key].append(msg)

    graphs = {}
    for time_key, bucket_msgs in buckets.items():
        graphs[time_key] = build_temporal_graph(bucket_msgs)

    return graphs
