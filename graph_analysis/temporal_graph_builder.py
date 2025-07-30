# /graph_analysis/temporal_graph_builder.py

import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from collections import defaultdict
import numpy as np

# --- Utility to Convert Timestamp to Hour Bucket ---
def to_hour_bucket(timestamp):
    return datetime.fromisoformat(timestamp).strftime("%Y-%m-%d-%H")

# --- Core: Temporal Social Graph Construction ---
def build_temporal_graph(messages, min_messages_per_node=1):
    """
    Build a temporal social graph with node & edge features.

    Args:
        messages (list of dicts): Each dict at least has 'sender', 'receiver', 'timestamp', 'risk_score', and optionally 'sent_score'.
    Returns:
        torch_geometric.data.Data object
    """
    node_features = defaultdict(lambda: {'count': 0, 'sent_sum': 0, 'risk_sum': 0})
    edge_list = []
    edge_attrs = []
    timestamps = []

    for msg in messages:
        sender = msg['sender']
        receiver = msg['receiver']
        time = msg['timestamp']
        risk_score = msg.get('risk_score', 0.0)
        sent_score = msg.get('sentiment_score') if 'sentiment_score' in msg else 0.0  # optionally fill

        # Node aggregation
        node_features[sender]['count'] += 1
        node_features[sender]['sent_sum'] += sent_score
        node_features[sender]['risk_sum'] += risk_score

        # Bidirectional edges
        for direction in [(sender, receiver), (receiver, sender)]:
            edge_list.append(direction)
            edge_attrs.append([sent_score, risk_score])
            timestamps.append(datetime.fromisoformat(time).timestamp())

    # Filter nodes with sufficient messages
    filtered_nodes = {k for k, v in node_features.items() if v['count'] >= min_messages_per_node}
    node_map = {node: i for i, node in enumerate(filtered_nodes)}

    # Node features tensor
    x_raw = []
    for node in filtered_nodes:
        v = node_features[node]
        avg_sent = v['sent_sum'] / v['count'] if v['count'] else 0
        avg_risk = v['risk_sum'] / v['count'] if v['count'] else 0
        x_raw.append([v['count'], avg_sent, avg_risk])
    if x_raw:
        x_np = StandardScaler().fit_transform(np.array(x_raw))
        x = torch.tensor(x_np, dtype=torch.float)
    else:
        x = torch.zeros((0, 3), dtype=torch.float)

    # Build edge_index and edge_attr
    edges = [(src, tgt) for (src, tgt), ts in zip(edge_list, timestamps)
             if src in filtered_nodes and tgt in filtered_nodes]
    edge_index = torch.tensor([[node_map[src], node_map[tgt]] for src, tgt in edges], dtype=torch.long).t().contiguous() if edges else torch.zeros((2,0), dtype=torch.long)
    edge_attr = torch.tensor(edge_attrs[:len(edges)], dtype=torch.float) if edges else torch.zeros((0,2), dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, node_map=node_map)

# --- Phase 2: Build Time-Windowed Graph Snapshots ---
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
