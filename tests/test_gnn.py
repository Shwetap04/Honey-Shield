from graph_analysis.temporal_graph_builder import build_graph_from_interactions
from graph_analysis.risk_scorer import score_graph

def test_pipeline():
    interactions = [
        {"source": "alice", "target": "bob", "timestamp": 1, "message": "Hey!"},
        {"source": "bob", "target": "alice", "timestamp": 2, "message": "How are you?"},
        {"source": "alice", "target": "eve", "timestamp": 3, "message": "Send me money urgently!"},
    ]
    data, user_idx = build_graph_from_interactions(interactions)
    scores, risk_levels = score_graph(data)
    print(scores, risk_levels)
