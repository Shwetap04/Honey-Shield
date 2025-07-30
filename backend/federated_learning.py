# backend/federated_learning.py

def simulate_federated_learning_v2(data_by_node):
    """
    Simulate federated learning:
    - Each node computes local averages
    - The server aggregates global metrics from local node stats
    """
    aggregated = {"message_length": [], "user_profile_risk": []}

    for node_id, node_data in data_by_node.items():
        if not node_data:
            continue  # Skip empty data lists to avoid division by zero

        ml_avg = sum(d.get("message_length", 0) for d in node_data) / len(node_data)
        risk_avg = sum(d.get("user_profile_risk", 0) for d in node_data) / len(node_data)

        aggregated["message_length"].append(ml_avg)
        aggregated["user_profile_risk"].append(risk_avg)

    # Prevent division by zero in global averaging
    total_nodes = len(aggregated["message_length"]) or 1

    return {
        "global_avg_msg_length": sum(aggregated["message_length"]) / total_nodes,
        "global_avg_profile_risk": sum(aggregated["user_profile_risk"]) / total_nodes
    }

def detect_social_drift(node_data, threshold=5):
    """
    Detect social drift based on message length variance.
    Returns True if the difference between max and min message length
    exceeds a given threshold.
    """
    lengths = [d.get("message_length", 0) for d in node_data]

    if not lengths:
        return False  # No data to detect drift

    drift_score = max(lengths) - min(lengths)
    return drift_score > threshold
