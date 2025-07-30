def preprocess_chat_data(chat_logs):
    return [
        {
            "user": entry["user"],
            "timestamp": entry["timestamp"],
            "message": entry["message"],
            "message_length": len(entry["message"]),
        }
        for entry in chat_logs
    ]

def personalize_chat_data(chat_data, node_id):
    bias = len(node_id)
    for chat in chat_data:
        chat["message_length"] += bias % 3
        chat["user_profile_risk"] = 0.1 * bias
    return chat_data
