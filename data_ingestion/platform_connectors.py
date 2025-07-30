import random
import datetime
from uuid import uuid4
import json
from pathlib import Path
from backend import mongo_utils


def generate_synthetic_chat_log():
    users = ["alice", "bob", "charlie", "david"]
    manipulation_phrases = [
        "You can trust me completely", "Don’t tell anyone about this",
        "Just send me your details", "It’s urgent, please act now",
        "You owe me this"
    ]
    casual_phrases = [
        "How was your day?", "Let's meet for coffee tomorrow.",
        "What's your plan for the weekend?", "Did you complete the task?"
    ]

    platforms = ["whatsapp", "telegram", "instagram", "signal"]
    devices = ["android", "ios", "web"]

    sender = random.choice(users)
    receiver = random.choice([u for u in users if u != sender])
    message_count = random.randint(4, 8)

    def gen_message():
        risky = random.random() < 0.3  # 30% chance of being risky
        text = random.choice(manipulation_phrases if risky else casual_phrases)
        current_sender = random.choice([sender, receiver])
        current_receiver = receiver if current_sender == sender else sender
        return {
            "sender": current_sender,
            "receiver": current_receiver,
            "text": text,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "manipulation_flag": risky,
            "sentiment": random.choice(["positive", "neutral", "negative"]),
            "device": random.choice(devices)
        }

    chat_log = {
        "chat_id": str(uuid4()),
        "participants": [sender, receiver],
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "platform": random.choice(platforms),
        "messages": [gen_message() for _ in range(message_count)],
        "session_meta": {
            "encryption": random.choice(["E2EE", "None"]),
            "suspicious_score": round(random.uniform(0, 1), 2)
        }
    }

    return chat_log


def load_sample_chats(num_chats=5):
    """
    Generate a list of synthetic chat sessions for testing/integration.
    """
    return [generate_synthetic_chat_log() for _ in range(num_chats)]


def integrate_external_chat(json_data):
    manipulation_keywords = [
        "trust me", "don’t tell", "confidential", "urgent", "favor", "bank info", "transfer",
        "keep this between us", "secret", "act now", "just between us"
    ]

    def flag_manipulation(text):
        lowered = text.lower()
        return any(phrase in lowered for phrase in manipulation_keywords)

    def infer_sentiment(text):
        lowered = text.lower()
        if any(w in lowered for w in ["thank", "angel", "beautiful", "looking forward"]):
            return "positive"
        elif any(w in lowered for w in ["risky", "carefully", "secret", "confidential"]):
            return "neutral"
        else:
            return "neutral"

    messages = []
    participants = set()
    for msg in json_data:
        participants.add(msg["sender"])
        participants.add(msg["receiver"])
        messages.append({
            "sender": msg["sender"],
            "receiver": msg["receiver"],
            "text": msg["message"],
            "timestamp": msg["timestamp"],
            "manipulation_flag": flag_manipulation(msg["message"]),
            "sentiment": infer_sentiment(msg["message"]),
            "device": "unknown"
        })

    chat_log = {
        "chat_id": str(uuid4()),
        "participants": list(participants),
        "timestamp": json_data[0]["timestamp"],
        "platform": json_data[0]["platform"].lower(),
        "messages": messages,
        "session_meta": {
            "encryption": "Unknown",
            "suspicious_score": round(sum(m["manipulation_flag"] for m in messages) / len(messages), 2)
        }
    }

    return chat_log


if __name__ == "__main__":
    # Load synthetic chats
    sample_chats = load_sample_chats(num_chats=2)

    # Path to external real-world chat JSON
    external_chat_path = Path(__file__).parent / "sample_chats.json"
    if external_chat_path.exists():
        with external_chat_path.open() as f:
            external_data = json.load(f)
            external_chat = integrate_external_chat(external_data)
            sample_chats.append(external_chat)
    else:
        print("[Warning] sample_chats.json not found. Skipping external chat integration.")

    # Display and store combined chat logs
    for idx, chat in enumerate(sample_chats):
        print(f"--- Chat Session {idx + 1} ---")
        print(json.dumps(chat, indent=2))
        print()

        # Insert messages into MongoDB
        for msg in chat["messages"]:
            mongo_utils.insert_chat_log(msg["sender"], msg["text"])
