# backend/run_analysis.py

from data_ingestion.platform_connectors import load_sample_chats
from . import chat_analysis  # Relative import - keep __init__.py present!

def main():
    chats = load_sample_chats(num_chats=3)  # Load 3 synthetic chat sessions

    for chat_idx, chat in enumerate(chats, 1):
        print(f"\n=== Chat Session {chat_idx} [{chat.get('platform', 'unknown')}] ===")
        messages = chat.get('messages', [])
        for msg_idx, message in enumerate(messages, 1):
            # Feed the text of each message through your main analysis function
            result = chat_analysis.analyze_chat_message(message['text'])

            print(f"Message {msg_idx} | {message['sender']} → {message['receiver']}")
            print(f"  Text: {message['text']}")
            print(f"  Risk Score: {result.get('risk_score', 'N/A')}")
            print(f"  Flags: {result.get('flags', [])}")
            print(f"  Explanation: {result.get('explanation', [])}")
            print("-" * 40)

if __name__ == "__main__":
    main()
