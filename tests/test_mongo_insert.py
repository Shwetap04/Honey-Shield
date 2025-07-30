from pymongo import MongoClient
from datetime import datetime, timezone
from data_ingestion.platform_connectors import generate_synthetic_chat_log


def insert_chat_log_to_mongo(chat_log):
    # Connect to local MongoDB instance
    client = MongoClient("mongodb://localhost:27017/")
    db = client["honeyshield"]
    collection = db["chat_logs"]

    result = collection.insert_one(chat_log)
    print("✓ Chat log inserted with ID:", result.inserted_id)
    return result.inserted_id


if __name__ == "__main__":
    chat_log = generate_synthetic_chat_log()
    insert_chat_log_to_mongo(chat_log)
