# backend/mongo_utils.py

from pymongo import MongoClient
from datetime import datetime
from cryptography.fernet import Fernet
import os
from dotenv import load_dotenv
load_dotenv()

# Load or generate encryption key
# In real deployments, store this in an environment variable or secure vault
SECRET_KEY = os.environ.get("FERNET_SECRET_KEY") or Fernet.generate_key().decode()
fernet = Fernet(SECRET_KEY.encode())

# MongoDB connection
client = MongoClient("mongodb://localhost:27017/")
db = client["honeyshield"]
collection = db["chat_logs"]

def encrypt_message(message: str) -> str:
    return fernet.encrypt(message.encode()).decode()

def decrypt_message(token: str) -> str:
    return fernet.decrypt(token.encode()).decode()

def insert_chat_log(user, message):
    encrypted_msg = encrypt_message(message)
    chat_entry = {
        "user": user,
        "encrypted_message": encrypted_msg,
        "timestamp": datetime.now().astimezone().isoformat()
    }
    result = collection.insert_one(chat_entry)
    return str(result.inserted_id)

def fetch_recent_chats(limit=100):
    encrypted_docs = collection.find().sort("timestamp", -1).limit(limit)
    decrypted = []
    for doc in encrypted_docs:
        try:
            decrypted.append({
                "user": doc["user"],
                "message": decrypt_message(doc["encrypted_message"]),
                "timestamp": doc["timestamp"]
            })
        except Exception:
            decrypted.append({
                "user": doc["user"],
                "message": "[decryption_failed]",
                "timestamp": doc["timestamp"]
            })
    return decrypted
