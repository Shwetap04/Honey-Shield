#HoneyShield\backend\chat_analysis.py
import re
import unicodedata
from rapidfuzz import fuzz
from transformers import pipeline
from datetime import datetime, timedelta
import torch

# === Load Models Once ===
sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
phishing_model = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-sms-spam-detection")  # or your fine-tuned version

# === Keyword Lists ===
MANIPULATION_KEYWORDS = [
    "trust me", "keep secret", "urgent", "act fast", "don't tell", "just between us"
]

FLATTERY_KEYWORDS = [
    "baby", "sweetie", "gorgeous", "angel", "my queen", "cutie", "handsome"
]

SUSPICIOUS_KEYWORDS = [
    "help me", "send", "transfer", "gift", "emergency", "bitcoin", "hospital", "loan", "pay"
]

# === Risk Weights ===
RISK_WEIGHTS = {
    "negative_sentiment": 30,
    "manipulation": 30,
    "suspicious_request": 40,
    "flattery": 15,
    "phishing_intent": 35,
}

# === Preprocess and Clean ===
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    normalized = unicodedata.normalize("NFKC", text)
    lowered = normalized.lower()
    cleaned = re.sub(r"[^\w\s]", " ", lowered)
    return re.sub(r"\s+", " ", cleaned).strip()

# === Fuzzy Keyword Match ===
def fuzzy_contains(text, keywords, threshold=85):
    matched = 0
    for kw in keywords:
        for word in text.split():
            if fuzz.ratio(kw, word) > threshold:
                matched += 1
    return matched

# === Sentiment Analysis ===
def analyze_sentiment(text):
    try:
        result = sentiment_model(text[:512])[0]
        label = result['label'].lower()
        score = result['score']
        sentiment = "positive" if label == "positive" and score > 0.7 else "negative" if label == "negative" and score > 0.7 else "neutral"
        return sentiment, score
    except Exception:
        return "neutral", 0.0

# === Phishing Intent Detection ===
def detect_phishing(text):
    try:
        result = phishing_model(text[:512])[0]
        return result['label'].lower() == "spam"
    except Exception:
        return False

# === Risk Calculation ===
def compute_risk(sentiment, flags, phishing, user_history=None):
    risk = 0
    if sentiment == "negative":
        risk += RISK_WEIGHTS["negative_sentiment"]
    for flag, count in flags.items():
        base = RISK_WEIGHTS.get(flag, 0)
        scaled = min(count * (base // 2), base)
        risk += scaled

        # Adjust for repeated behavior
        if user_history and flag in user_history and user_history[flag] > 2:
            risk += 5

    if phishing:
        risk += RISK_WEIGHTS["phishing_intent"]

    return min(100, risk)

# === Explanation Builder ===
def explain_analysis(sentiment, flags, phishing):
    explanation = []
    if sentiment == "negative":
        explanation.append("Negative sentiment detected.")
    for flag, count in flags.items():
        explanation.append(f"{count} instance(s) of {flag.replace('_', ' ')} detected.")
    if phishing:
        explanation.append("Phishing-like intent detected.")
    return explanation

# === Main Analysis Function ===
def analyze_chat_message(message, timestamp=None, user_id=None, user_history=None):
    cleaned = preprocess_text(message)
    sentiment, score = analyze_sentiment(cleaned)
    phishing = detect_phishing(cleaned)

    flag_counts = {
        "manipulation": fuzzy_contains(cleaned, MANIPULATION_KEYWORDS),
        "flattery": fuzzy_contains(cleaned, FLATTERY_KEYWORDS),
        "suspicious_request": fuzzy_contains(cleaned, SUSPICIOUS_KEYWORDS),
    }

    flag_counts = {k: v for k, v in flag_counts.items() if v > 0}
    risk_score = compute_risk(sentiment, flag_counts, phishing, user_history)

    explanation = explain_analysis(sentiment, flag_counts, phishing)

    result = {
        "sentiment": sentiment,
        "score": round(score, 3),
        "flags": list(flag_counts.keys()),
        "risk_score": risk_score,
        "explanation": explanation,
    }

    if timestamp:
        result["timestamp"] = timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp)

    return result
