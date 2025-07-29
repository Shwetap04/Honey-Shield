#HoneyShield\graph_analysis\risk_scorer.py

import re
import logging
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import numpy as np
from typing import Tuple
import html
import bleach

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')

# Load transformer model for sentiment/phishing
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

# Keywords with fuzzy scoring (weighting suspicious content)
SUSPICIOUS_KEYWORDS = [
    "secret", "trustworthy", "work alone", "admirer", "confidential", "phone", "click", "verify", "urgent",
    "password", "account", "login", "update"
]

# Escape and sanitize input to prevent text injection

def sanitize_text(text: str) -> str:
    clean_text = html.unescape(text)
    clean_text = bleach.clean(clean_text, tags=[], strip=True)
    return clean_text[:1000]  # truncate to safe length

# Predict sentiment risk using transformer

def predict_sentiment_score(text: str) -> float:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = softmax(logits.numpy()[0])
    negative_prob = float(probs[0])
    return negative_prob

# Regex and keyword-based scoring (rule-based)

def rule_based_score(text: str) -> float:
    text = text.lower()
    matches = sum(bool(re.search(rf"\\b{k}\\b", text)) for k in SUSPICIOUS_KEYWORDS)
    return min(matches / 5.0, 1.0)

# Final risk scoring with ensemble logic

def compute_risk_score(text: str) -> Tuple[float, dict]:
    clean = sanitize_text(text)
    sentiment_score = predict_sentiment_score(clean)
    rule_score = rule_based_score(clean)

    # Ensemble with weight logic
    final_score = 0.6 * rule_score + 0.4 * sentiment_score

    # Log audit
    logging.info("Text sanitized: %s", clean[:80])
    logging.info("Sentiment Score: %.3f | Rule Score: %.3f | Final Score: %.3f", sentiment_score, rule_score, final_score)

    return final_score, {
        "sentiment_score": sentiment_score,
        "rule_score": rule_score,
        "final_score": final_score,
        "flags": [
            k for k in SUSPICIOUS_KEYWORDS if re.search(rf"\\b{k}\\b", clean.lower())
        ]
    }

# Thresholds can be set dynamically from quantiles

def risk_level(score: float, quantiles=(0.4, 0.7)) -> str:
    if score >= quantiles[1]:
        return "high"
    elif score >= quantiles[0]:
        return "medium"
    return "low"

# Example usage (disable in prod)
if __name__ == "__main__":
    sample = "Hey click this secret link to update your password quickly."
    score, details = compute_risk_score(sample)
    print("Risk:", risk_level(score))
    print(details)
