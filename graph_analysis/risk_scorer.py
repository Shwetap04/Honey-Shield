# /graph_analysis/risk_scorer.py

import re

def risk_score(text: str) -> float:
    """
    Simple risk scoring function based on suspicious keyword matching.
    Each detected keyword contributes to the risk score.
    The total score is capped at 1.0.
    
    This update makes the scoring more sensitive:
      - 1 keyword match -> 0.5 risk score (nudging threshold)
      - 2 or more matches -> 1.0 risk score (alert threshold)
    """
    if not text:
        return 0.0

    text_lower = text.lower()
    suspicious_keywords = [
        "secret", "trustworthy", "work alone", "admirers", 
        "confidential", "phone", "allowed", "compliment", 
        "charm", "help", "meet", "private"
    ]

    matches = sum(bool(re.search(r"\b" + re.escape(k) + r"\b", text_lower)) for k in suspicious_keywords)

    if matches == 0:
        return 0.0
    elif matches == 1:
        return 0.5
    else:
        return 1.0
