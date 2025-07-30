# /graph_analysis/risk_scorer.py

import re

def risk_score(text: str) -> float:
    """
    Enhanced risk scoring using suspicious keyword/phrase detection.
    - 1 match -> 0.5 risk score (nudging threshold)
    - 2+ matches -> 1.0 risk score (alert threshold)
    This version detects both single words and phrases more flexibly.
    """
    if not text:
        return 0.0

    text_lower = text.lower()
    suspicious_keywords = [
        "secret", "trustworthy", "work alone", "admirers",
        "confidential", "phone", "allowed", "compliment",
        "charm", "help", "meet", "private"
    ]

    matches = 0
    for k in suspicious_keywords:
        if ' ' in k:  # Phrase: look for plain substring
            if k in text_lower:
                matches += 1
        else:  # Single word: word-boundary regex (no double-backslash needed)
            if re.search(rf"\b{re.escape(k)}\b", text_lower):
                matches += 1

    if matches == 0:
        return 0.0
    elif matches == 1:
        return 0.5
    else:
        return 1.0

def compute_composite_risk(
    gnn_prob: float,
    heuristic_flags: dict,
    red_team_risk: float = 0.0,
    weights: dict = None
) -> dict:
    """
    Blends GNN probability, heuristic flags, and red team risk into a final risk score.
    Returns dict with 'composite_score', 'components', and explanation text.

    Args:
        gnn_prob: Network-level risk probability from GNN.
        heuristic_flags: Dictionary of heuristic risk signals (e.g., counts/weights from chat analysis/red team).
        red_team_risk: Scalar risk value from red team simulation.
        weights: Optional dict to customize blending ratios.
    """
    if weights is None:
        weights = {
            "gnn": 0.5,
            "heuristic": 0.3,
            "red_team": 0.2
        }

    # Normalize and blend
    heuristic_score = sum(heuristic_flags.values()) if heuristic_flags else 0
    heuristic_norm = min(heuristic_score / max(len(heuristic_flags), 1), 1.0) if heuristic_flags else 0.0

    composite_score = (
        weights["gnn"] * gnn_prob +
        weights["heuristic"] * heuristic_norm +
        weights["red_team"] * red_team_risk
    )
    composite_score = min(max(composite_score, 0.0), 1.0)

    # Build explanation
    explanation = [f"GNN risk: {gnn_prob:.2f} (x{weights['gnn']})"]
    if heuristic_flags and sum(heuristic_flags.values()) > 0:
        explanation.append(
            f"Heuristic flags ({weights['heuristic']} weight): {heuristic_flags}"
        )
    if red_team_risk > 0:
        explanation.append(
            f"Red team sim risk: {red_team_risk:.2f} (x{weights['red_team']})"
        )

    return {
        "composite_score": composite_score,
        "components": {
            "gnn": gnn_prob,
            "heuristic": heuristic_norm,
            "red_team": red_team_risk
        },
        "explanation": explanation
    }
