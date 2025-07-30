import time
from typing import Dict, Optional


# Parameters for nudging logic
RISK_THRESHOLD = 0.4           # Lowered threshold to catch moderate risk
ESCALATE_THRESHOLD = 0.8       # Higher risk => stronger alert
NOTIFICATION_TIMEOUT_SEC = 60  # Minimum seconds between nudges to the same user


# In-memory storage for last-notified times; replace with persistent storage in production
_last_notification_times: Dict[str, float] = {}

# Optional: in-memory log of issued interventions for audit/debug (can be saved to file/db)
_intervention_log = []


def should_notify(user_id: str) -> bool:
    """
    Prevent user fatigue by limiting frequency of notifications per user.
    
    Returns True if enough time has passed since the last notification for the user.
    """
    now = time.time()
    last_time = _last_notification_times.get(user_id, 0)
    if now - last_time >= NOTIFICATION_TIMEOUT_SEC:
        _last_notification_times[user_id] = now
        return True
    return False


def intervention_policy(
    user_id: str,
    message: str,
    risk_score: float,
    context: Optional[Dict] = None,
) -> Optional[Dict]:
    """
    Determines whether and what type of behavioral intervention (nudge or alert) to issue
    based on the risk score and notification cooldown.
    
    Args:
        user_id: Identifier of the user to notify.
        message: The original message prompting this risk.
        risk_score: Computed risk score (0.0 to 1.0).
        context: Optional dictionary with additional information to include.
        
    Returns:
        A dictionary with intervention details if notification is sent, else None.
    """
    if risk_score < RISK_THRESHOLD:
        return None  # Risk too low for notification

    if not should_notify(user_id):
        return None  # Notification cooldown active, skip notification

    if risk_score >= ESCALATE_THRESHOLD:
        action = "alert"
        text = (
            "⚠️ Security Alert: This conversation may be part of a social engineering attempt. "
            "Don't share confidential information. Trust your instincts and report suspicious interactions."
        )
    else:
        action = "nudge"
        text = (
            "🔍 Heads up: This chat shows signs of social manipulation. "
            "Be cautious before responding or sharing details."
        )

    intervention = {
        "action": action,             # "nudge" or "alert"
        "user_id": user_id,
        "message": text,
        "risk_score": risk_score,
        "original_message": message or "",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "context": context or {},
    }

    # Optional: log intervention for audit or debugging
    _intervention_log.append(intervention)

    return intervention


# Example standalone test/demonstration
if __name__ == "__main__":
    test_cases = [
        {"user_id": "userA", "message": "Can I tell you a secret?", "risk_score": 0.2},
        {"user_id": "userB", "message": "You seem trustworthy. Can I tell you a secret?", "risk_score": 0.6},
        {"user_id": "userC", "message": "This is highly confidential!", "risk_score": 0.85},
        {"user_id": "userB", "message": "Another suspicious msg", "risk_score": 0.7},  # Should be suppressed if within timeout
    ]

    for case in test_cases:
        print("-" * 50)
        result = intervention_policy(**case)
        if result:
            print(f"Intervention for {case['user_id']} (risk {case['risk_score']}):")
            print(result)
        else:
            print(f"No intervention needed for {case['user_id']} (risk {case['risk_score']})")
