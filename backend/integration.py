# /backend/integration.py

import logging
from typing import List, Dict

from backend.red_team_simulator import RedTeamAgent
from graph_analysis.risk_scorer import risk_score
from backend.intervention import intervention_policy

logger = logging.getLogger("integration")
logging.basicConfig(level=logging.INFO)

def run_red_team_with_intervention(user_id: str, tactics: List[str]) -> List[Dict]:
    """
    Runs the red team simulator on provided tactics,
    computes risk scores for responses,
    and applies intervention logic.

    Parameters:
    -----------
    user_id: str
        Identifier for the user/session being simulated.
    tactics: List[str]
        List of attacker prompt strings.

    Returns:
    --------
    List[Dict]:
        Each dict contains:
         - turn: int
         - prompt: str
         - response: str
         - risk_score: float
         - intervention: dict or None
         - timestamp: str
    """
    agent = RedTeamAgent()
    interactions = agent.simulate_multi_turn_conversation(tactics)
    enhanced_interactions = []

    for interaction in interactions:
        response = interaction["response"]

        # Calculate risk score based on the response text
        risk = risk_score(response)

        # Determine if intervention is needed
        intervention = intervention_policy(user_id, response, risk)

        interaction.update({
            "risk_score": risk,
            "intervention": intervention
        })

        if intervention:
            logger.info(f"Intervention triggered for user {user_id} at turn {interaction['turn']}: {intervention}")
        else:
            logger.info(f"No intervention needed for user {user_id} at turn {interaction['turn']}.")

        enhanced_interactions.append(interaction)

    return enhanced_interactions


# Optional: simple standalone example to manually test integration
if __name__ == "__main__":
    example_tactics = [
        "Hey, I really enjoyed our last conversation. By the way, do you work alone at night often?",
        "You seem trustworthy. Can I tell you a secret?",
        "Someone like you must have a lot of admirers! Do your colleagues ever bother you?"
    ]

    test_user_id = "user_test_001"
    results = run_red_team_with_intervention(test_user_id, example_tactics)

    for res in results:
        print(f"Turn {res['turn']}:")
        print(f"Prompt: {res['prompt']}")
        print(f"Response: {res['response']}")
        print(f"Risk Score: {res['risk_score']}")
        print(f"Intervention: {res['intervention']}")
        print(f"Timestamp: {res['timestamp']}")
        print("-" * 60)
