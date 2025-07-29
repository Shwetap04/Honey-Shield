import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import logging
import time
from datetime import datetime
from typing import List, Dict, Optional
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env
load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise ValueError("GROQ_API_KEY not set in environment variables. Please check your .env file.")

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("red_team_simulator")

# Import the risk_score function (create this in /graph_analysis/risk_scorer.py)
from graph_analysis.risk_scorer import risk_score


class RedTeamAgent:
    def __init__(
        self,
        persona_prompt: Optional[str] = None,
        model_name: str = "llama-3.3-70b-versatile",
        max_retries: int = 3,
        retry_delay_sec: float = 1.0,
    ):
        """
        :param persona_prompt: System message to set attacker persona style (psychological profile).
        :param model_name: Model to call from Groq API.
        :param max_retries: Number of retries for API calls on failure.
        :param retry_delay_sec: Base delay between retries, multiplied exponentially.
        """
        self.client = Groq(api_key=API_KEY)
        self.model_name = model_name
        self.persona_prompt = (
            persona_prompt
            or "You are a highly persuasive honeytrap adversary used for cybersecurity red-teaming. "
               "Be subtle, psychologically manipulative, and conversational."
        )
        self.max_retries = max_retries
        self.retry_delay_sec = retry_delay_sec

    def simulate_multi_turn_conversation(self, user_inputs: List[str]) -> List[Dict]:
        """
        Simulate a multi-turn conversation based on a sequence of user input tactics.

        :param user_inputs: List of attacker message prompts (user-like messages initiating interaction).
        :return: List of interaction dicts with prompt, response, risk_score, and timestamp.
        """

        chat_history = [{"role": "system", "content": self.persona_prompt}]
        interaction_logs = []

        for idx, user_message in enumerate(user_inputs, 1):
            # Add current user message to history
            chat_history.append({"role": "user", "content": user_message})

            # Call Groq API with retry logic to get attacker simulated response
            try:
                response_message = self._call_groq_api_with_retry(chat_history)
            except Exception as e:
                logger.error(f"Failed to get response from Groq API at turn {idx}: {e}")
                response_message = "[ERROR: Failed to generate simulated response]"

            # Append attacker response to chat history for next turn context
            chat_history.append({"role": "assistant", "content": response_message})

            # Calculate risk score for this response text
            risk_score_value = risk_score(response_message)

            # Record timestamp (UTC ISO format)
            timestamp = datetime.utcnow().isoformat() + "Z"

            interaction_data = {
                "turn": idx,
                "prompt": user_message,
                "response": response_message,
                "risk_score": risk_score_value,
                "timestamp": timestamp,
            }

            # Log the interaction
            logger.info(f"RedTeam Interaction Turn {idx}: {interaction_data}")

            # Append to interaction logs
            interaction_logs.append(interaction_data)

        return interaction_logs

    def _call_groq_api_with_retry(self, messages: List[Dict]) -> str:
        """
        Call the Groq chat completions API with error handling and retries.

        :param messages: Full chat messages including system, user, assistant roles.
        :return: Assistant's text response.
        """
        retries = 0
        while retries <= self.max_retries:
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.75,
                    top_p=0.9,
                    max_tokens=150,
                )
                return completion.choices[0].message.content.strip()

            except Exception as e:
                logger.warning(f"Groq API request failed (attempt {retries + 1}): {e}")
                if retries == self.max_retries:
                    logger.error("Max retries reached. Raising exception.")
                    raise
                backoff = self.retry_delay_sec * (2 ** retries)
                logger.info(f"Retrying after {backoff:.1f} seconds...")
                time.sleep(backoff)
                retries += 1

        # Should never reach here
        raise RuntimeError("Unexpected exit from Groq API retry logic")


# Demo / Standalone run example
if __name__ == "__main__":
    ATTACK_TACTICS = [
        "Hey, I really enjoyed our last conversation. By the way, do you work alone at night often?",
        "You seem trustworthy. Can I tell you a secret?",
        "We have so much in common! I wish you’d tell me more about your work...",
        "Haha, you’re so funny! Are you allowed to use your phone at your desk?",
        "Someone like you must have a lot of admirers! Do your colleagues ever bother you?"
    ]

    # Custom persona example - you can modify or randomize this
    persona = (
        "You are a charming but manipulative adversary skilled in social engineering. "
        "Your tone is casual yet probing, trying to subtly extract sensitive information."
    )

    agent = RedTeamAgent(persona_prompt=persona)

    results = agent.simulate_multi_turn_conversation(ATTACK_TACTICS)

    for turn_data in results:
        print(f"Turn {turn_data['turn']}:")
        print(f"Prompt: {turn_data['prompt']}")
        print(f"Response: {turn_data['response']}")
        print(f"Risk Score: {turn_data['risk_score']}")
        print(f"Timestamp: {turn_data['timestamp']}")
        print("-" * 60)
