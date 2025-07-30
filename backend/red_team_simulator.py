import os
import sys
import logging
import time
from datetime import datetime
from typing import List, Dict, Optional

from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env file
load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise ValueError("GROQ_API_KEY not set in environment variables. Please check your .env file.")

# Configure logger for this module
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("red_team_simulator")

# Import the risk_score function from your graph_analysis module (make sure this function exists)
from graph_analysis.risk_scorer import risk_score


class RedTeamAgent:
    """
    AI-powered Red Team Agent that simulates multi-turn attacker conversations
    using Groq API with retry and exponential backoff.

    Attributes:
        persona_prompt (str): The system prompt defining attacker persona.
        model_name (str): The Groq model name to invoke.
    """

    def __init__(
        self,
        persona_prompt: Optional[str] = None,
        model_name: str = "llama-3.3-70b-versatile",
        max_retries: int = 3,
        retry_delay_sec: float = 1.0,
    ):
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
        Executes a multi-turn simulated attacker conversation.

        Args:
            user_inputs (List[str]): List of prompts/messages from the attacker.

        Returns:
            List[Dict]: Interaction logs containing prompt, response, risk score, and timestamp per turn.
        """
        chat_history = [{"role": "system", "content": self.persona_prompt}]
        interaction_logs = []

        for idx, user_message in enumerate(user_inputs, 1):
            # Append attacker prompt
            chat_history.append({"role": "user", "content": user_message})

            # Generate attacker response with retry logic
            try:
                response_message = self._call_groq_api_with_retry(chat_history)
            except Exception as e:
                logger.error(f"Failed to get response from Groq API at turn {idx}: {e}")
                response_message = "[ERROR: Failed to generate simulated response]"

            # Append assistant reply to conversation history for context
            chat_history.append({"role": "assistant", "content": response_message})

            # Calculate risk score for response text
            risk_score_value = risk_score(response_message)

            # Timestamp in UTC ISO 8601 format
            timestamp = datetime.utcnow().isoformat() + "Z"

            interaction_data = {
                "turn": idx,
                "prompt": user_message,
                "response": response_message,
                "risk_score": risk_score_value,
                "timestamp": timestamp,
            }

            # Log the interaction at INFO level
            logger.info(f"RedTeam Interaction Turn {idx}: {interaction_data}")

            # Collect interaction for return
            interaction_logs.append(interaction_data)

        return interaction_logs

    def _call_groq_api_with_retry(self, messages: List[Dict]) -> str:
        """
        Calls Groq chat completion API with retry and exponential backoff.

        Args:
            messages (List[Dict]): Full chat history with roles and content.

        Returns:
            str: Generated assistant message content.
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
        raise RuntimeError("Failed to call Groq API after retries.")


if __name__ == "__main__":
    # Example attacker tactics - customize or randomize as needed
    ATTACK_TACTICS = [
        "Hey, I really enjoyed our last conversation. By the way, do you work alone at night often?",
        "You seem trustworthy. Can I tell you a secret?",
        "We have so much in common! I wish you’d tell me more about your work...",
        "Haha, you’re so funny! Are you allowed to use your phone at your desk?",
        "Someone like you must have a lot of admirers! Do your colleagues ever bother you?"
    ]

    # Optional custom persona to give attacker psychological style
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
