from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from backend.integration import run_red_team_with_intervention

app = FastAPI(
    title="Honeytrap AI System API",
    description="API for red team simulation, risk scoring, and behavioral interventions",
    version="1.0.0",
)

# Request body schema
class RedTeamRequest(BaseModel):
    user_id: str = Field(..., description="Unique identifier for the user/session")
    tactics: List[str] = Field(
        ..., 
        description="List of attacker prompt tactics/messages to simulate"
    )

# Response subset schemas (optional)
class InterventionOutput(BaseModel):
    action: str
    user_id: str
    message: str
    risk_score: float
    original_message: str
    timestamp: str
    context: Optional[Dict[str, Any]] = None

class InteractionOutput(BaseModel):
    turn: int
    prompt: str
    response: str
    risk_score: float
    intervention: Optional[InterventionOutput] = None
    timestamp: str

class RedTeamResponse(BaseModel):
    results: List[InteractionOutput]


@app.post("/simulate_red_team", response_model=RedTeamResponse)
async def simulate_red_team(request: RedTeamRequest):
    """
    Run the red team AI simulation with given tactics, calculate risk scores,
    and generate interventions if necessary.

    Returns detailed interaction logs with intervention info.
    """
    try:
        # Run your integrated red team simulation + risk scoring + intervention pipeline
        results = run_red_team_with_intervention(request.user_id, request.tactics)

        # Pydantic model expects InterventionOutput or None for intervention field,
        # so convert dict to model or None accordingly
        for r in results:
            if r.get("intervention") is not None:
                r["intervention"] = InterventionOutput(**r["intervention"])
            else:
                r["intervention"] = None

        return {"results": [InteractionOutput(**r) for r in results]}

    except Exception as e:
        # Detailed error visible for debugging; customize in production
        raise HTTPException(status_code=500, detail=f"Error during simulation: {e}")


@app.get("/")
def root():
    return {"message": "Welcome to the Honeytrap AI System API. Use /simulate_red_team to run simulations."}
