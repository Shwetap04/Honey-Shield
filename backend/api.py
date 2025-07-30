from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import json
from pathlib import Path
from backend.integration import run_red_team_with_intervention

app = FastAPI(
    title="Honeytrap AI System API",
    description="API for red team simulation, risk scoring, behavioral interventions, and dashboard data.",
    version="1.0.0",
)

# --- Allow CORS for dashboard/frontend integration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your dashboard prod URL later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- SCHEMAS FOR RED TEAM SIMULATION ---
class RedTeamRequest(BaseModel):
    user_id: str = Field(..., description="Unique identifier for the user/session")
    tactics: List[str] = Field(
        ..., description="List of attacker prompt tactics/messages to simulate"
    )

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

# --- RED TEAM SIMULATION ENDPOINT ---
@app.post("/simulate_red_team", response_model=RedTeamResponse)
async def simulate_red_team(request: RedTeamRequest):
    """
    Run the red team AI simulation with given tactics, calculate risk scores,
    and generate interventions if necessary.
    Returns detailed interaction logs with intervention info.
    """
    try:
        results = run_red_team_with_intervention(request.user_id, request.tactics)
        for r in results:
            if r.get("intervention") is not None:
                r["intervention"] = InterventionOutput(**r["intervention"])
            else:
                r["intervention"] = None
        return {"results": [InteractionOutput(**r) for r in results]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during simulation: {e}")

# --- DASHBOARD DATA ENDPOINT ---
DASHBOARD_DATA_PATH = Path(__file__).parent.parent / "output" / "latest_results.json"

@app.get("/api/risk-results")
async def get_risk_results():
    """
    Get the latest risk analysis results for the dashboard (composite risk, user risks, flagged messages, etc.).
    """
    if DASHBOARD_DATA_PATH.exists():
        with open(DASHBOARD_DATA_PATH) as f:
            return json.load(f)
    return {"users": [], "flagged_messages": [], "num_nodes": 0, "num_edges": 0}

# --- ROOT ENDPOINT ---
@app.get("/")
def root():
    return {"message": "Welcome to the Honeytrap AI System API. Use /simulate_red_team for simulation and /api/risk-results for dashboard data."}
