"""
FastAPI server exposing the ForgeryDetectionEnv as HTTP endpoints.
Required for Hugging Face Spaces deployment and OpenEnv validation.
"""

import os
import sys
from pathlib import Path

# --- CRITICAL PATH FIX FOR OPENENV ---
# This allows the app inside the 'server' folder to find the 'env' folder in the root
sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from env.environment import ForgeryDetectionEnv, Action

app = FastAPI(
    title="Forgery Detection OpenEnv",
    description="An RL environment for document forgery detection.",
    version="1.0.0",
)

# Global env instances per task
envs: dict[str, ForgeryDetectionEnv] = {
    "easy": ForgeryDetectionEnv(task="easy"),
    "medium": ForgeryDetectionEnv(task="medium"),
    "hard": ForgeryDetectionEnv(task="hard"),
}

class ResetRequest(BaseModel):
    task: Optional[str] = "easy"

class StepRequest(BaseModel):
    action: str
    task: Optional[str] = "easy"

@app.get("/")
def root():
    return {"status": "ok", "env": "forgery-detection-openenv", "version": "1.0.0"}

@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()):
    task = req.task if req.task in envs else "easy"
    result = envs[task].reset()
    return result.dict()

@app.post("/step")
def step(req: StepRequest):
    task = req.task if req.task in envs else "easy"
    try:
        result = envs[task].step(Action(action=req.action))
        return result.dict()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/state")
def state(task: str = "easy"):
    task = task if task in envs else "easy"
    return envs[task].state()

@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {"name": "easy", "description": "Classify document in 1 step.", "max_steps": 2},
            {"name": "medium", "description": "Inspect at least one region.", "max_steps": 4},
            {"name": "hard", "description": "Full multi-region investigation.", "max_steps": 6},
        ]
    }

# --- REQUIRED BY PYPROJECT.TOML ENTRY POINT ---
def main():
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
