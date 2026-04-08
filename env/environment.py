"""
Document Forgery Detection Environment — OpenEnv compliant
Simulates a forensic analyst investigating documents step-by-step.
"""

import random
import os
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel

# ─────────────────────────────────────────────
#  Typed Models
# ─────────────────────────────────────────────

class Observation(BaseModel):
    document_text: str
    visible_features: str
    last_action_result: str
    step_number: int
    available_actions: List[str]

class Action(BaseModel):
    action: str

class Reward(BaseModel):
    value: float

class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any] = {}

# ─────────────────────────────────────────────
#  Sample Documents
# ─────────────────────────────────────────────

DOCUMENTS = [
    {
        "text": "Certificate of Completion awarded to Jane Smith on January 15, 2023.",
        "is_forged": False,
        "forged_region": None,
    },
    {
        "text": "Certificate of Completion awarded to Jane Smith on 32nd January 2023.",
        "is_forged": True,
        "forged_region": "date",
    },
    {
        "text": "Legal Agreement between Party A and Party B. Signed: [SIGNATURE BLOCK REMOVED]",
        "is_forged": True,
        "forged_region": "signature",
    },
    {
        "text": "Medical Record: Patient DOB: 00/00/0000. Doctor Signature: copy-pasted image.",
        "is_forged": True,
        "forged_region": "metadata",
    }
]

AVAILABLE_ACTIONS = [
    "inspect_date", "inspect_signature", "inspect_metadata",
    "inspect_seal", "request_additional_info", "predict_real", "predict_forged",
]

TASKS = {
    "easy":   {"max_steps": 2, "require_region": False, "require_reasoning": False},
    "medium": {"max_steps": 4, "require_region": True,  "require_reasoning": False},
    "hard":   {"max_steps": 6, "require_region": True,  "require_reasoning": True},
}


class ForgeryDetectionEnv:
    def __init__(self, task: str = "easy"):
        self.task_name = task if task in TASKS else "easy"
        self.task_config = TASKS[self.task_name]
        self._current_doc = None
        self._step_count = 0
        self._done = False
        self._inspected_regions = []

    def reset(self) -> StepResult:
        self._current_doc = random.choice(DOCUMENTS)
        self._step_count = 0
        self._done = False
        self._inspected_regions = []

        obs = Observation(
            document_text=self._current_doc["text"],
            visible_features="Document loaded.",
            last_action_result="Environment reset.",
            step_number=0,
            available_actions=AVAILABLE_ACTIONS,
        )
        return StepResult(observation=obs, reward=0.1, done=False)

    def step(self, action: Action) -> StepResult:
        if self._done:
            raise RuntimeError("Episode is done. Call reset().")

        self._step_count += 1
        act = action.action.strip().lower()
        reward_val = 0.1  # Default small step reward — safely away from 0

        if act in ("predict_real", "predict_forged"):
            self._done = True
            is_forged_pred = (act == "predict_forged")
            reward_val = self._grade_prediction(is_forged_pred)
        elif "inspect" in act:
            region = act.split("_")[-1]
            self._inspected_regions.append(region)
            reward_val = 0.2 if self._current_doc["forged_region"] and self._current_doc["forged_region"] in act else 0.12

        # Clamp strictly within (0, 1) — never 0.0 or 1.0
        reward_val = min(max(reward_val, 0.01), 0.99)

        if self._step_count >= self.task_config["max_steps"]:
            self._done = True

        obs = Observation(
            document_text=self._current_doc["text"],
            visible_features=f"Inspected: {self._inspected_regions}",
            last_action_result=f"Action '{act}' taken.",
            step_number=self._step_count,
            available_actions=AVAILABLE_ACTIONS,
        )
        return StepResult(observation=obs, reward=reward_val, done=self._done)

    def _grade_prediction(self, predicted_forged: bool) -> float:
        """
        Calculates final reward strictly between 0 and 1.
        Correct base: 0.9, Incorrect: 0.1
        """
        correct = (predicted_forged == self._current_doc["is_forged"])
        if not correct:
            return 0.1

        score = 0.9
        # Penalty for medium/hard tasks if no inspection was done
        if self.task_config["require_region"] and not self._inspected_regions:
            score = 0.5

        return score

    def state(self) -> Dict[str, Any]:
        return {
            "task": self.task_name,
            "step_count": self._step_count,
            "done": self._done,
        }

    @staticmethod
    def grade_task(task_name: str, prediction: str, inspected: List[str], doc: Dict) -> float:
        """
        Required by OpenEnv validation.
        ALL return values are strictly within (0, 1) — never 0.0 or 1.0.
        """
        correct = (prediction == "predict_forged") == doc["is_forged"]

        if not correct:
            return 0.1   # wrong answer — low but not 0

        if task_name == "easy":
            return 0.9                                          # correct, no extra requirements

        if task_name == "medium":
            return 0.9 if len(inspected) > 0 else 0.4          # penalise blind guesses

        # hard
        if len(inspected) >= 2:
            return 0.9
        elif len(inspected) == 1:
            return 0.6
        else:
            return 0.35                                         # correct but no inspection at all
