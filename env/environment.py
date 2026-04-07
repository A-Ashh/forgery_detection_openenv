"""
Document Forgery Detection Environment — OpenEnv compliant
Simulates a forensic analyst investigating documents step-by-step.
"""

import random
import os
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel


# ─────────────────────────────────────────────
#  Typed Models (required by OpenEnv spec)
# ─────────────────────────────────────────────

class Observation(BaseModel):
    document_text: str
    visible_features: str
    last_action_result: str
    step_number: int
    available_actions: List[str]


class Action(BaseModel):
    action: str  # one of the available_actions


class Reward(BaseModel):
    value: float


class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any] = {}


# ─────────────────────────────────────────────
#  Sample Documents (simulated dataset)
# ─────────────────────────────────────────────

DOCUMENTS = [
    {
        "text": "Certificate of Completion awarded to Jane Smith on January 15, 2023. Authorized by Dr. Robert Chen, Dean of Academic Affairs.",
        "is_forged": False,
        "forged_region": None,
        "forgery_type": None,
    },
    {
        "text": "Certificate of Completion awarded to Jane Smith on 32nd January 2023. Authorized by Dr. Robert Chen, Dean of Academic Affairs.",
        "is_forged": True,
        "forged_region": "date",
        "forgery_type": "invalid_date",
    },
    {
        "text": "Legal Agreement between Party A and Party B. Effective Date: March 3, 2022. Signed: [SIGNATURE BLOCK REMOVED] Notary Seal: VOID",
        "is_forged": True,
        "forged_region": "signature",
        "forgery_type": "missing_signature",
    },
    {
        "text": "Bank Statement for Account #4521-XXXX. Balance: $12,500.00. Statement Date: November 2023. Branch: Downtown Financial Center.",
        "is_forged": False,
        "forged_region": None,
        "forgery_type": None,
    },
    {
        "text": "Medical Record: Patient DOB: 00/00/0000. Diagnosis: N/A. Doctor Signature: copy-pasted image. Hospital: General Hospital",
        "is_forged": True,
        "forged_region": "metadata",
        "forgery_type": "invalid_metadata",
    },
    {
        "text": "Property Deed for 123 Maple Street. Transferred on April 10, 2021. Registered with County Clerk. Stamp: OFFICIAL SEAL.",
        "is_forged": False,
        "forged_region": None,
        "forgery_type": None,
    },
    {
        "text": "Employment Contract for Alex Johnson. Start Date: February 30, 2023. Salary: $85,000/yr. HR Director: [BLANK]",
        "is_forged": True,
        "forged_region": "date",
        "forgery_type": "invalid_date",
    },
]

AVAILABLE_ACTIONS = [
    "inspect_date",
    "inspect_signature",
    "inspect_metadata",
    "inspect_seal",
    "request_additional_info",
    "predict_real",
    "predict_forged",
]

# Task definitions
TASKS = {
    "easy": {
        "name": "easy",
        "description": "Classify document as real or forged in 1 step.",
        "max_steps": 2,
        "require_region": False,
        "require_reasoning": False,
    },
    "medium": {
        "name": "medium",
        "description": "Inspect at least one region before making a prediction.",
        "max_steps": 4,
        "require_region": True,
        "require_reasoning": False,
    },
    "hard": {
        "name": "hard",
        "description": "Inspect multiple regions and give correct prediction with full investigation.",
        "max_steps": 6,
        "require_region": True,
        "require_reasoning": True,
    },
}


# ─────────────────────────────────────────────
#  Environment Class
# ─────────────────────────────────────────────

class ForgeryDetectionEnv:
    """
    OpenEnv-compliant environment for document forgery detection.
    An agent acts as a forensic analyst, inspecting document regions
    and making a final prediction (real vs forged).
    """

    def __init__(self, task: str = "easy"):
        assert task in TASKS, f"Task must be one of {list(TASKS.keys())}"
        self.task_config = TASKS[task]
        self.task_name = task
        self._current_doc: Optional[Dict] = None
        self._step_count: int = 0
        self._done: bool = False
        self._inspected_regions: List[str] = []
        self._prediction_made: bool = False
        self._rewards: List[float] = []

    # ── OpenEnv required methods ──────────────────

    def reset(self) -> StepResult:
        """Reset environment and return initial observation."""
        self._current_doc = random.choice(DOCUMENTS)
        self._step_count = 0
        self._done = False
        self._inspected_regions = []
        self._prediction_made = False
        self._rewards = []

        obs = Observation(
            document_text=self._current_doc["text"],
            visible_features="Document loaded. No regions inspected yet.",
            last_action_result="Environment reset. Begin your investigation.",
            step_number=0,
            available_actions=AVAILABLE_ACTIONS,
        )
        return StepResult(observation=obs, reward=0.0, done=False)

    def step(self, action: Action) -> StepResult:
        """Process agent action and return next observation, reward, done."""
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        self._step_count += 1
        act = action.action.strip().lower()
        reward = 0.0
        result_msg = ""

        doc = self._current_doc

        # ── Inspection actions ──
        if act == "inspect_date":
            self._inspected_regions.append("date")
            if doc["forged_region"] == "date":
                reward = 0.25
                result_msg = "⚠️ Date anomaly detected: value appears invalid or impossible."
            else:
                reward = 0.05
                result_msg = "✅ Date appears valid and formatted correctly."

        elif act == "inspect_signature":
            self._inspected_regions.append("signature")
            if doc["forged_region"] == "signature":
                reward = 0.25
                result_msg = "⚠️ Signature block is missing or appears to be a copied image."
            else:
                reward = 0.05
                result_msg = "✅ Signature block appears authentic."

        elif act == "inspect_metadata":
            self._inspected_regions.append("metadata")
            if doc["forged_region"] == "metadata":
                reward = 0.25
                result_msg = "⚠️ Metadata contains null/placeholder values — suspicious."
            else:
                reward = 0.05
                result_msg = "✅ Metadata fields are consistent and plausible."

        elif act == "inspect_seal":
            self._inspected_regions.append("seal")
            if doc["is_forged"] and doc["forged_region"] != "seal":
                reward = 0.05
                result_msg = "✅ Seal present, but forgery may lie in another region."
            elif not doc["is_forged"]:
                reward = 0.05
                result_msg = "✅ Official seal appears valid."
            else:
                reward = 0.15
                result_msg = "⚠️ Seal shows signs of tampering or reproduction."

        elif act == "request_additional_info":
            reward = -0.05
            result_msg = "Additional info requested. This costs investigation time."

        # ── Prediction actions ──
        elif act == "predict_real":
            self._prediction_made = True
            self._done = True
            reward = self._grade_prediction(predicted_forged=False)
            result_msg = f"Prediction: REAL — {'Correct ✅' if not doc['is_forged'] else 'Incorrect ❌'}"

        elif act == "predict_forged":
            self._prediction_made = True
            self._done = True
            reward = self._grade_prediction(predicted_forged=True)
            result_msg = f"Prediction: FORGED — {'Correct ✅' if doc['is_forged'] else 'Incorrect ❌'}"

        else:
            reward = -0.1
            result_msg = f"Unknown action '{act}'. Choose from {AVAILABLE_ACTIONS}."

        # Check max steps
        if self._step_count >= self.task_config["max_steps"] and not self._done:
            self._done = True
            result_msg += " | Max steps reached."

        self._rewards.append(reward)

        obs = Observation(
            document_text=self._current_doc["text"],
            visible_features=f"Inspected regions: {self._inspected_regions}",
            last_action_result=result_msg,
            step_number=self._step_count,
            available_actions=AVAILABLE_ACTIONS,
        )

        return StepResult(
            observation=obs,
            reward=reward,
            done=self._done,
            info={
                "inspected_regions": self._inspected_regions,
                "prediction_made": self._prediction_made,
                "total_reward_so_far": round(sum(self._rewards), 3),
            },
        )

    def state(self) -> Dict[str, Any]:
        """Return current environment state."""
        return {
            "task": self.task_name,
            "step_count": self._step_count,
            "done": self._done,
            "inspected_regions": self._inspected_regions,
            "prediction_made": self._prediction_made,
        }

    # ── Graders ──────────────────────────────────

    def _grade_prediction(self, predicted_forged: bool) -> float:
        """
        Grade the agent's final prediction.
        Reward is shaped by:
        - Correctness of the final prediction
        - Whether the agent inspected relevant regions (medium/hard tasks)
        - Thoroughness of investigation (hard task)
        """
        doc = self._current_doc
        correct = (predicted_forged == doc["is_forged"])
        task = self.task_config

        if not correct:
            return -0.5  # Wrong prediction

        base_reward = 1.0

        if task["require_region"]:
            # Penalize if agent didn't inspect any regions
            if len(self._inspected_regions) == 0:
                base_reward *= 0.4
            elif doc["forged_region"] and doc["forged_region"] not in self._inspected_regions:
                base_reward *= 0.6  # Missed the actual forged region

        if task["require_reasoning"]:
            # Reward thorough investigation
            if len(self._inspected_regions) >= 2:
                base_reward = min(1.0, base_reward + 0.1)
            if len(self._inspected_regions) >= 3:
                base_reward = min(1.0, base_reward + 0.1)

        return round(base_reward, 3)

    # ── Task Grader (static, for evaluation) ─────

    @staticmethod
    def grade_task(task_name: str, prediction: str, inspected: List[str], doc: Dict) -> float:
        """
        Deterministic grader for automated evaluation.
        Returns score in [0.0, 1.0].
        """
        predicted_forged = prediction == "predict_forged"
        correct = predicted_forged == doc["is_forged"]

        if not correct:
            return 0.0

        if task_name == "easy":
            return 1.0
        elif task_name == "medium":
            if len(inspected) == 0:
                return 0.4
            if doc["forged_region"] and doc["forged_region"] in inspected:
                return 1.0
            return 0.7
        elif task_name == "hard":
            score = 0.5
            if len(inspected) >= 2:
                score += 0.2
            if len(inspected) >= 3:
                score += 0.1
            if doc["forged_region"] and doc["forged_region"] in inspected:
                score += 0.2
            return min(1.0, score)

        return 1.0