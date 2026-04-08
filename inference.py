import os
import sys
from typing import List
from openai import OpenAI
from env.environment import ForgeryDetectionEnv, Action

# ─── Configuration (Judges inject these) ──────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
API_KEY      = os.environ.get("HF_TOKEN", os.environ.get("API_KEY", "dummy-key"))
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

MAX_STEPS = 6
TASKS     = ["easy", "medium", "hard"]
BENCHMARK = "forgery-detection-openenv"

# ─── Logging helpers (STRICT FORMAT REQUIRED) ─────────────────────────────────
def log_start(task: str, env_name: str, model: str) -> None:
    print(f"[START] task={task} env={env_name} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error) -> None:
    err_str = str(error) if error else "none"
    print(f"[STEP] step={step} action={action} reward={reward:.3f} done={str(done).lower()} error={err_str}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# ─── LLM Agent Logic ──────────────────────────────────────────────────────────
def get_llm_action(observation, available_actions):
    prompt = f"""
You are a forensic document analyst.
Observation: {observation}
Available Actions: {available_actions}

Respond ONLY with the exact name of one action from the Available Actions list. No explanation.
"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        action = response.choices[0].message.content.strip().lower()
        action = action.replace("action:", "").replace("`", "").strip()
        # Validate it's a known action, fallback if not
        if action not in available_actions:
            action = "predict_real"
        return action
    except Exception:
        return "predict_real"  # Safe fallback

# ─── Task Runner ──────────────────────────────────────────────────────────────
def run_task(task_name: str) -> float:
    env = ForgeryDetectionEnv(task=task_name)
    reset_result = env.reset()
    obs = reset_result.observation.dict()

    log_start(task_name, BENCHMARK, MODEL_NAME)

    total_rewards = []
    done = False
    step_count = 0

    while not done and step_count < MAX_STEPS:
        step_count += 1
        available = obs.get("available_actions", [])
        action_name = get_llm_action(obs, available)

        try:
            step_res = env.step(Action(action=action_name))

            # ✅ FIX: reward is a float directly on StepResult, not a Reward object
            reward = float(step_res.reward)
            done   = step_res.done
            obs    = step_res.observation.dict()

            # Clamp strictly within (0, 1) — never 0.0 or 1.0
            reward = min(max(reward, 0.01), 0.99)

            total_rewards.append(reward)
            log_step(step_count, action_name, reward, done, None)

        except Exception as e:
            # ✅ FIX: fallback reward is 0.1, not 0.0 (0.0 is out of range)
            fallback_reward = 0.1
            total_rewards.append(fallback_reward)
            log_step(step_count, action_name, fallback_reward, True, e)
            break

    # ✅ FIX: clamp final score strictly within (0, 1)
    if total_rewards:
        score = sum(total_rewards) / len(total_rewards)
    else:
        score = 0.1  # safe default instead of 0.0

    score = min(max(score, 0.01), 0.99)

    log_end(score > 0.5, step_count, score, total_rewards)
    return score

# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    scores = {}
    for task in TASKS:
        scores[task] = run_task(task)

    avg = sum(scores.values()) / len(scores)
    avg = min(max(avg, 0.01), 0.99)
    print(f"\n[SUMMARY] easy={scores['easy']:.3f} medium={scores['medium']:.3f} hard={scores['hard']:.3f} overall={avg:.3f}")
