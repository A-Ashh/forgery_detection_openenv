import os
import sys
from typing import List
from openai import OpenAI
from env.environment import ForgeryDetectionEnv, Action

# ─── Configuration (Judges inject these) ──────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
API_KEY      = os.environ.get("API_KEY", "dummy-key")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

MAX_STEPS    = 6
TASKS        = ["easy", "medium", "hard"]
BENCHMARK    = "forgery-detection-openenv"

# ─── Logging helpers (STRICT FORMAT REQUIRED) ──────────────────────────────────

def log_start(task: str, env_name: str, model: str) -> None:
    print(f"[START] task={task} env={env_name} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error) -> None:
    err_str = str(error) if error else "none"
    print(f"[STEP] step={step} action={action} reward={reward:.3f} done={str(done).lower()} error={err_str}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# ─── LLM Agent Logic ─────────────────────────────────────────────────────────

def get_llm_action(observation, available_actions):
    prompt = f"""
    You are a forensic document analyst. 
    Observation: {observation}
    Available Actions: {available_actions}
    
    Respond ONLY with the name of the action to take.
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        action = response.choices[0].message.content.strip().lower()
        # Clean up common LLM extra formatting
        action = action.replace("action:", "").replace("`", "").strip()
        return action
    except Exception as e:
        return "predict_real" # Fallback

def run_task(task_name: str):
    env = ForgeryDetectionEnv(task=task_name)
    obs = env.reset().dict()
    log_start(task_name, BENCHMARK, MODEL_NAME)
    
    total_rewards = []
    done = False
    step_count = 0
    
    while not done and step_count < MAX_STEPS:
        step_count += 1
        # Extract actions from the observation list
        available = obs.get("available_actions", [])
        action_name = get_llm_action(obs, available)
        
        try:
            step_res = env.step(Action(action=action_name))
            reward = step_res.reward.value
            done = step_res.done
            obs = step_res.observation.dict()
            
            total_rewards.append(reward)
            log_step(step_count, action_name, reward, done, None)
        except Exception as e:
            log_step(step_count, action_name, 0.0, True, e)
            break
            
    score = sum(total_rewards) / len(total_rewards) if total_rewards else 0.0
    log_end(score > 0.5, step_count, score, total_rewards)
    return score

if __name__ == "__main__":
    scores = {}
    for task in TASKS:
        scores[task] = run_task(task)
    
    avg = sum(scores.values()) / len(scores)
    print(f"\n[SUMMARY] easy={scores['easy']:.3f} medium={scores['medium']:.3f} hard={scores['hard']:.3f} overall={avg:.3f}")
