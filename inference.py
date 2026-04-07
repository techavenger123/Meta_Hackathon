import os
import time
import requests
import json
from openai import OpenAI

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")
ENV_URL      = os.environ.get("ENV_URL",      "http://localhost:7860")

TASKS = ["task_easy", "task_medium", "task_hard"]
MAX_STEPS = 50

def print_log(log_dict):
    """Prints the required structured JSON logs strictly to stdout."""
    print(json.dumps(log_dict), flush=True)

def heuristic_action(obs, last_blocked: set) -> str:
    """
    Obstacle-aware heuristic. Tries to move toward the nearest garbage
    while avoiding known blocked directions from the current position.
    """
    if not obs["garbage_positions"]:
        return "UP"  # nothing to do

    r_pos = list(obs["robot_position"])
    obstacles = [list(o) for o in obs["obstacle_positions"]]
    grid_w, grid_h = obs["grid_size"]

    # Find nearest garbage by Manhattan distance
    target = min(obs["garbage_positions"], key=lambda g: abs(g[0]-r_pos[0]) + abs(g[1]-r_pos[1]))

    if list(target) == r_pos:
        return "COLLECT"

    # Build preference order toward the target
    dx = target[0] - r_pos[0]
    dy = target[1] - r_pos[1]

    candidates = []
    if dx > 0: candidates.append(("RIGHT", [r_pos[0]+1, r_pos[1]]))
    if dx < 0: candidates.append(("LEFT",  [r_pos[0]-1, r_pos[1]]))
    if dy > 0: candidates.append(("UP",    [r_pos[0],   r_pos[1]+1]))
    if dy < 0: candidates.append(("DOWN",  [r_pos[0],   r_pos[1]-1]))

    # Append perpendicular moves as fallback escape routes
    all_moves = [
        ("RIGHT", [r_pos[0]+1, r_pos[1]]),
        ("LEFT",  [r_pos[0]-1, r_pos[1]]),
        ("UP",    [r_pos[0],   r_pos[1]+1]),
        ("DOWN",  [r_pos[0],   r_pos[1]-1]),
    ]
    for m in all_moves:
        if m not in candidates:
            candidates.append(m)

    for direction, npos in candidates:
        # Skip if this direction was blocked on a previous step from this same cell
        if direction in last_blocked:
            continue
        # Skip if out of bounds
        if not (0 <= npos[0] < grid_w and 0 <= npos[1] < grid_h):
            continue
        # Skip if obstacle
        if npos in obstacles:
            continue
        return direction

    # Truly stuck — just pick the first non-blocked direction
    return "RIGHT"


def resolve_next_action(client, obs, context_history, last_blocked: set) -> str:
    """
    Asks the LLM for the next command based on sensor readings.
    """
    system_prompt = (
        "You are an AI brain controlling a simulated garbage collecting robot.\n"
        "Your goal is to navigate a grid room and collect all pieces of garbage before running out of battery.\n"
        "Your available actions are EXACTLY ONE of the following keywords:\n"
        "UP\nDOWN\nLEFT\nRIGHT\nCOLLECT\n\n"
        "If you are standing on top of garbage, execute COLLECT.\n"
        "Otherwise, move towards the garbage by matching your X/Y coordinates."
    )
    
    prompt = f"ENVIRONMENT STATUS:\n{obs['message']}\n\nWhat is your next command?"
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                *context_history,
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=6
        )
        action = response.choices[0].message.content.strip().upper()
        # Clean up output
        for valid in ["UP", "DOWN", "LEFT", "RIGHT", "COLLECT"]:
            if valid in action:
                return valid
        return "UP"  # Default fallback
    except Exception as e:
        print(f"[LLM ERROR] {e}")
        return heuristic_action(obs, last_blocked)

def main():
    print("=" * 60)
    print("Garbage Collecting Robot — Inference Script")
    print("=" * 60)
    print()

    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

    for task_id in TASKS:
        # 1. Reset Environment
        print(f"\n{'─' * 40}\nRunning task: {task_id}\n{'─' * 40}")
        try:
            res = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id})
            res.raise_for_status()
            obs = res.json()["observation"]
        except Exception as e:
            print(f"Failed to reset {task_id}: {e}")
            continue

        start_log = {
            "type": "[START]",
            "task_id": task_id,
            "env": "garbage-collecting-robot",
            "model": MODEL_NAME,
            "max_steps": MAX_STEPS
        }
        print_log(start_log)

        total_reward = 0.0
        done = False
        context_history = []
        last_blocked: set = set()   # directions blocked at the robot's CURRENT cell
        last_robot_pos = None

        # 2. Run Episode
        for step_idx in range(1, MAX_STEPS + 1):
            # Reset blocked directions whenever the robot successfully moves to a new cell
            current_pos = tuple(obs["robot_position"])
            if current_pos != last_robot_pos:
                last_blocked = set()
                last_robot_pos = current_pos

            action = resolve_next_action(client, obs, context_history, last_blocked)
            
            try:
                res = requests.post(f"{ENV_URL}/step", json={"command": action})
                res.raise_for_status()
                step_data = res.json()
            except Exception as e:
                print(f"Failed to execute step: {e}")
                break
                
            next_obs = step_data["observation"]
            reward = step_data["reward"]
            done = step_data["done"]
            total_reward += reward

            # If the robot didn't move (obstacle or wall), mark that direction as blocked
            if tuple(next_obs["robot_position"]) == tuple(obs["robot_position"]):
                if action in ["UP", "DOWN", "LEFT", "RIGHT"]:
                    last_blocked.add(action)

            # Update context history (optional memory for LLM)
            # context_history.append({"role": "user", "content": f"ENVIRONMENT STATUS:\n{obs['message']}\nWhat is your next command?"})
            # context_history.append({"role": "assistant", "content": action})

            obs = next_obs

            step_log = {
                "type": "[STEP]",
                "step": step_idx,
                "action": action,
                "reward": reward,
                "total_reward": total_reward,
                "done": done,
            }
            print_log(step_log)

            if done:
                break

            time.sleep(0.1)

        # 3. Grade & End Episode
        try:
            grade_res = requests.get(f"{ENV_URL}/grade/{task_id}")
            score = grade_res.json()["score"]
        except Exception:
            score = 0.0

        end_log = {
            "type": "[END]",
            "task_id": task_id,
            "total_steps": step_idx,
            "final_reward": total_reward,
            "score": score
        }
        print_log(end_log)

if __name__ == "__main__":
    main()