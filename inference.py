import os
import time
import requests
import json
from collections import deque
from openai import OpenAI

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")
# FIX: Match the Dockerfile port (7860) to avoid connection refuse errors during evaluation
ENV_URL      = os.environ.get("ENV_URL",      "http://localhost:7860")
HF_MODEL_ID = os.environ.get(
    "HF_MODEL_ID",
    "TechAvenger/GarbageBot-Weights"
)

MAX_STEPS = 200   # raised to account for recharge/unload detours

# Lazy-loaded local model — populated in main() if Unsloth is available
_local_model     = None
_local_tokenizer = None

# Q-Learning agent — loaded once in main(), used as primary policy
_ql_agent = None
try:
    from qlearning import QLearningAgent
except ImportError:
    QLearningAgent = None


# ──────────────────────────────────────────────────────────
# BFS CORE
# ──────────────────────────────────────────────────────────

def bfs(start, goal, obstacles, grid_w, grid_h):
    """
    BFS from start to goal avoiding obstacles.
    Returns (first_direction, path_length) or (None, inf) if unreachable.
    """
    start, goal = tuple(start), tuple(goal)
    if start == goal:
        return ("COLLECT", 0)

    obstacle_set = frozenset(tuple(o) for o in obstacles)
    dirs = [("RIGHT",(1,0)), ("LEFT",(-1,0)), ("UP",(0,1)), ("DOWN",(0,-1))]
    queue   = deque([(start, None, 0)])
    visited = {start}

    while queue:
        pos, first, depth = queue.popleft()
        for name, (dx, dy) in dirs:
            npos = (pos[0]+dx, pos[1]+dy)
            if not (0 <= npos[0] < grid_w and 0 <= npos[1] < grid_h):
                continue
            if npos in obstacle_set or npos in visited:
                continue
            move = first if first else name
            if npos == goal:
                return (move, depth + 1)
            visited.add(npos)
            queue.append((npos, move, depth + 1))

    return (None, float('inf'))


def nearest_neighbour_order(start, targets, obstacles, grid_w, grid_h):
    """
    Orders garbage by nearest-neighbour TSP using actual BFS cost.
    """
    remaining = list(targets)
    ordered   = []
    current   = tuple(start)
    while remaining:
        best = min(remaining, key=lambda t: bfs(current, t, obstacles, grid_w, grid_h)[1])
        ordered.append(best)
        remaining.remove(best)
        current = tuple(best)
    return ordered


# ──────────────────────────────────────────────────────────
# HEURISTIC
# ──────────────────────────────────────────────────────────

def heuristic_action(obs, _stuck_counter=None) -> str:
    if _stuck_counter is None:
        _stuck_counter = [0]

    robot_mode     = obs.get("robot_mode", "normal")
    r_pos          = list(obs["robot_position"])
    obstacles      = [list(o) for o in obs["obstacle_positions"]]
    grid_w, grid_h = obs["grid_size"]

    if robot_mode == "recharging":
        home = obs.get("home_position", r_pos)
        move, _ = bfs(r_pos, home, obstacles, grid_w, grid_h)
        return move or "UP"

    if robot_mode == "unloading":
        station = obs.get("unload_station", r_pos)
        move, _ = bfs(r_pos, station, obstacles, grid_w, grid_h)
        return move or "UP"

    garbage = [tuple(g) for g in obs["garbage_positions"]]
    if not garbage: return "UP"

    if tuple(r_pos) in garbage:
        _stuck_counter[0] = 0
        return "COLLECT"

    ordered = nearest_neighbour_order(r_pos, garbage, obstacles, grid_w, grid_h)
    if _stuck_counter[0] >= 4 and len(ordered) > 1:
        ordered = [ordered[1], ordered[0]] + ordered[2:]
    
    target = ordered[0]
    move, _ = bfs(r_pos, target, obstacles, grid_w, grid_h)
    if move and move != "COLLECT":
        return move

    return "RIGHT"


# ──────────────────────────────────────────────────────────
# ACTION RESOLVER
# ──────────────────────────────────────────────────────────

def resolve_next_action(client, obs, context_history, stuck_counter=None) -> str:
    heuristic = heuristic_action(obs, stuck_counter)
    if _ql_agent is not None:
        q_action = _ql_agent.get_action(obs)
        if q_action is not None: return q_action

    if _local_model is not None and _local_tokenizer is not None:
        try:
            prompt = f"### Instruction:\nAI control.\n\n### Input:\n{obs['message']}\n\n### Response:\n"
            inputs = _local_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(_local_model.device)
            with __import__('torch').no_grad():
                outputs = _local_model.generate(**inputs, max_new_tokens=6, do_sample=False, pad_token_id=_local_tokenizer.eos_token_id)
            token = _local_tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip().upper()
            for valid in ["UP", "DOWN", "LEFT", "RIGHT", "COLLECT"]:
                if valid in token: return valid
        except Exception: pass

    return heuristic


# ──────────────────────────────────────────────────────────
# EPISODE RUNNER
# ──────────────────────────────────────────────────────────

def print_log(msg):
    print(msg, flush=True)

def run_episode(client, task_id, obs):
    # Minimal START log for validator
    print_log(f"[START] task={task_id}")

    total_reward    = 0.0
    context_history = []
    step_idx        = 0
    stuck_counter   = [0]

    for step_idx in range(1, MAX_STEPS + 1):
        action = resolve_next_action(client, obs, context_history, stuck_counter)
        try:
            res = requests.post(f"{ENV_URL}/step", json={"command": action})
            res.raise_for_status()
            step_data = res.json()
        except: break

        obs          = step_data["observation"]
        reward       = step_data["reward"]
        done         = step_data["done"]
        total_reward += reward

        # Minimal STEP log for validator
        print_log(f"[STEP] step={step_idx} reward={round(reward, 2)} done={done}")
        if done: break
        time.sleep(0.01)

    try:
        score = requests.get(f"{ENV_URL}/grade/{task_id}").json()["score"]
    except: score = 0.0

    # Minimal END log for validator
    print_log(f"[END] task={task_id} score={score} steps={step_idx}")
    return score


# ──────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────

def main():
    global _local_model, _local_tokenizer, _ql_agent

    # Removed descriptive headers to keep stdout clean of anything but validation logs

    if QLearningAgent is not None:
        _ql_agent = QLearningAgent()
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        _local_tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
        has_cuda = torch.cuda.is_available()
        _local_model = AutoModelForCausalLM.from_pretrained(
            HF_MODEL_ID,
            torch_dtype=torch.float16 if has_cuda else torch.float32,
            device_map="auto" if has_cuda else None,
            load_in_4bit=has_cuda
        )
        _local_model.eval()
    except: pass

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="all")
    args = parser.parse_args()

    if args.task in ["1", "easy"]: tasks = ["task_easy"]
    elif args.task in ["2", "medium"]: tasks = ["task_medium"]
    elif args.task in ["3", "hard"]: tasks = ["task_hard"]
    else: tasks = ["task_easy", "task_medium", "task_hard"]

    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL) if HF_TOKEN else None

    for task_id in tasks:
        try:
            res = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id})
            res.raise_for_status()
            obs = res.json()["observation"]
            run_episode(client, task_id, obs)
        except: continue

if __name__ == "__main__":
    main()