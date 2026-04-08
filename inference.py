import os
import time
import requests
import json
from collections import deque
from openai import OpenAI
from typing import List, Optional

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")
ENV_URL      = os.environ.get("ENV_URL",      "http://localhost:7860")
HF_MODEL_ID  = os.environ.get("HF_MODEL_ID",  "TechAvenger/GarbageBot-Weights")
BENCHMARK    = os.environ.get("BENCHMARK",    "garbage_robot_v1")

MAX_STEPS = 200

# Model artifacts
_local_model     = None
_local_tokenizer = None
_ql_agent        = None

try:
    from qlearning import QLearningAgent
except ImportError:
    QLearningAgent = None

# ──────────────────────────────────────────────────────────
# LOGGING HELPERS (Mandatory Format)
# ──────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ──────────────────────────────────────────────────────────
# BFS & HEURISTIC
# ──────────────────────────────────────────────────────────

def bfs(start, goal, obstacles, grid_w, grid_h):
    start, goal = tuple(start), tuple(goal)
    if start == goal: return ("COLLECT", 0)
    obstacle_set = frozenset(tuple(o) for o in obstacles)
    dirs = [("RIGHT",(1,0)), ("LEFT",(-1,0)), ("UP",(0,1)), ("DOWN",(0,-1))]
    queue, visited = deque([(start, None, 0)]), {start}
    while queue:
        pos, first, depth = queue.popleft()
        for name, (dx, dy) in dirs:
            npos = (pos[0]+dx, pos[1]+dy)
            if not (0 <= npos[0] < grid_w and 0 <= npos[1] < grid_h): continue
            if npos in obstacle_set or npos in visited: continue
            move = first if first else name
            if npos == goal: return (move, depth + 1)
            visited.add(npos)
            queue.append((npos, move, depth + 1))
    return (None, float('inf'))

def nearest_neighbour_order(start, targets, obstacles, grid_w, grid_h):
    remaining, ordered, current = list(targets), [], tuple(start)
    while remaining:
        best = min(remaining, key=lambda t: bfs(current, t, obstacles, grid_w, grid_h)[1])
        ordered.append(best)
        remaining.remove(best)
        current = tuple(best)
    return ordered

def heuristic_action(obs, _stuck_counter=None) -> str:
    if _stuck_counter is None: _stuck_counter = [0]
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
    if _stuck_counter[0] >= 4 and len(ordered) > 1: ordered = [ordered[1], ordered[0]] + ordered[2:]
    target = ordered[0]
    move, _ = bfs(r_pos, target, obstacles, grid_w, grid_h)
    return move or "RIGHT"


# ──────────────────────────────────────────────────────────
# ACTION RESOLVER
# ──────────────────────────────────────────────────────────

def resolve_next_action(client, obs, stuck_counter=None) -> str:
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

def run_episode(client, task_id, obs):
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    rewards         = []
    steps_taken     = 0
    stuck_counter   = [0]
    score           = 0.0
    success         = False

    try:
        for step_idx in range(1, MAX_STEPS + 1):
            action = resolve_next_action(client, obs, stuck_counter)
            try:
                res = requests.post(f"{ENV_URL}/step", json={"command": action})
                res.raise_for_status()
                step_data = res.json()
            except Exception as e:
                log_step(step=step_idx, action=action, reward=0.0, done=True, error=str(e))
                break

            obs          = step_data["observation"]
            reward       = step_data["reward"]
            done         = step_data["done"]
            rewards.append(reward)
            steps_taken = step_idx

            log_step(step=step_idx, action=action, reward=reward, done=done, error=None)
            if done: break
            time.sleep(0.01)

        score = requests.get(f"{ENV_URL}/grade/{task_id}").json()["score"]
        success = score >= 0.5
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return score


# ──────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────

def main():
    global _local_model, _local_tokenizer, _ql_agent

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

    tasks = ["task_easy", "task_medium", "task_hard"] if args.task == "all" else [f"task_{args.task}" if not args.task.startswith("task_") else args.task]

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