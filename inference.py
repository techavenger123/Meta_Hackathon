import os
import time
import requests
import json
from collections import deque
from openai import OpenAI

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")
ENV_URL      = os.environ.get("ENV_URL",      "http://localhost:7861")
LOCAL_MODEL_PATH = os.environ.get(
    "LOCAL_MODEL_PATH",
    "/home/robotics-mu/.unsloth/studio/exports/unsloth_Llama-3.2-3B-Instruct-bnb-4bit_1775629280"
)

MAX_STEPS = 100

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
    Much better than Manhattan when obstacles split direct paths.
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
# HEURISTIC — pure BFS, provably never loops
# ──────────────────────────────────────────────────────────

def heuristic_action(obs, _stuck_counter=None) -> str:
    if _stuck_counter is None:
        _stuck_counter = [0]
    if not obs["garbage_positions"]:
        return "UP"

    r_pos          = list(obs["robot_position"])
    obstacles      = [list(o) for o in obs["obstacle_positions"]]
    grid_w, grid_h = obs["grid_size"]
    garbage        = [tuple(g) for g in obs["garbage_positions"]]

    if tuple(r_pos) in garbage:
        _stuck_counter[0] = 0
        return "COLLECT"

    ordered = nearest_neighbour_order(r_pos, garbage, obstacles, grid_w, grid_h)

    # If stuck for too many steps, try the SECOND-nearest target instead
    # (breaks out of dead-end loops on task_hard)
    if _stuck_counter[0] >= 4 and len(ordered) > 1:
        ordered = [ordered[1], ordered[0]] + ordered[2:]
    if _stuck_counter[0] >= 8:
        # Rotate the whole list one more step for a full direction change
        ordered = ordered[1:] + ordered[:1]
        _stuck_counter[0] = 0

    target = ordered[0]

    if tuple(target) == tuple(r_pos):
        _stuck_counter[0] = 0
        return "COLLECT"

    move, _ = bfs(r_pos, target, obstacles, grid_w, grid_h)
    if move and move != "COLLECT":
        _stuck_counter[0] = 0
        return move

    # BFS says this target is unreachable — try alternates
    for alt in ordered[1:]:
        move, _ = bfs(r_pos, alt, obstacles, grid_w, grid_h)
        if move and move != "COLLECT":
            _stuck_counter[0] = 0
            return move

    # Completely surrounded — take any open neighbour to escape
    _stuck_counter[0] += 1
    obstacle_set = frozenset(tuple(o) for o in obstacles)
    for name, (dx, dy) in [("RIGHT",(1,0)), ("LEFT",(-1,0)), ("UP",(0,1)), ("DOWN",(0,-1))]:
        npos = (r_pos[0]+dx, r_pos[1]+dy)
        if (0 <= npos[0] < grid_w and 0 <= npos[1] < grid_h
                and npos not in obstacle_set):
            return name

    return "RIGHT"


# ──────────────────────────────────────────────────────────
# ACTION RESOLVER  (priority: Q-table → LLM → BFS heuristic)
# ──────────────────────────────────────────────────────────

def resolve_next_action(client, obs, context_history, stuck_counter=None) -> str:
    heuristic = heuristic_action(obs, stuck_counter)

    # ── 1. Q-Learning policy (trained, deterministic) ──────────
    if _ql_agent is not None:
        q_action = _ql_agent.get_action(obs)
        if q_action is not None:
            return q_action

    system_prompt = (
        "You control a garbage collecting robot on a grid.\n"
        "Reply with EXACTLY ONE of: UP  DOWN  LEFT  RIGHT  COLLECT\n\n"
        "Rules:\n"
        "- COLLECT only when your position exactly matches a garbage position.\n"
        "- Never move into an obstacle tile.\n"
        "- Move toward the nearest reachable garbage.\n"
        f"- Pathfinding suggests: {heuristic}  (only override if clearly wrong)"
    )

    # ── 2. Try local fine-tuned merged model (Alpaca prompt format) ─────
    if _local_model is not None and _local_tokenizer is not None:
        try:
            # Use the same Alpaca format the model was trained on (code2.py / fixed_dataset.jsonl)
            alpaca_instruction = (
                "You are an AI brain controlling a garbage collecting robot.\n"
                "Reply with EXACTLY ONE of: UP DOWN LEFT RIGHT COLLECT"
            )
            prompt = (
                f"### Instruction:\n{alpaca_instruction}\n\n"
                f"### Input:\nENVIRONMENT STATUS:\n{obs['message']}\n\n"
                f"### Response:\n"
            )
            inputs = _local_tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=512
            ).to(_local_model.device)
            with __import__('torch').no_grad():
                outputs = _local_model.generate(
                    **inputs, max_new_tokens=6, do_sample=False,
                    pad_token_id=_local_tokenizer.eos_token_id
                )
            # Decode only the newly generated tokens
            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            token = _local_tokenizer.decode(new_tokens, skip_special_tokens=True).strip().upper()
            for valid in ["UP", "DOWN", "LEFT", "RIGHT", "COLLECT"]:
                if valid in token:
                    print(f"[LOCAL LLM] {token.split()[0] if token else '?'} (raw: {token!r})")
                    return valid
        except Exception as e:
            print(f"[LOCAL LLM ERROR] {e}")

    # ── Try remote OpenAI-compatible endpoint ─────────────────
    if client is not None:
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    *context_history,
                    {"role": "user", "content": f"STATUS:\n{obs['message']}\n\nCommand?"}
                ],
                temperature=0.0,
                max_tokens=6
            )
            action = response.choices[0].message.content.strip().upper()
            for valid in ["UP", "DOWN", "LEFT", "RIGHT", "COLLECT"]:
                if valid in action:
                    return valid
        except Exception as e:
            print(f"[REMOTE LLM ERROR] {e}")

    # ── Final fallback: pure BFS heuristic ────────────────────
    return heuristic


# ──────────────────────────────────────────────────────────
# DYNAMIC GARBAGE PLACEMENT
# ──────────────────────────────────────────────────────────

def prompt_custom_garbage(grid_w, grid_h, obstacles):
    """
    Interactive CLI — lets you place garbage tiles before each episode.
    Supports: manual 'x,y' | 'random N' | 'done'
    """
    obstacle_set = set(tuple(o) for o in obstacles)
    print(f"\n  Grid: {grid_w} x {grid_h}   Obstacles: {sorted(obstacle_set)}")
    print("  Enter garbage positions:")
    print("    x,y       place at column x, row y  (e.g. '4,4')")
    print("    random N  place N random pieces      (e.g. 'random 5')")
    print("    done      start the episode\n")

    garbage = []
    while True:
        raw = input("  Garbage > ").strip().lower()

        if raw == "done":
            if not garbage:
                print("  Need at least one garbage tile.")
                continue
            break

        if raw.startswith("random"):
            import random
            parts = raw.split()
            n = int(parts[1]) if len(parts) > 1 else 3
            candidates = [(x, y) for x in range(grid_w) for y in range(grid_h)
                          if (x, y) not in obstacle_set]
            garbage = random.sample(candidates, min(n, len(candidates)))
            print(f"  Random garbage: {garbage}")
            break

        try:
            x, y = map(int, raw.split(","))
            if not (0 <= x < grid_w and 0 <= y < grid_h):
                print(f"  Out of bounds — valid: 0-{grid_w-1}, 0-{grid_h-1}")
                continue
            if (x, y) in obstacle_set:
                print(f"  ({x},{y}) is an obstacle.")
                continue
            if (x, y) in garbage:
                print(f"  ({x},{y}) already added.")
                continue
            garbage.append((x, y))
            print(f"  Added ({x},{y})  total: {garbage}")
        except ValueError:
            print("  Format: x,y  e.g. '3,4'")

    return garbage


def reset_with_custom_garbage(task_id, garbage_positions):
    """
    Posts to /reset_custom to inject custom garbage positions at runtime.
    Falls back to standard /reset if something goes wrong.
    """
    try:
        res = requests.post(f"{ENV_URL}/reset_custom", json={
            "task_id": task_id,
            "garbage_positions": [list(g) for g in garbage_positions]
        })
        res.raise_for_status()
        return res.json()["observation"]
    except Exception as e:
        print(f"[WARN] /reset_custom failed ({e}), falling back to /reset")
        res = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id})
        res.raise_for_status()
        return res.json()["observation"]


# ──────────────────────────────────────────────────────────
# EPISODE RUNNER
# ──────────────────────────────────────────────────────────

def print_log(log_dict):
    print(json.dumps(log_dict), flush=True)


def run_episode(client, task_id, obs):
    policy = "q-table" if (_ql_agent and _ql_agent.loaded) else \
             ("local-llm" if _local_model else ("remote-llm" if client else "bfs"))
    print_log({"type": "[START]", "task_id": task_id,
               "model": MODEL_NAME, "policy": policy, "max_steps": MAX_STEPS})

    total_reward    = 0.0
    done            = False
    context_history = []
    step_idx        = 0
    stuck_counter   = [0]  # per-episode counter — no cross-episode state leak

    for step_idx in range(1, MAX_STEPS + 1):
        action = resolve_next_action(client, obs, context_history, stuck_counter)

        try:
            res = requests.post(f"{ENV_URL}/step", json={"command": action})
            res.raise_for_status()
            step_data = res.json()
        except Exception as e:
            print(f"Step error: {e}")
            break

        obs          = step_data["observation"]
        reward       = step_data["reward"]
        done         = step_data["done"]
        total_reward += reward

        print_log({"type": "[STEP]", "step": step_idx, "action": action,
                   "reward": round(reward, 2), "total_reward": round(total_reward, 2),
                   "done": done})

        if done:
            break

        time.sleep(0.05)

    try:
        score = requests.get(f"{ENV_URL}/grade/{task_id}").json()["score"]
    except Exception:
        score = 0.0

    print_log({"type": "[END]", "task_id": task_id, "total_steps": step_idx,
               "final_reward": round(total_reward, 2), "score": score})
    return score


# ──────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────

def main():
    global _local_model, _local_tokenizer, _ql_agent

    print("=" * 55)
    print("  Garbage Collecting Robot — Inference")
    print("=" * 55)

    # ── 1. Load Q-Learning policy (fastest, no GPU needed) ────
    if QLearningAgent is not None:
        _ql_agent = QLearningAgent()
        if _ql_agent.loaded:
            print(f"\n  [INFO] Q-table loaded ({len(_ql_agent.qtable):,} states). "
                  "Q-learning is the primary policy.")
        else:
            print("\n  [WARN] No Q-table found (qtable.json). "
                  "Run: python qlearning.py --train")
            print("          Falling through to LLM / BFS.")
    else:
        print("\n  [WARN] qlearning.py not found — skipping Q-table.")

    # ── 2. Attempt to load the fine-tuned merged model ────────────
    # The Unsloth Studio export is a full merged model (not a LoRA adapter),
    # so we load it with standard HuggingFace transformers.
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        print(f"\n  [INFO] Loading fine-tuned model from:\n         {LOCAL_MODEL_PATH}")
        _local_tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
        _local_model = AutoModelForCausalLM.from_pretrained(
            LOCAL_MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        _local_model.eval()
        print("  [INFO] Fine-tuned model loaded — used when Q-table misses a state.")
    except Exception as e:
        print(f"  [WARN] Fine-tuned model unavailable ({e}).")
        print("          Falling back to remote API / BFS heuristic.")
        _local_model, _local_tokenizer = None, None

    print("\n  Mode:")
    print("    [1] Predefined garbage  (standard scenarios)")
    print("    [2] Dynamic  —  you place garbage each run")
    dynamic = input("  Select (1 or 2): ").strip() == "2"

    print("\n  Task:")
    print("    [1] task_easy    5x5   1 piece   no obstacles")
    print("    [2] task_medium  7x7   3 pieces  obstacles")
    print("    [3] task_hard    10x10 5 pieces  maze")
    print("    [4] All tasks")
    choice = input("  Select (1/2/3/4): ").strip()

    task_map = {"1": ["task_easy"], "2": ["task_medium"],
                "3": ["task_hard"],  "4": ["task_easy", "task_medium", "task_hard"]}
    tasks = task_map.get(choice, ["task_easy"])

    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL) if HF_TOKEN else None
    if not client and _local_model is None:
        print("\n  [INFO] No HF_TOKEN and no local model — pure BFS heuristic mode.")
    elif not client:
        print("\n  [INFO] No HF_TOKEN — using local Unsloth model + BFS fallback.")

    for task_id in tasks:
        print(f"\n{'─'*40}\n  {task_id}\n{'─'*40}")

        try:
            res = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id})
            res.raise_for_status()
            base_obs = res.json()["observation"]
        except Exception as e:
            print(f"Reset failed: {e}")
            continue

        if dynamic:
            garbage = prompt_custom_garbage(
                base_obs["grid_size"][0],
                base_obs["grid_size"][1],
                base_obs["obstacle_positions"]
            )
            obs = reset_with_custom_garbage(task_id, garbage)
        else:
            obs = base_obs

        run_episode(client, task_id, obs)


if __name__ == "__main__":
    main()