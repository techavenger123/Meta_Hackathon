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
# HEURISTIC — BFS-based, mode-aware
# ──────────────────────────────────────────────────────────

def heuristic_action(obs, _stuck_counter=None) -> str:
    """
    Pure-BFS heuristic that respects the robot's autonomous mode.

    When the environment reports robot_mode == 'recharging' or 'unloading',
    the action suggested here is overridden by the environment's own resolver
    anyway — but we still return a sensible direction so logs are readable.

    In normal mode the heuristic targets the nearest garbage via BFS with a
    nearest-neighbour tour order, plus a stuck-counter escape hatch.
    """
    if _stuck_counter is None:
        _stuck_counter = [0]

    robot_mode     = obs.get("robot_mode", "normal")
    r_pos          = list(obs["robot_position"])
    obstacles      = [list(o) for o in obs["obstacle_positions"]]
    grid_w, grid_h = obs["grid_size"]

    # ── Recharging: head to home ───────────────────────────────
    if robot_mode == "recharging":
        home = obs.get("home_position", r_pos)
        move, _ = bfs(r_pos, home, obstacles, grid_w, grid_h)
        return move or "UP"

    # ── Unloading: head to unload station ─────────────────────
    if robot_mode == "unloading":
        station = obs.get("unload_station", r_pos)
        move, _ = bfs(r_pos, station, obstacles, grid_w, grid_h)
        return move or "UP"

    # ── Normal: collect nearest garbage ───────────────────────
    garbage = [tuple(g) for g in obs["garbage_positions"]]
    if not garbage:
        return "UP"   # nothing to do; env will mark episode done

    if tuple(r_pos) in garbage:
        _stuck_counter[0] = 0
        return "COLLECT"

    ordered = nearest_neighbour_order(r_pos, garbage, obstacles, grid_w, grid_h)

    # Stuck-counter escape: try alternate targets after repeated no-progress steps
    if _stuck_counter[0] >= 4 and len(ordered) > 1:
        ordered = [ordered[1], ordered[0]] + ordered[2:]
    if _stuck_counter[0] >= 8:
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

    # Primary target unreachable — try alternates
    for alt in ordered[1:]:
        move, _ = bfs(r_pos, alt, obstacles, grid_w, grid_h)
        if move and move != "COLLECT":
            _stuck_counter[0] = 0
            return move

    # Fully boxed in: take any open neighbouring cell to escape
    _stuck_counter[0] += 1
    obstacle_set = frozenset(tuple(o) for o in obstacles)
    for name, (dx, dy) in [("RIGHT",(1,0)),("LEFT",(-1,0)),("UP",(0,1)),("DOWN",(0,-1))]:
        npos = (r_pos[0]+dx, r_pos[1]+dy)
        if (0 <= npos[0] < grid_w and 0 <= npos[1] < grid_h
                and npos not in obstacle_set):
            return name

    return "RIGHT"


# ──────────────────────────────────────────────────────────
# ACTION RESOLVER  (priority: Q-table → LLM → BFS heuristic)
# ──────────────────────────────────────────────────────────

def resolve_next_action(client, obs, context_history, stuck_counter=None) -> str:
    """
    Decide the next action using the priority chain:
      1. Q-table (trained, deterministic, fastest)
      2. Fine-tuned local LLM (Unsloth export)
      3. Remote OpenAI-compatible endpoint
      4. BFS heuristic (fallback, always works)

    The BFS heuristic is mode-aware and is passed as a hint to the LLM.
    Note: when the environment is in MODE_RECHARGE or MODE_UNLOAD it will
    override whatever action we return, so correctness in those modes is
    the heuristic's responsibility, not the LLM's.
    """
    heuristic = heuristic_action(obs, stuck_counter)

    # ── 1. Q-Learning policy (trained, deterministic) ──────────
    if _ql_agent is not None:
        q_action = _ql_agent.get_action(obs)
        if q_action is not None:
            return q_action

    # Build a mode-aware system prompt for the LLM
    robot_mode   = obs.get("robot_mode", "normal")
    dist_home    = obs.get("distance_from_home", -1)
    storage_load = obs.get("current_storage_load", 0)
    capacity     = obs.get("storage_capacity", 6)
    home         = obs.get("home_position", (0, 0))
    station      = obs.get("unload_station", (0, 0))

    mode_note = ""
    if robot_mode == "recharging":
        mode_note = (
            f"\n⚠ ROBOT MODE: RECHARGING — navigate to home {home} "
            f"({dist_home} steps away). Do NOT collect garbage until recharged."
        )
    elif robot_mode == "unloading":
        mode_note = (
            f"\n⚠ ROBOT MODE: UNLOADING — navigate to unload station {station}. "
            f"Storage is full ({storage_load}/{capacity}). "
            f"Do NOT collect garbage until unloaded."
        )
    else:
        mode_note = (
            f"\nBattery distance to home: {dist_home} steps. "
            f"Storage: {storage_load}/{capacity}."
        )

    system_prompt = (
        "You control a garbage collecting robot on a grid.\n"
        "Reply with EXACTLY ONE of: UP  DOWN  LEFT  RIGHT  COLLECT\n\n"
        "Rules:\n"
        "- COLLECT only when your position exactly matches a garbage position.\n"
        "- Never move into an obstacle tile.\n"
        "- The environment handles recharging and unloading automatically.\n"
        f"- Pathfinding suggests: {heuristic}  (only override if clearly wrong)"
        f"{mode_note}"
    )

    # ── 2. Try local fine-tuned merged model (Alpaca prompt format) ─────
    if _local_model is not None and _local_tokenizer is not None:
        try:
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
            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            token = _local_tokenizer.decode(new_tokens, skip_special_tokens=True).strip().upper()
            for valid in ["UP", "DOWN", "LEFT", "RIGHT", "COLLECT"]:
                if valid in token:
                    print(f"[LOCAL LLM] {token.split()[0] if token else '?'} (raw: {token!r})")
                    return valid
        except Exception as e:
            print(f"[LOCAL LLM ERROR] {e}")

    # ── 3. Try remote OpenAI-compatible endpoint ─────────────────
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

    # ── 4. Final fallback: pure BFS heuristic ─────────────────
    return heuristic


# ──────────────────────────────────────────────────────────
# INTERACTIVE GARBAGE PLACEMENT
# ──────────────────────────────────────────────────────────

def prompt_custom_garbage(grid_w, grid_h, obstacles):
    """
    Interactive CLI helper: prompts the user to enter garbage positions
    for a dynamic episode.
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

def print_log(msg):
    # Print strings directly for the validator, ensured to be flushed for parsing.
    print(msg, flush=True)


def run_episode(client, task_id, obs):
    policy = (
        "q-table"    if (_ql_agent and _ql_agent.loaded)  else
        "local-llm"  if _local_model                      else
        "remote-llm" if client                            else
        "bfs"
    )
    print_log(f"\n[START] task={task_id} model={MODEL_NAME} policy={policy}")

    total_reward    = 0.0
    done            = False
    context_history = []
    step_idx        = 0
    stuck_counter   = [0]   # per-episode; no cross-episode state leak

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
        info         = step_data.get("info", {})
        total_reward += reward

        # Log includes autonomous-override details for debugging
        # Simplified string log for the validator
        print_log(f"[STEP] step={step_idx} action={action} reward={round(reward, 2)} done={done}")

        if done:
            break

        time.sleep(0.05)

    try:
        score = requests.get(f"{ENV_URL}/grade/{task_id}").json()["score"]
    except Exception:
        score = 0.0

    print_log(f"[END] task={task_id} score={score} steps={step_idx}\n")
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
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        print(f"\n  [INFO] Loading fine-tuned model from:\n         {HF_MODEL_ID}")
        _local_tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
        # CPU-safe loading adjustment
        has_cuda = torch.cuda.is_available()
        _local_model = AutoModelForCausalLM.from_pretrained(
            HF_MODEL_ID,
            torch_dtype=torch.float16 if has_cuda else torch.float32,
            device_map="auto" if has_cuda else None,
            load_in_4bit=has_cuda
        )
        _local_model.eval()
        print("  [INFO] Fine-tuned model loaded — used when Q-table misses a state.")
    except Exception as e:
        print(f"  [WARN] Fine-tuned model unavailable ({e}).")
        print("          Falling back to remote API / BFS heuristic.")
        _local_model, _local_tokenizer = None, None

    import argparse
    parser = argparse.ArgumentParser(description="Run GarbageBot Inference")
    parser.add_argument("--dynamic", action="store_true",
                        help="Interactive dynamic garbage placement")
    parser.add_argument("--task",
                        choices=["1","2","3","4","easy","medium","hard","all"],
                        default="all",
                        help="Task to run: 'easy', 'medium', 'hard', or 'all'")
    args = parser.parse_args()

    if args.task in ["1", "easy"]:
        tasks = ["task_easy"]
    elif args.task in ["2", "medium"]:
        tasks = ["task_medium"]
    elif args.task in ["3", "hard"]:
        tasks = ["task_hard"]
    else:
        tasks = ["task_easy", "task_medium", "task_hard"]

    print(f"\n  [INFO] Running tasks: {', '.join(tasks)}")

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

        if args.dynamic:
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