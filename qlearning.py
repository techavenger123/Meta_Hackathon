"""
qlearning.py — Tabular Q-Learning for the Garbage Collecting Robot.

Training runs directly against GarbageRobotEnv (no HTTP server needed).
The Q-table is persisted to disk as JSON and loaded by inference.py at startup.

State representation:
    (robot_x, robot_y, sorted_garbage_tuple)
    e.g. (2, 3, ((1,1),(4,4)))   — compact, hashable, fully describes the relevant world

Actions:
    0=UP  1=DOWN  2=LEFT  3=RIGHT  4=COLLECT

Usage:
    # Train all tasks and save
    python3 qlearning.py --train --episodes 8000

    # Evaluate silently (uses saved Q-table)
    python3 qlearning.py --eval
"""

import os
import json
import math
import random
import argparse
from collections import defaultdict
from environment import GarbageRobotEnv
from scenarios import SCENARIOS

# ── Constants ──────────────────────────────────────────────────────────────

ACTIONS      = ["UP", "DOWN", "LEFT", "RIGHT", "COLLECT"]
ACTION_IDX   = {a: i for i, a in enumerate(ACTIONS)}
Q_TABLE_PATH = os.environ.get("Q_TABLE_PATH", "qtable.json")

# ── Hyperparameters ─────────────────────────────────────────────────────────

ALPHA         = 0.15    # learning rate
GAMMA         = 0.97    # discount factor — high so the agent plans many steps ahead
EPSILON_START = 1.0     # full exploration at the start
EPSILON_END   = 0.05    # minimum exploration floor
EPSILON_DECAY = 0.9995  # multiplicative decay per episode


# ── State Encoding ──────────────────────────────────────────────────────────

def encode_state(obs: dict) -> tuple:
    """
    Convert a raw observation dict (from env.step or env.get_observation)
    into a hashable tuple suitable as a Q-table key.

    We include:
      - robot position (x, y)
      - sorted garbage positions (so order doesn't create phantom new states)

    Obstacles and grid size are fixed per task — no need to encode them.
    """
    rx, ry = obs["robot_position"]
    garbage = tuple(sorted((int(g[0]), int(g[1])) for g in obs["garbage_positions"]))
    return (rx, ry, garbage)


# ── Q-Table ─────────────────────────────────────────────────────────────────

class QTable:
    """
    Dictionary-backed Q-table with defaultdict initialisation.
    Values default to a small optimistic initial value to encourage exploration.
    """

    def __init__(self, optimistic_init: float = 0.5):
        self.optimistic_init = optimistic_init
        # q[state][action_index] = Q-value
        self._q: dict[tuple, list[float]] = {}

    def _ensure(self, state: tuple):
        if state not in self._q:
            self._q[state] = [self.optimistic_init] * len(ACTIONS)

    def get(self, state: tuple, action_idx: int) -> float:
        self._ensure(state)
        return self._q[state][action_idx]

    def update(self, state: tuple, action_idx: int, value: float):
        self._ensure(state)
        self._q[state][action_idx] = value

    def best_action(self, state: tuple) -> int:
        """Return the index of the greedy best action."""
        self._ensure(state)
        return int(max(range(len(ACTIONS)), key=lambda i: self._q[state][i]))

    def best_q(self, state: tuple) -> float:
        self._ensure(state)
        return max(self._q[state])

    # ── Persistence ─────────────────────────────────────────────────────────

    def save(self, path: str = Q_TABLE_PATH):
        # JSON requires string keys; encode tuples as strings
        serialisable = {json.dumps(list(k)): v for k, v in self._q.items()}
        with open(path, "w") as f:
            json.dump(serialisable, f)
        print(f"[Q-Table] Saved {len(self._q):,} states → {path}")

    def load(self, path: str = Q_TABLE_PATH) -> bool:
        if not os.path.exists(path):
            return False
        with open(path, "r") as f:
            raw = json.load(f)
        self._q = {}
        for k_str, v in raw.items():
            k = tuple(
                tuple(x) if isinstance(x, list) else x
                for x in json.loads(k_str)
            )
            # Nested garbage tuple: last element is a list of [x,y] pairs
            rebuilt = []
            for elem in json.loads(k_str):
                if isinstance(elem, list) and len(elem) == 2 and all(isinstance(e, int) for e in elem):
                    rebuilt.append(tuple(elem))  # flat [x, y] → (x, y)
                elif isinstance(elem, list):
                    rebuilt.append(tuple(tuple(p) for p in elem))  # nested garbage tuples
                else:
                    rebuilt.append(elem)
            self._q[tuple(rebuilt)] = v
        print(f"[Q-Table] Loaded {len(self._q):,} states ← {path}")
        return True

    def __len__(self):
        return len(self._q)


# ── Observation Helper ───────────────────────────────────────────────────────

def _obs_from_step(step_result: dict) -> dict:
    """Unpack observation from env.step() return dict (already a plain dict)."""
    return step_result["observation"]


def _obs_from_env(env) -> dict:
    """
    Build an obs dict directly from GarbageRobotEnv fields.
    Avoids calling .model_dump()/.dict() since the Pydantic version may vary.
    """
    obs_obj = env.get_observation()
    return {
        "robot_position":    obs_obj.robot_position,
        "garbage_positions": list(obs_obj.garbage_positions),
        "obstacle_positions": list(obs_obj.obstacle_positions),
        "grid_size":         obs_obj.grid_size,
        "battery_level":     obs_obj.battery_level,
        "inventory_count":   obs_obj.inventory_count,
        "message":           obs_obj.message,
    }


# ── Training ─────────────────────────────────────────────────────────────────

def train(
    task_ids: list[str] | None = None,
    episodes: int = 8000,
    qtable: QTable | None = None,
    verbose: bool = True,
) -> QTable:
    """
    Run Q-learning over the given task_ids for `episodes` total episodes.
    Tasks are sampled uniformly so the agent generalises across difficulties.
    """
    if task_ids is None:
        task_ids = list(SCENARIOS.keys())

    if qtable is None:
        qtable = QTable()

    env     = GarbageRobotEnv()
    epsilon = EPSILON_START

    best_scores: dict[str, float] = {t: 0.0 for t in task_ids}

    for ep in range(1, episodes + 1):
        task_id = random.choice(task_ids)
        state_obj = env.reset(task_id)
        obs       = _obs_from_env(env)   # Pydantic-version-safe
        state     = encode_state(obs)

        total_reward = 0.0
        done         = False

        while not done:
            # ε-greedy action selection
            if random.random() < epsilon:
                action_idx = random.randrange(len(ACTIONS))
            else:
                action_idx = qtable.best_action(state)

            action   = ACTIONS[action_idx]
            result     = env.step(action)
            next_obs   = result["observation"]   # already a dict from env.step()
            reward     = result["reward"]
            done       = result["done"]

            next_state = encode_state(next_obs)

            # Bellman update
            old_q    = qtable.get(state, action_idx)
            td_target = reward + (0.0 if done else GAMMA * qtable.best_q(next_state))
            new_q    = old_q + ALPHA * (td_target - old_q)
            qtable.update(state, action_idx, new_q)

            state = next_state
            obs   = next_obs
            total_reward += reward

        # Track best score per task
        score = env.grade(task_id)
        if score > best_scores[task_id]:
            best_scores[task_id] = score

        # Decay exploration
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        if verbose and ep % 500 == 0:
            avg_best = sum(best_scores.values()) / len(best_scores)
            print(
                f"  Ep {ep:5d}/{episodes}  ε={epsilon:.4f}  "
                f"states={len(qtable):,}  "
                f"best_scores={best_scores}  avg={avg_best:.2f}"
            )

    return qtable


# ── Inference Helper (used by inference.py) ──────────────────────────────────

class QLearningAgent:
    """
    Thin wrapper around a loaded Q-table for use by inference.py.
    Falls through (returns None) when the state has never been seen during training.
    """

    def __init__(self, path: str = Q_TABLE_PATH):
        self.qtable = QTable()
        self.loaded = self.qtable.load(path)

    def get_action(self, obs: dict) -> str | None:
        """
        Return the greedy Q-table action for `obs`, or None if the state is unknown.
        inference.py should fall back to BFS when None is returned.
        """
        if not self.loaded:
            return None
        state = encode_state(obs)
        if state not in self.qtable._q:
            return None  # unseen state — let BFS handle it
        return ACTIONS[self.qtable.best_action(state)]


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(qtable: QTable, task_ids: list[str] | None = None, runs: int = 5) -> dict:
    """Run `runs` greedy episodes per task and return average scores."""
    if task_ids is None:
        task_ids = list(SCENARIOS.keys())

    env     = GarbageRobotEnv()
    results = {}

    for task_id in task_ids:
        scores = []
        for _ in range(runs):
            env.reset(task_id)
            obs  = _obs_from_env(env)   # Pydantic-version-safe
            done = False
            while not done:
                state      = encode_state(obs)
                action_idx = qtable.best_action(state)
                result     = env.step(ACTIONS[action_idx])
                obs        = result["observation"]  # already a dict
                done       = result["done"]
            scores.append(env.grade(task_id))
        avg = sum(scores) / len(scores)
        results[task_id] = round(avg, 3)
        print(f"  {task_id:12s}  avg score = {avg:.3f}  ({scores})")

    return results


# ── CLI Entry Point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Q-Learning for Garbage Robot")
    parser.add_argument("--train",    action="store_true", help="Run training")
    parser.add_argument("--eval",     action="store_true", help="Run evaluation only")
    parser.add_argument("--episodes", type=int, default=8000, help="Training episodes (default 8000)")
    parser.add_argument("--tasks",    nargs="+", default=None,
                        help="Task IDs to train on (default: all)")
    parser.add_argument("--output",   default=Q_TABLE_PATH, help="Q-table save path")
    args = parser.parse_args()

    if args.train:
        print("=" * 55)
        print("  Q-Learning Training — Garbage Collecting Robot")
        print("=" * 55)
        task_ids = args.tasks or list(SCENARIOS.keys())
        print(f"  Tasks    : {task_ids}")
        print(f"  Episodes : {args.episodes}")
        print(f"  α={ALPHA}  γ={GAMMA}  ε {EPSILON_START}→{EPSILON_END}  decay={EPSILON_DECAY}")
        print()

        qt = train(task_ids=task_ids, episodes=args.episodes, verbose=True)
        qt.save(args.output)

        print("\n  — Evaluation on greedy policy —")
        evaluate(qt, task_ids)

    elif args.eval:
        print("=" * 55)
        print("  Q-Learning Evaluation")
        print("=" * 55)
        qt = QTable()
        if not qt.load(args.output):
            print(f"[ERROR] No Q-table found at {args.output}. Run with --train first.")
        else:
            evaluate(qt)
    else:
        parser.print_help()
