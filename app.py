"""
FastAPI server for the Garbage Collecting Robot OpenEnv environment.
Exposes reset / step / state / tasks / grade endpoints.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from environment import GarbageRobotEnv
from models import (
    Action, StepOutput, ResetInput, ResetOutput, CustomResetInput, State, Task,
)

app = FastAPI(
    title="Garbage Collecting Robot — OpenEnv",
    description=(
        "An OpenEnv-compliant robotics environment for garbage collection. "
        "AI agents must navigate a grid room to pick up garbage while managing battery constraints."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

env = GarbageRobotEnv()

TASKS = [
    Task(
        id="task_easy",
        name="Small Room Clean",
        description="Navigate a small 5x5 grid to collect 1 piece of garbage.",
        difficulty="easy",
        reward_range=[0.0, 1.0],
    ),
    Task(
        id="task_medium",
        name="Medium Room with Obstacles",
        description="Navigate a 7x7 grid to collect 3 pieces of garbage with limited battery.",
        difficulty="medium",
        reward_range=[0.0, 1.0],
    ),
    Task(
        id="task_hard",
        name="Large Maze Cleanup",
        description="Navigate a 10x10 maze avoiding obstacles to collect 5 pieces of garbage with strict battery usage.",
        difficulty="hard",
        reward_range=[0.0, 1.0],
    ),
]

VALID_IDS = {t.id for t in TASKS}

@app.get("/", tags=["health"])
def health():
    return {"status": "ok", "env": "garbage-collecting-robot"}

@app.post("/reset", response_model=ResetOutput, tags=["openenv"])
def reset(body: ResetInput = ResetInput()):
    if body.task_id not in VALID_IDS:
        raise HTTPException(400, f"task_id must be one of {sorted(VALID_IDS)}")
    state = env.reset(task_id=body.task_id)
    return {"observation": env.get_observation().dict()}

@app.post("/reset_custom", response_model=ResetOutput, tags=["openenv"])
def reset_custom(body: CustomResetInput):
    """
    Dynamic reset endpoint. Lets callers specify garbage positions,
    obstacle positions, robot start, grid size and battery at runtime.
    Any omitted field falls back to the base scenario's value.

    Example — drop 3 garbage tiles on a 7x7 medium base:
    POST /reset_custom
    {
      "task_id": "task_medium",
      "garbage_positions": [[0,6],[6,0],[3,3]]
    }

    Example — fully custom blank grid:
    POST /reset_custom
    {
      "task_id": "custom",
      "grid_size": [8,8],
      "robot_start": [0,0],
      "garbage_positions": [[7,7],[4,4]],
      "obstacle_positions": [[2,2],[2,3],[3,2]],
      "max_battery": 50
    }
    """
    env.reset_custom(
        task_id=body.task_id,
        grid_size=body.grid_size,
        robot_start=body.robot_start,
        garbage_positions=body.garbage_positions,
        obstacle_positions=body.obstacle_positions,
        max_battery=body.max_battery,
    )
    return {"observation": env.get_observation().dict()}

@app.post("/step", response_model=StepOutput, tags=["openenv"])
def step(body: Action):
    result = env.step(command=body.command)
    return result

@app.get("/state", response_model=State, tags=["openenv"])
def state():
    return env.state()

@app.get("/tasks", response_model=list[Task], tags=["openenv"])
def tasks():
    return TASKS

@app.get("/grade/{task_id}", tags=["grading"])
def grade(task_id: str):
    if task_id not in VALID_IDS:
        raise HTTPException(400, f"task_id must be one of {sorted(VALID_IDS)}")
    score = env.grade(task_id)
    return {"task_id": task_id, "score": score, "reward_range": [0.0, 1.0]}


# ── Policy endpoint (fine-tuned LLM) ──────────────────────
import os as _os

LOCAL_MODEL_PATH = _os.environ.get(
    "LOCAL_MODEL_PATH",
    "/home/robotics-mu/.unsloth/studio/exports/unsloth_Llama-3.2-3B-Instruct-bnb-4bit_1775629280"
)

_policy_model     = None
_policy_tokenizer = None
_policy_loaded    = False

def _load_policy():
    global _policy_model, _policy_tokenizer, _policy_loaded
    if _policy_loaded:
        return
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        _policy_tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
        _policy_model = AutoModelForCausalLM.from_pretrained(
            LOCAL_MODEL_PATH, torch_dtype=torch.float16, device_map="auto"
        )
        _policy_model.eval()
        print(f"[Policy] Fine-tuned model loaded from {LOCAL_MODEL_PATH}")
    except Exception as e:
        print(f"[Policy] Model unavailable: {e}")
    _policy_loaded = True


class PolicyInput(BaseModel):
    message: str   # the obs.message string from the environment

@app.post("/policy", tags=["openenv"])
def policy(body: PolicyInput):
    """
    Ask the fine-tuned LLM for the next action.
    Returns {"action": "UP|DOWN|LEFT|RIGHT|COLLECT", "source": "llm|bfs"}
    """
    _load_policy()
    VALID = ["UP", "DOWN", "LEFT", "RIGHT", "COLLECT"]

    if _policy_model is not None and _policy_tokenizer is not None:
        try:
            import torch
            instruction = (
                "You are an AI brain controlling a garbage collecting robot.\n"
                "Reply with EXACTLY ONE of: UP DOWN LEFT RIGHT COLLECT"
            )
            prompt = (
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\nENVIRONMENT STATUS:\n{body.message}\n\n"
                f"### Response:\n"
            )
            inputs = _policy_tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=512
            ).to(_policy_model.device)
            with torch.no_grad():
                outputs = _policy_model.generate(
                    **inputs, max_new_tokens=6, do_sample=False,
                    pad_token_id=_policy_tokenizer.eos_token_id
                )
            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            raw = _policy_tokenizer.decode(new_tokens, skip_special_tokens=True).strip().upper()
            for v in VALID:
                if v in raw:
                    return {"action": v, "source": "llm", "raw": raw}
        except Exception as e:
            print(f"[Policy] Inference error: {e}")

    # BFS fallback using current env state
    from inference import heuristic_action
    obs_dict = {
        "robot_position":    env.robot_position,
        "garbage_positions": env.garbage_positions,
        "obstacle_positions": env.obstacle_positions,
        "grid_size":         env.grid_size,
        "message":           body.message,
    }
    return {"action": heuristic_action(obs_dict), "source": "bfs"}


# ── Dynamic garbage placement ──────────────────────────────────
from pydantic import BaseModel
from typing import List, Tuple

class ConfigureInput(BaseModel):
    task_id: str = "task_easy"
    garbage_positions: List[List[int]]  # [[x,y], ...]

@app.post("/configure", tags=["openenv"])
def configure(body: ConfigureInput):
    """
    Reset the environment for task_id, then override garbage positions
    with whatever the caller supplies. Obstacles and grid size come from
    the standard scenario; only garbage is dynamic.
    """
    if body.task_id not in VALID_IDS:
        raise HTTPException(400, f"task_id must be one of {sorted(VALID_IDS)}")

    # Standard reset first (sets grid, obstacles, battery)
    env.reset(task_id=body.task_id)

    # Validate and override garbage positions
    validated = []
    for pos in body.garbage_positions:
        if len(pos) != 2:
            raise HTTPException(400, f"Each position must be [x, y], got {pos}")
        x, y = pos
        gw, gh = env.grid_size
        if not (0 <= x < gw and 0 <= y < gh):
            raise HTTPException(400, f"Position {pos} out of bounds for grid {env.grid_size}")
        if [x, y] in env.obstacle_positions:
            raise HTTPException(400, f"Position {pos} is an obstacle")
        validated.append([x, y])

    env.garbage_positions = validated

    return {"observation": env.get_observation().dict()}