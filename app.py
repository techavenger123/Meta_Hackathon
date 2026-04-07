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
    return {"observation": env.get_observation().model_dump()}

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
    return {"observation": env.get_observation().model_dump()}

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


# ── Dynamic garbage placement ──────────────────────────────
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

    return {"observation": env.get_observation().model_dump()}