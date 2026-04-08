"""
FastAPI server for the Garbage Collecting Robot OpenEnv environment.
Exposes reset / step / state / tasks / grade / policy / configure endpoints.

Fix applied:
  - /policy BFS fallback now uses env.get_observation().dict() instead of
    a hand-built incomplete dict (which was missing robot_mode, home_position,
    unload_station, current_storage_load, storage_capacity, distance_from_home).
  - Static files and /ui route added so the HTML dashboard is served from the
    same origin — required for HuggingFace Spaces deployment.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from typing import List
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

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
    """
    env.reset_custom(
        task_id=body.task_id,
        grid_size=body.grid_size,
        robot_start=body.robot_start,
        garbage_positions=body.garbage_positions,
        obstacle_positions=body.obstacle_positions,
        max_battery=body.max_battery,
        storage_capacity=body.storage_capacity,
        home_position=body.home_position,
        unload_station=body.unload_station,
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


# ── Policy endpoint (fine-tuned LLM) ──────────────────────────────────────

HF_MODEL_ID = os.environ.get(
    "HF_MODEL_ID",
    "TechAvenger/GarbageBot-Weights"
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
        _policy_tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
        
        # CPU-safe loading adjustment
        has_cuda = torch.cuda.is_available()
        _policy_model = AutoModelForCausalLM.from_pretrained(
            HF_MODEL_ID,
            torch_dtype=torch.float16 if has_cuda else torch.float32,
            device_map="auto" if has_cuda else None,
            load_in_4bit=has_cuda
        )
        _policy_model.eval()
        print(f"[Policy] Fine-tuned model loaded from {HF_MODEL_ID}")
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

    # FIX: use env.get_observation().dict() so heuristic_action() receives
    # all required fields (robot_mode, home_position, unload_station, etc.)
    # instead of the previous hand-built incomplete dict.
    from inference import heuristic_action
    obs_dict = env.get_observation().dict()
    obs_dict["message"] = body.message   # use the caller's message for context
    return {"action": heuristic_action(obs_dict), "source": "bfs"}


# ── Dynamic garbage placement ──────────────────────────────────────────────

class ConfigureInput(BaseModel):
    task_id: str = "task_easy"
    garbage_positions: List[List[int]]  # [[x,y], ...]

@app.post("/configure", tags=["openenv"])
def configure(body: ConfigureInput):
    """
    Reset the environment for task_id, then override garbage positions
    with whatever the caller supplies.
    """
    if body.task_id not in VALID_IDS:
        raise HTTPException(400, f"task_id must be one of {sorted(VALID_IDS)}")

    env.reset(task_id=body.task_id)

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


# ── Serve HTML dashboard ───────────────────────────────────────────────────
# This makes the frontend accessible at /ui on the same origin as the API,
# which is required for HuggingFace Spaces (no localhost cross-origin issues).

@app.get("/ui", include_in_schema=False)
def ui():
    """Serve the dashboard HTML."""
    return FileResponse("frontend/index.html")

# Mount static assets (style.css, script.js) at /static
if os.path.exists("frontend/style.css") or os.path.exists("frontend/script.js"):
    app.mount("/static", StaticFiles(directory="frontend"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)