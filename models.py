from pydantic import BaseModel, ConfigDict
from typing import List, Literal, Optional, Tuple

# --- Custom observation and action logic ---

class Observation(BaseModel):
    model_config = ConfigDict(strict=True)
    grid_size: Tuple[int, int]
    robot_position: Tuple[int, int]
    garbage_positions: List[Tuple[int, int]]
    obstacle_positions: List[Tuple[int, int]]
    battery_level: int
    inventory_count: int
    message: str                  # Textual context for LLM

class Action(BaseModel):
    model_config = ConfigDict(strict=True)
    command: Literal["UP", "DOWN", "LEFT", "RIGHT", "COLLECT"]

# --- OpenEnv Standard Spec Models ---

class State(BaseModel):
    model_config = ConfigDict(strict=True)
    task_id: Optional[str]
    total_reward: float
    steps_taken: int
    done: bool

class ResetInput(BaseModel):
    task_id: str = "task_easy"

class ResetOutput(BaseModel):
    observation: Observation

class StepOutput(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict = {}

class Task(BaseModel):
    id: str
    name: str
    description: str
    difficulty: str
    reward_range: List[float]
