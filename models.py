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

class CustomResetInput(BaseModel):
    """
    Fully dynamic reset — caller specifies the entire layout at runtime.
    grid_size, robot_start, garbage positions, obstacles, and battery
    are all optional overrides on top of a base task_id.
    Pass task_id='custom' to skip scenario defaults entirely.
    """
    task_id: str = "task_easy"
    grid_size: Optional[Tuple[int, int]] = None
    robot_start: Optional[Tuple[int, int]] = None
    garbage_positions: Optional[List[Tuple[int, int]]] = None
    obstacle_positions: Optional[List[Tuple[int, int]]] = None
    max_battery: Optional[int] = None

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