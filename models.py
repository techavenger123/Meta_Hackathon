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
    message: str                        # Textual context for LLM

    # ── Autonomous resource-management fields ──────────────────
    home_position: Tuple[int, int]      # Charging station coordinates
    unload_station: Tuple[int, int]     # Designated unload-corner coordinates
    storage_capacity: int               # Max items robot can carry before unloading
    current_storage_load: int           # Items currently held (resets after unload)
    distance_from_home: int             # BFS steps to home (-1 if unreachable)
    robot_mode: str                     # 'normal' | 'recharging' | 'unloading'


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

    # ── Extended state for resource management ─────────────────
    robot_mode: str = "normal"
    current_storage_load: int = 0
    battery_level: int = 0
    distance_from_home: int = 0


class ResetInput(BaseModel):
    task_id: str = "task_easy"

class CustomResetInput(BaseModel):
    """
    Fully dynamic reset — caller specifies the entire layout at runtime.
    grid_size, robot_start, garbage positions, obstacles, battery, storage_capacity,
    home_position and unload_station are all optional overrides on top of a base task_id.
    Pass task_id='custom' to skip scenario defaults entirely.
    """
    task_id: str = "task_easy"
    grid_size: Optional[Tuple[int, int]] = None
    robot_start: Optional[Tuple[int, int]] = None
    garbage_positions: Optional[List[Tuple[int, int]]] = None
    obstacle_positions: Optional[List[Tuple[int, int]]] = None
    max_battery: Optional[int] = None
    storage_capacity: Optional[int] = None
    home_position: Optional[Tuple[int, int]] = None
    unload_station: Optional[Tuple[int, int]] = None

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