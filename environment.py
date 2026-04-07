from typing import Any, Dict
from models import Observation, State
from scenarios import SCENARIOS

class GarbageRobotEnv:
    """
    Core RL Environment for the Garbage Collecting Robot.
    """
    def __init__(self):
        self.current_task_id = None
        self.grid_size = (0, 0)
        self.robot_position = (0, 0)
        self.garbage_positions = []
        self.obstacle_positions = []
        self.battery_level = 0
        self.max_battery = 0
        self.inventory_count = 0
        
        self.total_reward = 0.0
        self.steps_taken = 0
        self.done = False

    def reset(self, task_id: str) -> State:
        if task_id not in SCENARIOS:
            raise ValueError(f"Task ID {task_id} not found in scenarios.")
        
        scenario = SCENARIOS[task_id]
        self.current_task_id = task_id
        self.grid_size = scenario["grid_size"]
        self.robot_position = list(scenario["robot_start"])
        self.garbage_positions = [list(g) for g in scenario["garbage_starts"]]
        self.obstacle_positions = [list(o) for o in scenario["obstacle_starts"]]
        self.battery_level = scenario["max_battery"]
        self.max_battery = scenario["max_battery"]
        self.inventory_count = 0
        
        self.total_reward = 0.0
        self.steps_taken = 0
        self.done = False
        
        return self.state()

    def get_observation(self, message: str = "") -> Observation:
        if not message:
            message = f"You are at {tuple(self.robot_position)}. "
            if self.garbage_positions:
                message += f"Garbage remaining: {len(self.garbage_positions)}. "
            else:
                message += "All garbage collected! "
            message += f"Battery: {self.battery_level}/{self.max_battery}."

        return Observation(
            grid_size=self.grid_size,
            robot_position=tuple(self.robot_position),
            garbage_positions=[tuple(g) for g in self.garbage_positions],
            obstacle_positions=[tuple(o) for o in self.obstacle_positions],
            battery_level=self.battery_level,
            inventory_count=self.inventory_count,
            message=message
        )

    def state(self) -> State:
        return State(
            task_id=self.current_task_id,
            total_reward=self.total_reward,
            steps_taken=self.steps_taken,
            done=self.done
        )

    def step(self, command: str) -> Dict[str, Any]:
        if self.done:
            obs = self.get_observation("Episode already finished.")
            return {"observation": obs.model_dump(), "reward": 0.0, "done": True, "info": {}}

        reward = -0.1  # small base penalty for taking a step
        message = ""
        
        self.battery_level -= 1
        self.steps_taken += 1

        new_pos = list(self.robot_position)
        
        if command == "UP":
            new_pos[1] += 1
        elif command == "DOWN":
            new_pos[1] -= 1
        elif command == "LEFT":
            new_pos[0] -= 1
        elif command == "RIGHT":
            new_pos[0] += 1
        elif command == "COLLECT":
            # Check if garbage is at current pos
            if self.robot_position in self.garbage_positions:
                self.garbage_positions.remove(self.robot_position)
                self.inventory_count += 1
                reward += 10.0
                message = "Collected garbage!"
            else:
                reward -= 1.0
                message = "No garbage to collect here."
        else:
            reward -= 1.0
            message = f"Invalid command: {command}."

        # Apply movement logic if movement command
        if command in ["UP", "DOWN", "LEFT", "RIGHT"]:
            # Check grid bounds
            if (0 <= new_pos[0] < self.grid_size[0]) and (0 <= new_pos[1] < self.grid_size[1]):
                if new_pos in self.obstacle_positions:
                    reward -= 5.0
                    # Tell the agent exactly which directions are blocked from current pos
                    blocked = []
                    direction_map = {"UP": [0,1], "DOWN": [0,-1], "LEFT": [-1,0], "RIGHT": [1,0]}
                    for d, delta in direction_map.items():
                        neighbour = [self.robot_position[0]+delta[0], self.robot_position[1]+delta[1]]
                        if neighbour in self.obstacle_positions:
                            blocked.append(d)
                    blocked_str = ", ".join(blocked) if blocked else "none"
                    message = (
                        f"BLOCKED! {command} is an obstacle. Do NOT try {command} again. "
                        f"Blocked directions from here: {blocked_str}. "
                        f"Choose a different direction."
                    )
                else:
                    self.robot_position = new_pos
                    message = f"Moved {command}."
            else:
                reward -= 1.0
                message = f"Hit a wall trying to move {command}. Do NOT try {command} again from this position."

        # Check termination conditions
        if len(self.garbage_positions) == 0:
            self.done = True
            reward += 50.0  # Completion bonus
            message += " All garbage collected! Task complete."
        elif self.battery_level <= 0:
            self.done = True
            message += " Battery depleted! Game over."

        self.total_reward += reward

        return {
            "observation": self.get_observation(message).model_dump(),
            "reward": reward,
            "done": self.done,
            "info": {"inventory_count": self.inventory_count, "steps": self.steps_taken}
        }

    def grade(self, task_id: str) -> float:
        # Standardised metric [0.0, 1.0] for leaderboard
        if task_id not in SCENARIOS:
            return 0.0
        
        # Max out at 1.0 if all garbage collected, proportional otherwise
        total_starting_garbage = len(SCENARIOS[task_id]["garbage_starts"])
        collected = self.inventory_count
        
        return min(max(collected / total_starting_garbage, 0.0), 1.0)