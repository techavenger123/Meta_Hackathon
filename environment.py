"""
environment.py — Garbage Collecting Robot Core RL Environment.

Fixes applied:
  • Battery no longer drains during autonomous CHARGE / UNLOAD_HERE steps.
  • Recharge guard now checks `not self.done` instead of `self.garbage_positions`
    so it also fires correctly at episode boundaries.
"""

from typing import Any, Dict, Optional, List, Tuple
from collections import deque
from models import Observation, State
from scenarios import SCENARIOS


# ─────────────────────────────────────────────────────────────
# BFS PATHFINDING HELPER
# ─────────────────────────────────────────────────────────────

def _bfs(
    start,
    goal,
    obstacles,
    grid_w: int,
    grid_h: int,
) -> Tuple[Optional[str], float]:
    """
    Breadth-First Search from *start* to *goal* on a rectangular grid.

    Avoids all cells listed in *obstacles*.  Returns:
        (first_direction, path_length)  — the single step that begins the
                                          shortest path, and how many steps
                                          the full path takes.
        (None, 0)                        — start == goal (already there).
        (None, inf)                      — goal is unreachable.

    Directions: "UP" (+y), "DOWN" (−y), "LEFT" (−x), "RIGHT" (+x).
    """
    start = (int(start[0]), int(start[1]))
    goal  = (int(goal[0]),  int(goal[1]))

    if start == goal:
        return (None, 0)

    obstacle_set = frozenset((int(o[0]), int(o[1])) for o in obstacles)
    dirs = [("RIGHT", (1, 0)), ("LEFT", (-1, 0)), ("UP", (0, 1)), ("DOWN", (0, -1))]

    queue: deque = deque([(start, None, 0)])   # (pos, first_move, depth)
    visited = {start}

    while queue:
        pos, first, depth = queue.popleft()
        for name, (dx, dy) in dirs:
            npos = (pos[0] + dx, pos[1] + dy)
            if not (0 <= npos[0] < grid_w and 0 <= npos[1] < grid_h):
                continue
            if npos in obstacle_set or npos in visited:
                continue
            move = first if first else name
            if npos == goal:
                return (move, depth + 1)
            visited.add(npos)
            queue.append((npos, move, depth + 1))

    return (None, float("inf"))


# ─────────────────────────────────────────────────────────────
# ENVIRONMENT
# ─────────────────────────────────────────────────────────────

class GarbageRobotEnv:
    """
    Core RL Environment for the Garbage Collecting Robot.

    Robot modes
    -----------
    MODE_NORMAL   — agent controls the robot normally.
    MODE_RECHARGE — battery critically low; robot auto-navigates home,
                    recharges, then switches back to NORMAL.
    MODE_UNLOAD   — storage full; robot auto-navigates to unload_station,
                    empties its bin, then switches back to NORMAL.

    Autonomous overrides happen *inside* step(): the command the caller
    sends is silently replaced when the robot is in a non-normal mode.
    This keeps the external API unchanged while giving the robot
    self-managing capabilities.

    FIX: Battery is only decremented for real movement/collection commands,
         NOT for internal CHARGE or UNLOAD_HERE commands.
    """

    MODE_NORMAL   = "normal"
    MODE_RECHARGE = "recharging"
    MODE_UNLOAD   = "unloading"

    # Safety margin added on top of BFS distance when deciding to recharge.
    RECHARGE_BUFFER = 4

    def __init__(self):
        self.current_task_id    = None
        self.grid_size          = (0, 0)
        self.robot_position     = [0, 0]
        self.garbage_positions  = []
        self.obstacle_positions = []
        self.battery_level      = 0
        self.max_battery        = 0
        self.inventory_count    = 0

        # Resource management state
        self.home_position        = [0, 0]
        self.unload_station       = [0, 0]
        self.storage_capacity     = 6
        self.current_storage_load = 0

        # Episode accounting
        self.total_reward = 0.0
        self.steps_taken  = 0
        self.done         = False

        # Autonomous navigation mode
        self._mode = self.MODE_NORMAL

    # ── Reset ─────────────────────────────────────────────────

    def reset(self, task_id: str) -> State:
        if task_id not in SCENARIOS:
            raise ValueError(f"Task ID '{task_id}' not found in scenarios.")

        s = SCENARIOS[task_id]
        self.current_task_id    = task_id
        self.grid_size          = tuple(s["grid_size"])
        self.robot_position     = list(s["robot_start"])
        self.garbage_positions  = [list(g) for g in s["garbage_starts"]]
        self.obstacle_positions = [list(o) for o in s["obstacle_starts"]]
        self.battery_level      = s["max_battery"]
        self.max_battery        = s["max_battery"]

        self.home_position        = list(s.get("home_position", s["robot_start"]))
        self.unload_station       = list(s.get("unload_station", [0, self.grid_size[1] - 1]))
        self.storage_capacity     = s.get("storage_capacity", 6)
        self.current_storage_load = 0
        self.inventory_count      = 0

        self.total_reward = 0.0
        self.steps_taken  = 0
        self.done         = False
        self._mode        = self.MODE_NORMAL

        return self.state()

    def reset_custom(
        self,
        task_id: str = "task_easy",
        grid_size=None,
        robot_start=None,
        garbage_positions=None,
        obstacle_positions=None,
        max_battery=None,
        storage_capacity=None,
        home_position=None,
        unload_station=None,
    ) -> State:
        """
        Dynamic reset: start from a scenario baseline and override any fields.
        Pass task_id='custom' with all fields supplied to skip scenario lookup.
        """
        if task_id in SCENARIOS:
            s = SCENARIOS[task_id]
            base_grid       = s["grid_size"]
            base_robot      = s["robot_start"]
            base_garbage    = s["garbage_starts"]
            base_obstacles  = s["obstacle_starts"]
            base_battery    = s["max_battery"]
            base_home       = s.get("home_position", s["robot_start"])
            base_unload     = s.get("unload_station", [0, s["grid_size"][1] - 1])
            base_capacity   = s.get("storage_capacity", 5)
        else:
            base_grid      = (10, 10)
            base_robot     = (0, 0)
            base_garbage   = []
            base_obstacles = []
            base_battery   = 60
            base_home      = (0, 0)
            base_unload    = (9, 0)
            base_capacity  = 6

        self.current_task_id    = task_id
        self.grid_size          = tuple(grid_size)        if grid_size        is not None else tuple(base_grid)
        self.robot_position     = list(robot_start)       if robot_start      is not None else list(base_robot)
        self.garbage_positions  = [list(g) for g in garbage_positions]  if garbage_positions  is not None else [list(g) for g in base_garbage]
        self.obstacle_positions = [list(o) for o in obstacle_positions] if obstacle_positions is not None else [list(o) for o in base_obstacles]
        self.battery_level      = max_battery             if max_battery      is not None else base_battery
        self.max_battery        = self.battery_level
        self.home_position      = list(home_position)     if home_position    is not None else list(base_home)
        self.unload_station     = list(unload_station)    if unload_station   is not None else list(base_unload)
        self.storage_capacity   = storage_capacity        if storage_capacity is not None else base_capacity

        self.current_storage_load = 0
        self.inventory_count      = 0
        self.total_reward         = 0.0
        self.steps_taken          = 0
        self.done                 = False
        self._mode                = self.MODE_NORMAL

        # Remove any garbage placed on top of an obstacle
        self.garbage_positions = [
            g for g in self.garbage_positions if g not in self.obstacle_positions
        ]
        return self.state()

    # ── Observation & State helpers ───────────────────────────

    def _bfs_distance(self, target) -> int:
        """Return BFS step-count from current robot position to *target*."""
        _, dist = _bfs(
            self.robot_position, target,
            self.obstacle_positions, self.grid_size[0], self.grid_size[1],
        )
        return int(dist) if dist != float("inf") else -1

    def _should_recharge(self) -> bool:
        """
        Return True when the robot must leave immediately to reach home
        before battery runs out.

        Threshold = BFS distance to home + RECHARGE_BUFFER.
        A buffer of 4 gives comfortable headroom for obstacle detours.
        """
        if self.battery_level <= 1:
            return True
        dist = self._bfs_distance(self.home_position)
        if dist < 0:
            # Home unreachable via BFS — fall back to Manhattan distance
            dist = (abs(self.robot_position[0] - self.home_position[0]) +
                    abs(self.robot_position[1] - self.home_position[1]))
        return self.battery_level <= (dist + self.RECHARGE_BUFFER)

    def _should_unload(self) -> bool:
        """Return True when the storage bin is at capacity."""
        return self.current_storage_load >= self.storage_capacity

    def get_observation(self, message: str = "") -> Observation:
        dist_home = self._bfs_distance(self.home_position)

        if not message:
            message = (
                f"You are at {tuple(self.robot_position)}. "
                f"Garbage remaining: {len(self.garbage_positions)}. "
                f"Battery: {self.battery_level}/{self.max_battery}. "
                f"Storage: {self.current_storage_load}/{self.storage_capacity}. "
                f"Home (charging): {tuple(self.home_position)} "
                f"[{dist_home if dist_home >= 0 else 'unreachable'} steps]. "
                f"Unload station: {tuple(self.unload_station)}. "
                f"Mode: {self._mode}."
            )

        return Observation(
            grid_size          = self.grid_size,
            robot_position     = tuple(self.robot_position),
            garbage_positions  = [tuple(g) for g in self.garbage_positions],
            obstacle_positions = [tuple(o) for o in self.obstacle_positions],
            battery_level      = self.battery_level,
            inventory_count    = self.inventory_count,
            message            = message,
            home_position        = tuple(self.home_position),
            unload_station       = tuple(self.unload_station),
            storage_capacity     = self.storage_capacity,
            current_storage_load = self.current_storage_load,
            distance_from_home   = dist_home,
            robot_mode           = self._mode,
        )

    def state(self) -> State:
        return State(
            task_id              = self.current_task_id,
            total_reward         = self.total_reward,
            steps_taken          = self.steps_taken,
            done                 = self.done,
            robot_mode           = self._mode,
            current_storage_load = self.current_storage_load,
            battery_level        = self.battery_level,
            distance_from_home   = self._bfs_distance(self.home_position),
        )

    # ── Autonomous command resolver ────────────────────────────

    def _resolve_command(self, requested: str) -> Tuple[str, str]:
        """
        Determine the *effective* command for this step.

        When the robot is in MODE_RECHARGE or MODE_UNLOAD the caller's
        command is replaced by an autonomously-computed one.

        Returns
        -------
        (effective_command, mode_message)
        """

        # ── Trigger check (only when in normal mode) ───────────
        # FIX: use `not self.done` guard instead of `self.garbage_positions`
        # so recharge still fires even if all garbage is collected this step.
        if self._mode == self.MODE_NORMAL:
            if self._should_recharge() and not self.done:
                self._mode = self.MODE_RECHARGE
            elif self._should_unload():
                self._mode = self.MODE_UNLOAD

        # ── Recharging mode ────────────────────────────────────
        if self._mode == self.MODE_RECHARGE:
            if tuple(self.robot_position) == tuple(self.home_position):
                # Arrived — charge and return to normal
                self._mode = self.MODE_NORMAL
                return (
                    "CHARGE",
                    (f"Reached charging station {tuple(self.home_position)}. "
                     f"Battery fully restored to {self.max_battery}. "
                     f"Resuming garbage collection."),
                )
            else:
                move, dist = _bfs(
                    self.robot_position, self.home_position,
                    self.obstacle_positions, self.grid_size[0], self.grid_size[1],
                )
                dist_str = f"{int(dist)} steps" if dist != float("inf") else "route blocked"
                return (
                    move or "UP",
                    (f"⚡ Battery critical ({self.battery_level}/{self.max_battery}). "
                     f"Auto-navigating to charging station {tuple(self.home_position)} "
                     f"[{dist_str}]."),
                )

        # ── Unloading mode ─────────────────────────────────────
        if self._mode == self.MODE_UNLOAD:
            if tuple(self.robot_position) == tuple(self.unload_station):
                # Arrived — empty the bin and return to normal
                freed      = self.current_storage_load
                self._mode = self.MODE_NORMAL
                return (
                    "UNLOAD_HERE",
                    (f"Reached unload station {tuple(self.unload_station)}. "
                     f"Emptied {freed} item(s) from storage. "
                     f"Resuming garbage collection."),
                )
            else:
                move, dist = _bfs(
                    self.robot_position, self.unload_station,
                    self.obstacle_positions, self.grid_size[0], self.grid_size[1],
                )
                dist_str = f"{int(dist)} steps" if dist != float("inf") else "route blocked"
                return (
                    move or "UP",
                    (f"📦 Storage full ({self.current_storage_load}/{self.storage_capacity}). "
                     f"Auto-navigating to unload station {tuple(self.unload_station)} "
                     f"[{dist_str}]."),
                )

        # ── Normal mode — use caller's command ─────────────────
        return (requested, "")

    # ── Step ──────────────────────────────────────────────────

    def step(self, command: str) -> Dict[str, Any]:
        if self.done:
            obs = self.get_observation("Episode already finished.")
            return {"observation": obs.dict(), "reward": 0.0, "done": True, "info": {}}

        self.steps_taken += 1

        # Resolve autonomous overrides BEFORE battery decrement so that
        # CHARGE / UNLOAD_HERE commands do NOT consume battery.
        effective_cmd, mode_message = self._resolve_command(command)

        # FIX: only drain battery for real movement / collection actions.
        # Autonomous internal commands (CHARGE, UNLOAD_HERE) are free.
        if effective_cmd in ("CHARGE", "UNLOAD_HERE"):
            reward = 0.0
        else:
            self.battery_level -= 1
            reward = -0.1

        message = mode_message  # may be overwritten below

        # ── CHARGE (internal — issued autonomously at home) ────
        if effective_cmd == "CHARGE":
            self.battery_level = self.max_battery
            reward += 5.0
            # message already set from resolver

        # ── UNLOAD_HERE (internal — issued autonomously at station) ──
        elif effective_cmd == "UNLOAD_HERE":
            freed                     = self.current_storage_load
            self.current_storage_load = 0
            reward += 2.0
            # message already set from resolver

        # ── COLLECT ───────────────────────────────────────────
        elif effective_cmd == "COLLECT":
            if self.robot_position in self.garbage_positions:
                self.garbage_positions.remove(self.robot_position)
                self.inventory_count      += 1
                self.current_storage_load += 1
                reward += 10.0
                message = (
                    f"Collected garbage! "
                    f"Storage: {self.current_storage_load}/{self.storage_capacity}."
                )
                if self._should_unload() and self.garbage_positions:
                    self._mode = self.MODE_UNLOAD
                    message += (
                        f" Storage full — auto-routing to "
                        f"unload station {tuple(self.unload_station)}."
                    )
            else:
                reward  -= 1.0
                message  = "No garbage to collect here."

        # ── Movement commands ──────────────────────────────────
        elif effective_cmd in ("UP", "DOWN", "LEFT", "RIGHT"):
            new_pos = list(self.robot_position)
            if effective_cmd == "UP":
                new_pos[1] += 1
            elif effective_cmd == "DOWN":
                new_pos[1] -= 1
            elif effective_cmd == "LEFT":
                new_pos[0] -= 1
            elif effective_cmd == "RIGHT":
                new_pos[0] += 1

            gw, gh = self.grid_size
            if 0 <= new_pos[0] < gw and 0 <= new_pos[1] < gh:
                if new_pos in self.obstacle_positions:
                    reward -= 5.0
                    blocked = []
                    direction_map = {
                        "UP":    [0,  1], "DOWN":  [0, -1],
                        "LEFT": [-1,  0], "RIGHT": [1,  0],
                    }
                    for d, delta in direction_map.items():
                        nb = [self.robot_position[0] + delta[0],
                              self.robot_position[1] + delta[1]]
                        if nb in self.obstacle_positions:
                            blocked.append(d)
                    blocked_str = ", ".join(blocked) if blocked else "none"
                    message = (
                        f"BLOCKED! {effective_cmd} leads to an obstacle. "
                        f"Blocked directions from here: {blocked_str}. "
                        f"Choose a different direction."
                    )
                else:
                    self.robot_position = new_pos
                    if not message:
                        message = f"Moved {effective_cmd}."
            else:
                reward -= 1.0
                if not message:
                    message = (
                        f"Hit a wall trying to move {effective_cmd}. "
                        f"Do NOT try {effective_cmd} again from this position."
                    )

        # ── Unknown command ────────────────────────────────────
        else:
            reward  -= 1.0
            message  = f"Invalid command: '{effective_cmd}'."

        # ── Termination checks ─────────────────────────────────
        if len(self.garbage_positions) == 0:
            self.done  = True
            reward    += 50.0
            message   += " All garbage collected! Task complete."
        elif self.battery_level <= 0:
            self.done  = True
            message   += " Battery depleted! Game over."

        self.total_reward += reward

        return {
            "observation": self.get_observation(message).dict(),
            "reward":      reward,
            "done":        self.done,
            "info": {
                "inventory_count":      self.inventory_count,
                "steps":                self.steps_taken,
                "current_storage_load": self.current_storage_load,
                "robot_mode":           self._mode,
                "autonomous_override":  effective_cmd != command,
                "original_command":     command,
                "effective_command":    effective_cmd,
            },
        }

    # ── Grading ───────────────────────────────────────────────

    def grade(self, task_id: str) -> float:
        """Normalised [0.0, 1.0] completion score for the leaderboard."""
        if task_id not in SCENARIOS:
            return 0.0
        total = len(SCENARIOS[task_id]["garbage_starts"])
        return min(max(self.inventory_count / total, 0.0), 1.0)