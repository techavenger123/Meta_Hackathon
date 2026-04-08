from typing import Tuple, List, Dict, Any

SCENARIOS: Dict[str, Dict[str, Any]] = {
    "task_easy": {
        "grid_size": (5, 5),
        "robot_start": (0, 0),
        "garbage_starts": [(4, 4)],
        "obstacle_starts": [],
        "max_battery": 30,
        # ── Resource management ────────────────────────────────
        # Home (charging station) is the robot's spawn point.
        "home_position": (0, 0),
        # Unload corner is the cell diagonally opposite to home.
        "unload_station": (4, 0),
        # 1 garbage piece; capacity=1 forces an unload cycle before finishing,
        # demonstrating the mechanic even on the simplest task.
        "storage_capacity": 6,
    },
    "task_medium": {
        "grid_size": (7, 7),
        "robot_start": (3, 3),
        "garbage_starts": [(1, 1), (5, 5), (1, 5)],
        "obstacle_starts": [(2, 2), (2, 3), (2, 4), (4, 2), (4, 3), (4, 4)],
        "max_battery": 50,
        # ── Resource management ────────────────────────────────
        "home_position": (3, 3),
        # Far corner from centre home — no obstacles there.
        "unload_station": (6, 0),
        # Capacity 2 out of 3 garbage pieces forces exactly one unload cycle.
        "storage_capacity": 6,
    },
    "task_hard": {
        "grid_size": (10, 10),
        "robot_start": (0, 0),
        "garbage_starts": [(8, 8), (9, 1), (1, 9), (5, 5), (8, 2)],
        "obstacle_starts": [
            (1, 1), (1, 2), (1, 3), (1, 4),
            (3, 1), (3, 2), (3, 3), (3, 4),
            (6, 5), (6, 6), (6, 7), (6, 8),   # shifted so (5,5) stays clear for garbage
            (7, 7), (7, 8), (7, 9),
        ],
        "max_battery": 80,
        # ── Resource management ────────────────────────────────
        "home_position": (0, 0),
        # Bottom-right corner — clear of all obstacles.
        "unload_station": (9, 0),
        # Capacity 2 out of 5 garbage pieces → two unload cycles required.
        "storage_capacity": 6,
    },
}