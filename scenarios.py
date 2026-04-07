from typing import Tuple, List, Dict, Any

SCENARIOS: Dict[str, Dict[str, Any]] = {
    "task_easy": {
        "grid_size": (5, 5),
        "robot_start": (0, 0),
        "garbage_starts": [(4, 4)],
        "obstacle_starts": [],
        "max_battery": 30,
    },
    "task_medium": {
        "grid_size": (7, 7),
        "robot_start": (3, 3),
        "garbage_starts": [(1, 1), (5, 5), (1, 5)],
        "obstacle_starts": [(2, 2), (2, 3), (2, 4), (4, 2), (4, 3), (4, 4)],
        "max_battery": 50,
    },
    "task_hard": {
        "grid_size": (10, 10),
        "robot_start": (0, 0),
        "garbage_starts": [(8, 8), (9, 1), (1, 9), (5, 5), (8, 2)],
        "obstacle_starts": [
            (1, 1), (1, 2), (1, 3), (1, 4),
            (3, 1), (3, 2), (3, 3), (3, 4),
            (5, 5), (5, 6), (5, 7), (5, 8),
            (7, 7), (7, 8), (7, 9)
        ],
        "max_battery": 80,
    }
}
