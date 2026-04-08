"""
code.py — Seed training data generator for the Garbage Collecting Robot.

Fix applied:
  - All trajectory entries now use the unified {"text": "..."} Alpaca format.
  - Previously the first entry used {"text": ...} while all others used
    {"obs": ..., "action": ...}, causing fixer.py to silently skip them
    (KeyError on the missing "text" key).
"""

import json

INSTRUCTION = (
    "You control a garbage collecting robot. "
    "Reply with ONE of: UP DOWN LEFT RIGHT COLLECT"
)

def alpaca(obs: str, action: str) -> dict:
    """Wrap an obs/action pair into the Alpaca fine-tuning format."""
    return {
        "text": (
            f"### Instruction:\n{INSTRUCTION}\n\n"
            f"### Input:\nENVIRONMENT STATUS:\n{obs}\n\n"
            f"### Response:\n{action}"
        )
    }


trajectories = [
    # --- task_easy: straight-line approach ---
    alpaca("You are at (0, 0). Garbage at [(4, 4)]. Battery: 30/30. No obstacles nearby.", "RIGHT"),
    alpaca("You are at (1, 0). Garbage at [(4, 4)]. Battery: 29/30. No obstacles nearby.", "RIGHT"),
    alpaca("You are at (2, 0). Garbage at [(4, 4)]. Battery: 28/30. No obstacles nearby.", "RIGHT"),
    alpaca("You are at (3, 0). Garbage at [(4, 4)]. Battery: 27/30. No obstacles nearby.", "RIGHT"),
    alpaca("You are at (4, 0). Garbage at [(4, 4)]. Battery: 26/30. No obstacles nearby.", "UP"),
    alpaca("You are at (4, 1). Garbage at [(4, 4)]. Battery: 25/30. No obstacles nearby.", "UP"),
    alpaca("You are at (4, 2). Garbage at [(4, 4)]. Battery: 24/30. No obstacles nearby.", "UP"),
    alpaca("You are at (4, 3). Garbage at [(4, 4)]. Battery: 23/30. No obstacles nearby.", "UP"),
    alpaca("You are at (4, 4). Garbage at [(4, 4)]. Battery: 22/30. You are ON the garbage.", "COLLECT"),

    # --- task_medium: obstacle avoidance ---
    alpaca("You are at (3, 3). Garbage at [(1,1),(5,5),(1,5)]. Battery: 50/50. BLOCKED! DOWN is an obstacle. Blocked directions: DOWN, LEFT. Choose a different direction.", "UP"),
    alpaca("You are at (3, 4). Garbage at [(1,1),(5,5),(1,5)]. Battery: 49/50. Moving toward (1,5).", "LEFT"),
    alpaca("You are at (2, 4). Garbage at [(1,1),(5,5),(1,5)]. Battery: 48/50. BLOCKED! LEFT is an obstacle. Blocked directions: LEFT. Choose RIGHT or UP.", "UP"),
    alpaca("You are at (2, 5). Garbage at [(1,1),(5,5),(1,5)]. Battery: 47/50. Clear path left.", "LEFT"),
    alpaca("You are at (1, 5). Garbage at [(1,1),(5,5),(1,5)]. Battery: 46/50. You are ON the garbage.", "COLLECT"),
    alpaca("You are at (1, 5). Garbage at [(1,1),(5,5)]. Battery: 45/50. Next target (5,5), moving right.", "RIGHT"),
    alpaca("You are at (2, 5). Garbage at [(1,1),(5,5)]. Battery: 44/50. Continuing right.", "RIGHT"),
    alpaca("You are at (3, 5). Garbage at [(1,1),(5,5)]. Battery: 43/50. Continuing right.", "RIGHT"),
    alpaca("You are at (4, 5). Garbage at [(1,1),(5,5)]. Battery: 42/50. Continuing right.", "RIGHT"),
    alpaca("You are at (5, 5). Garbage at [(1,1),(5,5)]. Battery: 41/50. You are ON the garbage.", "COLLECT"),
    alpaca("You are at (5, 5). Garbage at [(1,1)]. Battery: 40/50. Last garbage at (1,1), heading left+down.", "LEFT"),
    alpaca("You are at (4, 5). Garbage at [(1,1)]. Battery: 39/50. Continuing toward (1,1).", "LEFT"),
    alpaca("You are at (3, 5). Garbage at [(1,1)]. Battery: 38/50. BLOCKED! DOWN is an obstacle. Go LEFT.", "LEFT"),
    alpaca("You are at (2, 5). Garbage at [(1,1)]. Battery: 37/50. BLOCKED! DOWN is an obstacle. Go LEFT.", "LEFT"),
    alpaca("You are at (1, 5). Garbage at [(1,1)]. Battery: 36/50. Path down is clear now.", "DOWN"),
    alpaca("You are at (1, 4). Garbage at [(1,1)]. Battery: 35/50. Continuing down.", "DOWN"),
    alpaca("You are at (1, 3). Garbage at [(1,1)]. Battery: 34/50. Continuing down.", "DOWN"),
    alpaca("You are at (1, 2). Garbage at [(1,1)]. Battery: 33/50. Continuing down.", "DOWN"),
    alpaca("You are at (1, 1). Garbage at [(1,1)]. Battery: 32/50. You are ON the last garbage.", "COLLECT"),

    # --- low battery urgency ---
    alpaca("You are at (2, 2). Garbage at [(4,4)]. Battery: 5/30. CRITICAL battery! Move directly: RIGHT.", "RIGHT"),
    alpaca("You are at (3, 2). Garbage at [(4,4)]. Battery: 4/30. CRITICAL battery! Move directly: RIGHT.", "RIGHT"),
    alpaca("You are at (4, 2). Garbage at [(4,4)]. Battery: 3/30. CRITICAL battery! Move directly: UP.", "UP"),
    alpaca("You are at (4, 3). Garbage at [(4,4)]. Battery: 2/30. CRITICAL battery! Move directly: UP.", "UP"),
    alpaca("You are at (4, 4). Garbage at [(4,4)]. Battery: 1/30. You are ON the garbage. COLLECT NOW.", "COLLECT"),

    # --- do not collect when not on garbage ---
    alpaca("You are at (2, 3). Garbage at [(4,4)]. Battery: 20/30. You are NOT on garbage. Move toward it.", "RIGHT"),
    alpaca("You are at (0, 0). Garbage at [(3,3)]. Battery: 15/30. You are NOT on garbage. Do not COLLECT.", "RIGHT"),
]

with open("garbage_robot_dataset.jsonl", "w") as f:
    for row in trajectories:
        f.write(json.dumps(row) + "\n")

print(f"Wrote {len(trajectories)} samples to garbage_robot_dataset.jsonl")