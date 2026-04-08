---
title: GarbageBot — RL Control Center
emoji: 🗑️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
  - robotics
  - reinforcement-learning
  - llama-3.2
---

> [!NOTE]
> This Space uses fine-tuned model weights hosted at [TechAvenger/GarbageBot-Weights](https://huggingface.co/TechAvenger/GarbageBot-Weights). The model is loaded in 4-bit precision for efficiency.

# 🤖 Garbage Collecting Robot — OpenEnv

An OpenEnv-compliant reinforcement learning environment for a garbage collecting robot. The agent must navigate a grid room to pick up garbage while managing battery constraints and storage capacity.

## Why Garbage Collection?

Autonomous garbage collection is a classic robotics challenge involving pathfinding, resource management (battery), and state management (storage capacity). This environment provides a realistic training ground for AI agents to learn:
- **Optimal Navigation** — shortest paths via BFS and Q-Learning.
- **Resource Management** — returning to base for charging before battery depletion.
- **Logistics** — managing a 6-unit storage bin and prioritizing unload cycles.

---

## Architecture

The environment is a discrete grid world where the robot interacts with garbage, obstacles, a charging station (Home), and an Unload Station.

```
┌──────────┐
│ Dashboard│ (FastAPI + Vanilla JS)
└─────┬────┘
      ▼
┌──────────┐
│ API      │ (app.py)
└─────┬────┘
      ▼
┌──────────┐
│ Env Logic│ (environment.py)
└──────────┘
```

---

## Tasks

| Task ID | Difficulty | Description | Grid Size |
|---------|-----------|-------------|-----------|
| `task_easy` | 🟢 Easy | Small 5x5 grid, 1 piece of garbage. | 5x5 |
| `task_medium` | 🟡 Medium | 7x7 grid with obstacles, 3 pieces of garbage. | 7x7 |
| `task_hard` | 🔴 Hard | 10x10 maze, 5 pieces of garbage, strict battery. | 10x10 |

---

## Action Space

Movement and interaction commands:
- `UP`, `DOWN`, `LEFT`, `RIGHT`: Move the robot one cell.
- `COLLECT`: Pick up garbage if the robot is on its cell.

---

## Observation Space

The environment returns a detailed state:
- `robot_position`: `(x, y)`
- `garbage_positions`: List of `(x, y)`
- `battery_level`: Current battery vs max.
- `current_storage_load`: Current items vs capacity (6).
- `robot_mode`: `normal`, `recharging`, or `unloading`.

---

## Policy Priority Chain

Decisions can be driven by:
1. **Q-Learning Table** — pre-trained optimal policy.
2. **Llama-3.2-3B-Instruct** — fine-tuned LLM policy.
3. **BFS Heuristic** — reliable fallback pathfinding.

---

## Local Development

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the server
uvicorn app:app --host 0.0.0.0 --port 7860

# 3. Training
python qlearning.py --train --episodes 10000
```

---

## Project Structure

```
├── app.py              # FastAPI server
├── environment.py      # Core RL logic
├── models.py           # Data schemas
├── scenarios.py        # Task definitions
├── qlearning.py        # Tabular RL training
├── inference.py        # Policy resolver
├── frontend/           # Dashboard HTML/CSS/JS
├── qtable.json         # Trained policy weights
├── Dockerfile          # Deployment container
└── README.md           # This file
```
