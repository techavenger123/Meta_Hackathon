---
title: Incident Response Triage
emoji: 🚨
colorFrom: red
colorTo: orange
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
---

# 🚨 Incident Response Triage — OpenEnv

An OpenEnv-compliant reinforcement learning environment simulating **production incident response and triage**. AI agents act as Site Reliability Engineers (SREs) who must diagnose and resolve real-world infrastructure incidents across a simulated microservices architecture.

## Why Incident Response?

Incident response costs the tech industry **$70B+ annually** in downtime. SREs spend countless hours triaging alerts, correlating logs, and remediating failures. This environment provides a realistic training ground for AI agents to learn:

- **Alert triage** — prioritize and correlate production alerts
- **Root cause analysis** — trace through service dependencies to find the source
- **Remediation** — apply the correct fix (restart, rollback, scale, etc.)
- **Efficient resolution** — minimize time-to-resolution under pressure

---

## Architecture

The environment simulates a **10-service microservices system**:

```
┌──────────────┐
│ web-frontend │
└──────┬───────┘
       ▼
┌──────────────┐
│ api-gateway  │──────┬──────┬──────┬──────────────┐
└──────────────┘      ▼      ▼      ▼              ▼
              ┌──────────┐ ┌────────────┐ ┌──────────────────┐ ┌───────┐
              │auth-svc  │ │user-svc    │ │order-svc         │ │notif  │
              └────┬─────┘ └─────┬──────┘ └──┬────┬─────┬────┘ └───┬───┘
                   ▼             ▼            ▼    ▼     ▼          ▼
              ┌──────────┐  ┌───────┐  ┌──────────┐ ┌─────────┐ ┌───────┐
              │ database │  │ cache │  │payment   │ │ msg-q   │ │ cache │
              └──────────┘  └───────┘  └──────────┘ └─────────┘ └───────┘
```

---

## Tasks

| Task ID | Difficulty | Description | Max Steps |
|---------|-----------|-------------|-----------|
| `task_easy` | 🟢 Easy | Memory leak in user-service. Clear symptoms, straightforward fix. | 15 |
| `task_medium` | 🟡 Medium | Database connection pool exhaustion causing cascading timeouts. Multiple services affected. | 20 |
| `task_hard` | 🔴 Hard | Bad deployment of payment-service causes cascading failures with misleading symptoms and red herrings. | 25 |

---

## Action Space

Text commands sent as strings:

| Command | Description |
|---------|-------------|
| `INVESTIGATE <service>` | View recent logs for a service |
| `CHECK_METRICS <service>` | View CPU, memory, latency, error rate, etc. |
| `CHECK_DEPENDENCIES <service>` | View upstream/downstream dependencies |
| `RESTART <service>` | Restart a service |
| `SCALE <service> <count>` | Scale service instances |
| `ROLLBACK <service>` | Roll back to previous deployment version |
| `FLUSH_CACHE` | Clear the Redis cache |
| `INCREASE_CONNECTIONS <service>` | Increase database connection pool |
| `DIAGNOSE <root_cause>` | Submit your root cause diagnosis |
| `RESOLVE <summary>` | Mark incident as resolved |

**Services:** `web-frontend`, `api-gateway`, `auth-service`, `user-service`, `order-service`, `payment-service`, `notification-service`, `database`, `cache`, `message-queue`

---

## Observation Space

```json
{
  "alerts": [
    {"id": "A1", "severity": "critical", "service": "user-service",
     "message": "Memory usage critical: 92%", "timestamp": "10:24:00"}
  ],
  "services": {
    "user-service": {
      "cpu_percent": 45.0, "memory_percent": 92.0,
      "request_latency_ms": 850, "error_rate_percent": 12.0,
      "active_connections": 156, "status": "degraded"
    }
  },
  "action_result": "Logs for user-service showing memory leak...",
  "step_number": 3,
  "steps_remaining": 12,
  "incident_resolved": false,
  "task_id": "task_easy",
  "done": false
}
```

---

## Reward Function

| Event | Reward |
|-------|--------|
| Each step | `-0.01` (time pressure) |
| Investigate root cause service | `+0.10` |
| Investigate context/symptoms | `+0.05` |
| Correct root cause diagnosis | `+0.20 to +0.25` |
| Correct primary remediation | `+0.25 to +0.35` |
| Correct secondary remediation (hard) | `+0.10` |
| Successful resolution | `+0.25` |
| Wrong remediation action | `-0.05` |
| Invalid command | `-0.03` |

**Total possible: ~1.0** per task (minus step penalties). Score normalized to **0.0 – 1.0**.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/reset` | Reset environment `{"task_id": "task_easy"}` |
| `POST` | `/step` | Execute action `{"command": "INVESTIGATE user-service"}` |
| `GET` | `/state` | Get current state with milestones |
| `GET` | `/tasks` | List all tasks |
| `GET` | `/grade/{task_id}` | Run grader for a task |

---

## Setup & Running

### Local

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the server
uvicorn app:app --host 0.0.0.0 --port 7860 --reload

# 3. Set environment variables
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export HF_TOKEN=your_api_key
export ENV_URL=http://localhost:7860

# 4. Run inference
python inference.py
```

### Docker

```bash
docker build -t incident-response-triage .
docker run -p 7860:7860 incident-response-triage
```

Then in another terminal:
```bash
export ENV_URL=http://localhost:7860
python inference.py
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `API_BASE_URL` | LLM API endpoint (e.g., `https://api.openai.com/v1`) |
| `MODEL_NAME` | Model identifier (e.g., `gpt-4o-mini`) |
| `HF_TOKEN` | Hugging Face / API key |
| `ENV_URL` | URL where the env server is running |

---

## Baseline Scores

| Task | Greedy Agent | Expected LLM Agent |
|------|-------------|-------------------|
| `task_easy` | ~0.94 | 0.70 – 0.95 |
| `task_medium` | ~0.92 | 0.50 – 0.85 |
| `task_hard` | ~0.90 | 0.30 – 0.75 |

---

## Project Structure

```
├── app.py              # FastAPI server (OpenEnv endpoints)
├── environment.py      # Core environment logic (reset/step/state/grade)
├── models.py           # Pydantic typed models
├── scenarios.py        # Incident scenario definitions
├── inference.py        # Baseline inference script
├── openenv.yaml        # OpenEnv specification
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container definition
└── README.md           # This file
```
# Meta_Hackathon
# Meta_Hackathon
# Meta_Hackathon
