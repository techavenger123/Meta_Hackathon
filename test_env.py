import requests

ENV_URL = "http://localhost:7860"

print("Resetting task_easy...")
res = requests.post(f"{ENV_URL}/reset", json={"task_id": "task_easy"})
print("Observation:", res.json()["observation"])

print("\nStepping UP...")
res = requests.post(f"{ENV_URL}/step", json={"command": "UP"})
print("Result:", res.json())

print("\nStepping UP...")
res = requests.post(f"{ENV_URL}/step", json={"command": "UP"})
print("Result:", res.json())

print("\nGrading...")
res = requests.get(f"{ENV_URL}/grade/task_easy")
print("Grade:", res.json())
