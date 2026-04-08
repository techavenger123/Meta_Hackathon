from qlearning import QTable, ACTIONS, encode_state
from environment import GarbageRobotEnv
from scenarios import SCENARIOS
import json

qt = QTable()
qt.load('qtable.json')
env = GarbageRobotEnv()

instruction = '''You are an AI brain controlling a garbage collecting robot.
Reply with EXACTLY ONE of: UP DOWN LEFT RIGHT COLLECT'''

alpaca = '''### Instruction:\n{}\n\n### Input:\nENVIRONMENT STATUS:\n{}\n\n### Response:\n{}'''

data = []
for task_id in SCENARIOS:
    for _ in range(10):  # 10 episodes per task
        env.reset(task_id)
        done = False
        while not done:
            obs_obj = env.get_observation()
            obs = {'robot_position': obs_obj.robot_position,
                   'garbage_positions': list(obs_obj.garbage_positions),
                   'grid_size': obs_obj.grid_size}
            state = encode_state(obs)
            action = ACTIONS[qt.best_action(state)]
            data.append({'text': alpaca.format(instruction, obs_obj.message, action)})
            result = env.step(action)
            done = result['done']

with open('rl_trajectories.jsonl', 'w') as f:
    for row in data:
        f.write(json.dumps(row) + '\n')
print(f'Generated {len(data)} samples')