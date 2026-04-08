import json

input_file = "rl_trajectories.jsonl"
output_file = "fixed_dataset.jsonl"

def extract_parts(text):
    try:
        user_part = text.split("### Response:")[0].strip()
        assistant_part = text.split("### Response:")[1].strip()
        return user_part, assistant_part
    except:
        return None, None

with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
    for line in f_in:
        data = json.loads(line)
        text = data.get("text", "")
        
        user, assistant = extract_parts(text)
        
        if user and assistant:
            new_entry = {
                "user": user,
                "assistant": assistant
            }
            f_out.write(json.dumps(new_entry) + "\n")

print("Done. Fixed dataset saved.")