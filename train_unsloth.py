"""
Fine-tuning Llama-3 with Unsloth for the Garbage Collecting Robot OpenEnv Project.
"""

import os
import json
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
#try:
    
#except ImportError:
#    print("Unsloth, TRL, or Transformers not installed. Please install unsloth via their official documentation.")

max_seq_length = 2048
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage

def get_dummy_rl_dataset():
    """
    Synthetic dataset of successful RL trajectories derived from all three
    scenarios. Covers: straight navigation, obstacle avoidance, multi-target
    collection, and low-battery urgency. This gives the model patterns
    to generalise from rather than memorise 2 steps.
    """
    trajectories = [
        # --- task_easy: straight path to (4,4) ---
        {"obs": "You are at (0, 0). Garbage at [(4, 4)]. Battery: 30/30. No obstacles nearby.", "action": "RIGHT"},
        {"obs": "You are at (1, 0). Garbage at [(4, 4)]. Battery: 29/30. No obstacles nearby.", "action": "RIGHT"},
        {"obs": "You are at (2, 0). Garbage at [(4, 4)]. Battery: 28/30. No obstacles nearby.", "action": "RIGHT"},
        {"obs": "You are at (3, 0). Garbage at [(4, 4)]. Battery: 27/30. No obstacles nearby.", "action": "RIGHT"},
        {"obs": "You are at (4, 0). Garbage at [(4, 4)]. Battery: 26/30. No obstacles nearby.", "action": "UP"},
        {"obs": "You are at (4, 1). Garbage at [(4, 4)]. Battery: 25/30. No obstacles nearby.", "action": "UP"},
        {"obs": "You are at (4, 2). Garbage at [(4, 4)]. Battery: 24/30. No obstacles nearby.", "action": "UP"},
        {"obs": "You are at (4, 3). Garbage at [(4, 4)]. Battery: 23/30. No obstacles nearby.", "action": "UP"},
        {"obs": "You are at (4, 4). Garbage at [(4, 4)]. Battery: 22/30. You are ON the garbage.", "action": "COLLECT"},

        # --- task_medium: obstacle avoidance ---
        {"obs": "You are at (3, 3). Garbage at [(1,1),(5,5),(1,5)]. Battery: 50/50. BLOCKED! DOWN is an obstacle. Blocked directions: DOWN, LEFT. Choose a different direction.", "action": "UP"},
        {"obs": "You are at (3, 4). Garbage at [(1,1),(5,5),(1,5)]. Battery: 49/50. Moving toward (1,5).", "action": "LEFT"},
        {"obs": "You are at (2, 4). Garbage at [(1,1),(5,5),(1,5)]. Battery: 48/50. BLOCKED! LEFT is an obstacle. Blocked directions: LEFT. Choose RIGHT or UP.", "action": "UP"},
        {"obs": "You are at (2, 5). Garbage at [(1,1),(5,5),(1,5)]. Battery: 47/50. Clear path left.", "action": "LEFT"},
        {"obs": "You are at (1, 5). Garbage at [(1,1),(5,5),(1,5)]. Battery: 46/50. You are ON the garbage.", "action": "COLLECT"},
        {"obs": "You are at (1, 5). Garbage at [(1,1),(5,5)]. Battery: 45/50. Next target (5,5), moving right.", "action": "RIGHT"},
        {"obs": "You are at (2, 5). Garbage at [(1,1),(5,5)]. Battery: 44/50. Continuing right.", "action": "RIGHT"},
        {"obs": "You are at (3, 5). Garbage at [(1,1),(5,5)]. Battery: 43/50. Continuing right.", "action": "RIGHT"},
        {"obs": "You are at (4, 5). Garbage at [(1,1),(5,5)]. Battery: 42/50. Continuing right.", "action": "RIGHT"},
        {"obs": "You are at (5, 5). Garbage at [(1,1),(5,5)]. Battery: 41/50. You are ON the garbage.", "action": "COLLECT"},
        {"obs": "You are at (5, 5). Garbage at [(1,1)]. Battery: 40/50. Last garbage at (1,1), heading left+down.", "action": "LEFT"},
        {"obs": "You are at (4, 5). Garbage at [(1,1)]. Battery: 39/50. Continuing toward (1,1).", "action": "LEFT"},
        {"obs": "You are at (3, 5). Garbage at [(1,1)]. Battery: 38/50. BLOCKED! DOWN is an obstacle. Go LEFT.", "action": "LEFT"},
        {"obs": "You are at (2, 5). Garbage at [(1,1)]. Battery: 37/50. BLOCKED! DOWN is an obstacle. Go LEFT.", "action": "LEFT"},
        {"obs": "You are at (1, 5). Garbage at [(1,1)]. Battery: 36/50. Path down is clear now.", "action": "DOWN"},
        {"obs": "You are at (1, 4). Garbage at [(1,1)]. Battery: 35/50. Continuing down.", "action": "DOWN"},
        {"obs": "You are at (1, 3). Garbage at [(1,1)]. Battery: 34/50. Continuing down.", "action": "DOWN"},
        {"obs": "You are at (1, 2). Garbage at [(1,1)]. Battery: 33/50. Continuing down.", "action": "DOWN"},
        {"obs": "You are at (1, 1). Garbage at [(1,1)]. Battery: 32/50. You are ON the last garbage.", "action": "COLLECT"},

        # --- low battery urgency ---
        {"obs": "You are at (2, 2). Garbage at [(4,4)]. Battery: 5/30. CRITICAL battery! Move directly: RIGHT.", "action": "RIGHT"},
        {"obs": "You are at (3, 2). Garbage at [(4,4)]. Battery: 4/30. CRITICAL battery! Move directly: RIGHT.", "action": "RIGHT"},
        {"obs": "You are at (4, 2). Garbage at [(4,4)]. Battery: 3/30. CRITICAL battery! Move directly: UP.", "action": "UP"},
        {"obs": "You are at (4, 3). Garbage at [(4,4)]. Battery: 2/30. CRITICAL battery! Move directly: UP.", "action": "UP"},
        {"obs": "You are at (4, 4). Garbage at [(4,4)]. Battery: 1/30. You are ON the garbage. COLLECT NOW.", "action": "COLLECT"},

        # --- do not collect when not on garbage ---
        {"obs": "You are at (2, 3). Garbage at [(4,4)]. Battery: 20/30. You are NOT on garbage. Move toward it.", "action": "RIGHT"},
        {"obs": "You are at (0, 0). Garbage at [(3,3)]. Battery: 15/30. You are NOT on garbage. Do not COLLECT.", "action": "RIGHT"},
    ]

    instruction = (
        "You are an AI brain controlling a simulated garbage collecting robot.\n"
        "Your goal is to navigate a grid room and collect all pieces of garbage before running out of battery.\n"
        "Your available actions are EXACTLY ONE of the following keywords:\n"
        "UP\nDOWN\nLEFT\nRIGHT\nCOLLECT\n\n"
        "Rules:\n"
        "- If you are standing exactly on a garbage tile, execute COLLECT.\n"
        "- If a direction is BLOCKED by an obstacle, do NOT choose it.\n"
        "- Move toward the nearest garbage using the shortest unobstructed path.\n"
        "- When battery is critically low (<=5), prioritise the closest garbage only.\n"
        "- Never COLLECT when you are not on a garbage tile."
    )

    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
ENVIRONMENT STATUS:
{}

### Response:
{}"""

    formatted_data = []
    for step in trajectories:
        text = alpaca_prompt.format(instruction, step["obs"], step["action"])
        formatted_data.append({"text": text})

    return Dataset.from_list(formatted_data)

def main():
    print("Loading internal Unsloth libraries and Llama-3-8B-Instruct...")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/llama-3-8b-Instruct-bnb-4bit",
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            # token=os.environ.get("HF_TOKEN")
        )
    except NameError:
        print("Skipping actual load since dependencies might be missing in this environment.")
        return

    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    dataset = get_dummy_rl_dataset()

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False, # Can make training 5x faster for short sequences
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=100,  # increased to match richer dataset (was 30)
            learning_rate=3e-4, # Increased for quicker convergence
            fp16=not FastLanguageModel.is_bfloat16_supported(),
            bf16=FastLanguageModel.is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
        ),
    )

    print("Starting fine-tuning process...")
    trainer_stats = trainer.train()

    print("Saving the adapter model...")
    model.save_pretrained("lora_garbage_robot")
    tokenizer.save_pretrained("lora_garbage_robot")

    print("\nTraining complete! You can now load this model in Unsloth Studio or via HuggingFace for inference.")

if __name__ == "__main__":
    main()