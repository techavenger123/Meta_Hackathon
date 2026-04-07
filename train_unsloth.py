"""
Fine-tuning Llama-3 with Unsloth for the Garbage Collecting Robot OpenEnv Project.
"""

import os
import json

try:
    from datasets import Dataset
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments
except ImportError:
    print("Unsloth, TRL, or Transformers not installed. Please install unsloth via their official documentation.")

max_seq_length = 2048
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage

def get_dummy_rl_dataset():
    """
    Creates a small synthetic dataset representing offline RL trajectories
    where the robot successfully navigates and collects garbage.
    """
    trajectories = [
        {
            "instruction": "You are an AI brain controlling a simulated garbage collecting robot.\nYour goal is to navigate a grid room and collect all pieces of garbage before running out of battery.\nYour available actions are EXACTLY ONE of the following keywords:\nUP\nDOWN\nLEFT\nRIGHT\nCOLLECT\n\nIf you are standing on top of garbage, execute COLLECT.\nOtherwise, move towards the garbage by matching your X/Y coordinates.",
            "observation": "You are at (0, 0). Garbage remaining: 1. Garbage at (0, 1). Battery: 30/30.",
            "action": "UP"
        },
        {
            "instruction": "You are an AI brain controlling a simulated garbage collecting robot.\nYour goal is to navigate a grid room and collect all pieces of garbage before running out of battery.\nYour available actions are EXACTLY ONE of the following keywords:\nUP\nDOWN\nLEFT\nRIGHT\nCOLLECT\n\nIf you are standing on top of garbage, execute COLLECT.\nOtherwise, move towards the garbage by matching your X/Y coordinates.",
            "observation": "You are at (0, 1). Garbage remaining: 1. Garbage at (0, 1). Battery: 29/30.",
            "action": "COLLECT"
        }
    ]
    
    # Alpaca style formatting
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
        text = alpaca_prompt.format(step["instruction"], step["observation"], step["action"])
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
            max_steps=30, # Increased for demonstration
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
