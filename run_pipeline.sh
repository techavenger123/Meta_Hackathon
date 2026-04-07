#!/bin/bash

# Activate the existing virtual environment where dependencies are being installed
source venv/bin/activate

echo "Ensuring pip dependencies are installed and PyTorch is active..."
# Install remaining dependencies. Pip will use lock files to wait or pass if already installed by the background process.
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" datasets trl peft transformers bitsandbytes --no-cache-dir

echo "======================================"
echo " Starting Unsloth LoRA Fine-Tuning... "
echo "======================================"

# Run the python script and route errors to standard out
python train_unsloth.py > train_output.log 2>&1

echo "Process completed. Check train_output.log for details."
