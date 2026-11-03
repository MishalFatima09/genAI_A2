#!/usr/bin/env python3
"""
Kaggle/Colab compatible script for Task 1 fine-tuning.
Run as: python kaggle_finetune.py
"""

import os
import subprocess
import sys

# 1. Install deps
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "accelerate", "peft", "transformers", "datasets", "scikit-learn", "pandas", "numpy", "tqdm"], check=True)

# 2. Clone repo (EDIT URL)
REPO_URL = "https://github.com/YOUR_USERNAME/YOUR_REPO.git"
if not os.path.exists("genAI_A2"):
    subprocess.run(["git", "clone", REPO_URL, "genAI_A2"], check=True)
os.chdir("genAI_A2")

# 3. Prepare data
subprocess.run([sys.executable, "code/prepare_data.py", "--max-prompt-tokens", "2048"], check=True)

# 4. Fine-tune
subprocess.run([
    sys.executable, "code/finetune_lora.py",
    "--base-model", "Qwen/Qwen2.5-1.5B-Instruct",
    "--epochs", "3",
    "--batch-size", "4",
    "--grad-accum", "4",
    "--lr", "2e-4",
    "--r", "16",
    "--alpha", "32",
    "--dropout", "0.05",
    "--max-length", "2048",
    "--warmup-ratio", "0.03",
    "--weight-decay", "0.01",
    "--dtype", "auto",
    "--grad-checkpoint",
    "--out", "adapters/qwen25-1.5b-lora"
], check=True)

# 5. Before/after eval on val
subprocess.run([
    sys.executable, "code/evaluate.py",
    "--split", "val",
    "--compare", "adapters/qwen25-1.5b-lora",
    "--batch-size", "8",
    "--max-new-tokens", "64"
], check=True)

print("✅ Task 1 complete. Adapter in adapters/qwen25-1.5b-lora/")
print("📥 Download the adapter folder from Kaggle output/Colab files.")