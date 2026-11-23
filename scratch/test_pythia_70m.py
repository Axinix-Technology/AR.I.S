"""
Test EleutherAI/pythia-70m (~70M Parameters)
Evaluates baseline generation and assesses 70M capacity.
"""

import os
import sys
import torch
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# Patch transformers multi-threaded tensor loading for Windows Python 3.15 stability
import transformers.core_model_loading as cml
_orig_spawn = cml.spawn_materialize
cml.spawn_materialize = lambda tp, *a, **kw: _orig_spawn(None, *a, **kw)

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "EleutherAI/pythia-70m"


def run_pythia_assessment():
    print("=" * 75)
    print(f"LOADING PRETRAINED BENCHMARK MODEL: {MODEL_ID}")
    print("Architecture: 70.4M Parameters (6 layers, d_model=512, 8 heads)")
    print("Pretraining:  300 Billion Tokens on the Pile")
    print("=" * 75)

    print("\nLoading tokenizer and weights from local Hugging Face cache...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, local_files_only=True)
    model.eval()
    print("Model loaded successfully into memory!\n")

    test_prompts = [
        "Hello! Who are you?",
        "Explain what a database transaction is in simple terms.",
        "Calculate: 25 + 17 = ",
        "Write a Python function to check if a word is a palindrome.",
    ]

    print("=" * 75)
    print("GENERATING RESPONSES FROM 70M MODEL (Next-Token Prediction)")
    print("=" * 75)

    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=60,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.15,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
            )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n--- PROMPT: {prompt} ---")
        print(f"{generated_text.strip()}\n")

    print("=" * 75)
    print("ASSESSMENT COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    run_pythia_assessment()
