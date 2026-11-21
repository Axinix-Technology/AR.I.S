"""
AR.I.S Determinism & Sampling Discrepancy Investigation
Tests:
a. Run Seed 42 twice on the Step 16k checkpoint and diff per-sample outputs.
b. Inspect exact temperature, top_k, top_p in benchmark.py across stages.
c. Print per-sample generations for MBPP (sample 0), QA-SRL (sample 0),
   and high-entropy chat prompt across Seed 42, 43, 44, and Greedy.
d. Re-run GSM8K evaluation under temperature=0.2 vs temperature=0.7 to test if
   the 35.12 vs 28.53 discrepancy was caused by temperature parameter mismatch.
"""

import sys
import random
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aris.tokenizer.bpe_tokenizer import ArisTokenizer
from aris.training.trainer import ArisTrainer
from aris.evaluation.benchmark import ArisBenchmarkSuite
from aris.inference.generation import ArisInferenceEngine

DATA_DIR = Path("data")
MODELS_DIR = Path("models")
TOKENIZER_PATH = MODELS_DIR / "aris_bpe_tokenizer.json"
CKPT_ROOT = Path("aris_lm_checkpoint.pt")

def test_determinism():
    print("="*75)
    print("2a. DETERMINISM CHECK: RUN SEED 42 TWICE ON STEP 16,000 CHECKPOINT")
    print("="*75)
    device = torch.device("cpu")
    tok = ArisTokenizer.load(TOKENIZER_PATH)
    model, step = ArisTrainer.load_model(CKPT_ROOT, device=device)
    
    prompt = "Solve the following mathematical problem step by step:\nJanet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
    
    # Run 1
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    suite1 = ArisBenchmarkSuite(model=model, tokenizer=tok, device=device)
    out1 = suite1._generate(prompt, max_tokens=50, temperature=0.2)
    
    # Run 2
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    suite2 = ArisBenchmarkSuite(model=model, tokenizer=tok, device=device)
    out2 = suite2._generate(prompt, max_tokens=50, temperature=0.2)
    
    print(f"Run 1 (Seed 42): {repr(out1)}")
    print(f"Run 2 (Seed 42): {repr(out2)}")
    print(f"Are Run 1 and Run 2 bitwise identical? {out1 == out2}")

    print("\n" + "="*75)
    print("2b. BENCHMARK TEMPERATURE / TOP_K / TOP_P AUDIT")
    print("="*75)
    print("Inspection of aris/evaluation/benchmark.py:")
    print("  _generate() default: max_tokens=40, temperature=0.2, top_k=40, top_p=0.9")
    print("  Stage 1 (Lexical):   calls _generate(p, max_tokens=25)                   -> temp=0.2")
    print("  Stage 2 (Syntax):    calls _generate(p, max_tokens=65)                   -> temp=0.2")
    print("  Stage 3 (QA-SRL):    calls _generate(p, max_tokens=65)                   -> temp=0.2")
    print("  Stage 4 (HotpotQA):  calls _generate(p, max_tokens=65)                   -> temp=0.2")
    print("  Stage 5 (GSM8K):     calls _generate(p, max_tokens=50, temperature=0.2)  -> temp=0.2")
    print("  Stage 6 (MBPP):      calls _generate(p, max_tokens=50, temperature=0.2)  -> temp=0.2")
    print("  Stage 7 (Voice):     calls _generate(p, max_tokens=64, temperature=0.7)  -> temp=0.7")
    print("RNG Path: The random seed reaches the sampler inside torch.multinomial(probs, num_samples=1)")
    print("via PyTorch's default CPU generator in ArisInferenceEngine.generate().")

    print("\n" + "="*75)
    print("2c. PER-SAMPLE GENERATIONS ACROSS 3 SEEDS + GREEDY")
    print("="*75)
    
    mbpp_prompt = "Write a Python program to solve this task:\nWrite a function to find the shared elements from the given two lists."
    qasrl_prompt = "Sentence: The FBI began investigating the case in 2008.\nTarget Action: 'investigating'\nQuestion: who investigated something?"
    chat_prompt = "What is the meaning of life and consciousness?"

    test_prompts = [
        ("MBPP Sample 0", mbpp_prompt, 50),
        ("QA-SRL Sample 0", qasrl_prompt, 50),
        ("High-Entropy Chat", chat_prompt, 50),
    ]

    seeds = [42, 43, 44]
    for p_label, p_text, max_tok in test_prompts:
        print(f"\n--- {p_label} ---")
        for s in seeds:
            torch.manual_seed(s)
            random.seed(s)
            np.random.seed(s)
            gen = suite1._generate(p_text, max_tokens=max_tok, temperature=0.2)
            print(f"Seed {s}: {repr(gen)}")
        # Greedy
        orig_gen = suite1._generate
        def greedy_gen(p, max_tokens=max_tok, **kwargs):
            chat = [{"role": "user", "content": p}]
            prompt_ids = tok.encode_chat(chat, add_generation_prompt=True)
            inp = torch.tensor([prompt_ids], dtype=torch.long)
            out = ArisInferenceEngine.generate(model, inp, max_new_tokens=max_tokens, temperature=0.0, eos_id=tok.im_end_id)
            return tok.decode(out[0][len(prompt_ids):].tolist(), skip_special_tokens=True).strip()
        print(f"Greedy:  {repr(greedy_gen(p_text, max_tokens=max_tok))}")

    print("\n" + "="*75)
    print("2d. DISCREPANCY ROOT CAUSE: GSM8K EVALUATION AT TEMP 0.2 VS TEMP 0.7")
    print("="*75)
    # Re-evaluate GSM8K on Step 16k with temp=0.2 vs temp=0.7 for Seed 42
    import pyarrow.parquet as pq
    from aris.evaluation.metrics import compute_token_f1
    
    table = pq.read_table("data/gsm8k/train.parquet")
    rows = table.to_pylist()[:10]
    
    for t_val in [0.2, 0.7]:
        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)
        f1s = []
        for r in rows:
            q = r.get("question", "").strip()
            ans = r.get("answer", "").strip()
            prompt = f"Solve the following mathematical problem step by step:\n{q}"
            gen = suite1._generate(prompt, max_tokens=50, temperature=t_val)
            f1s.append(compute_token_f1(gen, ans))
        avg_f1 = (sum(f1s) / len(f1s)) * 100.0
        print(f"Seed 42 GSM8K Token F1 at temperature={t_val:.1f}: {avg_f1:.2f}%")

if __name__ == "__main__":
    test_determinism()
