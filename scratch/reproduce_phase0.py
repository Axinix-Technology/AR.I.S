import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import random
import numpy as np
import torch

from aris.tokenizer.bpe_tokenizer import ArisTokenizer
from aris.training.trainer import ArisTrainer
from aris.evaluation.benchmark import ArisBenchmarkSuite
from aris.config import TOKENIZER_PATH, LM_CHECKPOINT_PATH, BASE_DIR

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

def run_suite(model, tok, seed: int, max_samples: int, force_greedy: bool):
    set_all_seeds(seed)
    suite = ArisBenchmarkSuite(model=model, tokenizer=tok, device=torch.device("cpu"))
    
    if force_greedy:
        orig_gen = suite._generate
        def greedy_gen(prompt, max_tokens=40, temperature=0.0):
            return orig_gen(prompt, max_tokens=max_tokens, temperature=0.0)
        suite._generate = greedy_gen
        
    ud_path = list(BASE_DIR.glob("ud-treebanks-*.tgz"))[0]
    hp_path = list(BASE_DIR.glob("data/hotpotqa/**/*.parquet"))[0]
    qasrl_path = list(BASE_DIR.glob("data/qasrl_gs/**/*.jsonl.gz"))[0]
    gsm_path = list(BASE_DIR.glob("data/gsm8k/**/*.parquet"))[0]
    mbpp_path = list(BASE_DIR.glob("data/mbpp/**/*.parquet"))[0]

    return suite.run_full_evaluation(
        hotpotqa_path=hp_path,
        qasrl_path=qasrl_path,
        ud_archive_path=ud_path,
        gsm8k_path=gsm_path,
        mbpp_path=mbpp_path,
        max_samples=max_samples,
    )

def extract_summary(rep):
    s1 = rep["stage1"]
    s2 = rep["stage2"]
    s3 = rep["stage3"]
    s4 = rep["stage4"]
    s5 = rep["stage5"]
    s6 = rep["stage6"]
    s7 = rep["stage7"]
    return {
        "s1_validity": s1.get("word_validity_rate", 0.0),
        "s2_upos": s2.get("upos_accuracy", 0.0),
        "s3_span_f1": s3.get("span_f1", 0.0),
        "s4_token_f1": s4.get("token_f1", 0.0),
        "s4_em": s4.get("exact_match", 0.0),
        "s5_token_f1": s5.get("token_f1", 0.0),
        "s5_num_acc": s5.get("numeric_accuracy", 0.0),
        "s6_code_f1": s6.get("code_f1", 0.0),
        "s6_syntax": s6.get("syntax_validity", 0.0),
        "s7_probes": [(p["prompt"], p["response"][:80]) for p in s7],
    }

def main():
    print("Loading Canonical Checkpoint aris_lm_checkpoint.pt...")
    tok = ArisTokenizer.load(TOKENIZER_PATH)
    model, step = ArisTrainer.load_model(LM_CHECKPOINT_PATH, device=torch.device("cpu"))
    
    seeds = [42, 43, 44]
    sampled_runs = {}
    
    print("\n=======================================================")
    print("  RUNNING 3 SEEDS (SAMPLES=10, ORIGINAL HARNESS)")
    print("  LABEL: LEGACY METRICS (known-flawed harness: 50-token cap, substring numeric check)")
    print("=======================================================")
    for s in seeds:
        print(f"\n--- Running Seed {s} ---")
        rep = run_suite(model, tok, seed=s, max_samples=10, force_greedy=False)
        sampled_runs[s] = extract_summary(rep)
        
    print("\n=======================================================")
    print("  RUNNING GREEDY DECODING (SAMPLES=10, ORIGINAL HARNESS)")
    print("  LABEL: LEGACY METRICS (known-flawed harness: 50-token cap, substring numeric check)")
    print("=======================================================")
    rep_greedy = run_suite(model, tok, seed=42, max_samples=10, force_greedy=True)
    greedy_summary = extract_summary(rep_greedy)
    
    print("\n=======================================================")
    print("  RUNNING LARGER SAMPLE COUNT (SAMPLES=25, GREEDY DECODING)")
    print("  LABEL: LEGACY METRICS (known-flawed harness: 50-token cap, substring numeric check)")
    print("=======================================================")
    rep_large = run_suite(model, tok, seed=42, max_samples=25, force_greedy=True)
    large_summary = extract_summary(rep_large)

    print("\n=======================================================")
    print("  PHASE 0 REPRODUCTION RESULTS TABLE")
    print("=======================================================")
    metrics = [
        ("Stage 1 Word Validity (%)", "s1_validity"),
        ("Stage 2 UPOS Alignment (%)", "s2_upos"),
        ("Stage 3 QA-SRL Span F1 (%)", "s3_span_f1"),
        ("Stage 4 HotpotQA Token F1 (%)", "s4_token_f1"),
        ("Stage 4 HotpotQA EM (%)", "s4_em"),
        ("Stage 5 GSM8K Token F1 (%)", "s5_token_f1"),
        ("Stage 5 GSM8K Num Acc (%)", "s5_num_acc"),
        ("Stage 6 MBPP Code F1 (%)", "s6_code_f1"),
        ("Stage 6 MBPP Syntax Valid (%)", "s6_syntax"),
    ]
    
    orig = {
        "s1_validity": 91.04,
        "s2_upos": 70.16,
        "s3_span_f1": 25.50,
        "s4_token_f1": 7.22,
        "s4_em": 0.00,
        "s5_token_f1": 31.28,
        "s5_num_acc": 0.00,
        "s6_code_f1": 12.89,
        "s6_syntax": 65.00,
    }
    
    print(f"{'Metric':32} | {'Original':8} | {'Seed 42':8} | {'Seed 43':8} | {'Seed 44':8} | {'Mean':8} | {'Spread':8} | {'Greedy (N=10)':13} | {'Greedy (N=25)':13}")
    print("-" * 135)
    for label, key in metrics:
        v_orig = orig[key]
        v_42 = sampled_runs[42][key]
        v_43 = sampled_runs[43][key]
        v_44 = sampled_runs[44][key]
        vals = [v_42, v_43, v_44]
        v_mean = np.mean(vals)
        v_spread = np.max(vals) - np.min(vals)
        v_gr10 = greedy_summary[key]
        v_gr25 = large_summary[key]
        print(f"{label:32} | {v_orig:8.2f} | {v_42:8.2f} | {v_43:8.2f} | {v_44:8.2f} | {v_mean:8.2f} | {v_spread:8.2f} | {v_gr10:13.2f} | {v_gr25:13.2f}")
        
    print("\nStage 7 Voice Probes (Seed 42 Sampled):")
    for prompt, resp in sampled_runs[42]["s7_probes"]:
        print(f"  User:   {prompt}")
        print(f"  AR.I.S: {resp}")
        
    print("\nStage 7 Voice Probes (Greedy):")
    for prompt, resp in greedy_summary["s7_probes"]:
        print(f"  User:   {prompt}")
        print(f"  AR.I.S: {resp}")

if __name__ == "__main__":
    main()
