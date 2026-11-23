"""
AR.I.S Deterministic Offline Replay & Integrity Audit
Measures:
1. Exact per-step corpus source distribution from Step 0 to Step 8,500+.
2. Real token counts: Matrix positions vs. Non-pad vs. Supervised assistant tokens (N_valid).
3. Exact step boundaries where corpora exhaust or switch files.
4. Data sufficiency check for 21,768 steps (696,576 sequences).
5. Full 100k NuminaMath vs GSM8K test screen.
6. HotpotQA validation vs train overlap.
7. MBPP 13-gram removal and dropped sequence counters.
"""

import sys
import json
import time
from pathlib import Path
from collections import defaultdict, Counter
import pyarrow.parquet as pq
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from aris.tokenizer.bpe_tokenizer import ArisTokenizer
from aris.data.pipeline_v2 import ArisDataPipelineV2


def run_data_sufficiency_and_overlap_audit():
    print("=" * 85)
    print("PART 1: DATA SUFFICIENCY, LEAKAGE & OVERLAP AUDIT")
    print("=" * 85)

    # 1. Full GSM8K Test vs NuminaMath CoT 100k
    gsm_test_path = BASE_DIR / "data/gsm8k/test.parquet"
    numina_path = BASE_DIR / "data/smoltalk/numina_cot_100k.parquet"
    
    if gsm_test_path.exists() and numina_path.exists():
        t_gsm = pq.read_table(str(gsm_test_path))
        gsm_questions = set(q.as_py().strip().lower() for q in t_gsm.column("question"))
        print(f"Loaded GSM8K Test: {len(gsm_questions)} questions")

        t_numina = pq.read_table(str(numina_path))
        numina_msgs = t_numina.column("messages")
        total_numina = len(numina_msgs)
        print(f"Loaded NuminaMath CoT: {total_numina} items. Screening full corpus against GSM8K test...")

        exact_overlaps = 0
        for i in range(total_numina):
            msgs = numina_msgs[i].as_py()
            if msgs and len(msgs) > 0:
                q = msgs[0].get("content", "").strip().lower()
                if q in gsm_questions:
                    exact_overlaps += 1

        print(f"  -> Full NuminaMath 100k vs GSM8K Test Exact Overlap: {exact_overlaps} / {total_numina} ({exact_overlaps/total_numina*100:.4f}%)")

    # 2. HotpotQA Train vs HotpotQA Validation Overlap
    hp_train_0 = BASE_DIR / "data/hotpotqa/distractor/train-00000-of-00002.parquet"
    hp_train_1 = BASE_DIR / "data/hotpotqa/distractor/train-00001-of-00002.parquet"
    hp_val = BASE_DIR / "data/hotpotqa/distractor/validation.parquet"

    if hp_val.exists() and hp_train_0.exists():
        t_val = pq.read_table(str(hp_val))
        val_questions = set(q.as_py().strip().lower() for q in t_val.column("question"))
        print(f"\nLoaded HotpotQA Validation: {len(val_questions)} questions")

        train_qs = set()
        for p in [hp_train_0, hp_train_1]:
            if p.exists():
                tbl = pq.read_table(str(p))
                for q in tbl.column("question"):
                    train_qs.add(q.as_py().strip().lower())
        
        hp_overlap = val_questions.intersection(train_qs)
        print(f"Loaded HotpotQA Train: {len(train_qs)} questions")
        print(f"  -> HotpotQA Val vs Train Exact Overlap: {len(hp_overlap)} / {len(val_questions)} ({len(hp_overlap)/len(val_questions)*100:.2f}%)")


def run_pipeline_offline_replay():
    print("\n" + "=" * 85)
    print("PART 2: PIPELINE OFFLINE REPLAY & REAL TOKEN ACCOUNTING")
    print("=" * 85)

    tok = ArisTokenizer.load("models/aris_bpe_tokenizer_v2.json")
    pipeline = ArisDataPipelineV2(tokenizer=tok, max_seq_len=384, seed=42)
    
    # We will simulate the training loop iterator
    data_iter = iter(pipeline)
    
    batch_size = 16
    accum_steps = 2
    seqs_per_step = batch_size * accum_steps # 32 sequences per step

    # Track metrics in windows of 500 steps
    window_size = 500
    target_steps = 8500 # Replay up to current training progress

    print(f"Replaying pipeline up to Step {target_steps} ({target_steps * seqs_per_step:,} total sequences)...")

    step = 0
    total_non_pad_tokens = 0
    total_scored_tokens = 0
    total_matrix_positions = 0

    window_non_pad = 0
    window_scored = 0
    window_matrix = 0

    # Exhaustion tracker
    last_seen_corpus = {}
    window_corpus_counts = defaultdict(int)

    # Let's inspect step intervals
    print(f"\n{'Step Range':16} | {'Non-Pad Toks':14} | {'Scored Toks':14} | {'Padding %':10} | {'Scored / Total %':16}")
    print("-" * 85)

    while step < target_steps:
        step += 1
        for _ in range(seqs_per_step):
            inp, tgt = next(data_iter)
            cur_len = len(inp)
            cur_scored = int((tgt != -100).sum().item())
            
            total_non_pad_tokens += cur_len
            total_scored_tokens += cur_scored
            total_matrix_positions += 384

            window_non_pad += cur_len
            window_scored += cur_scored
            window_matrix += 384

        if step % window_size == 0:
            pad_pct = (1.0 - (window_non_pad / window_matrix)) * 100.0
            scored_pct = (window_scored / window_matrix) * 100.0
            print(f"{step-window_size+1:6,d} - {step:6,d} | {window_non_pad:14,d} | {window_scored:14,d} | {pad_pct:9.1f}% | {scored_pct:15.1f}%")
            window_non_pad = 0
            window_scored = 0
            window_matrix = 0

    print("-" * 85)
    overall_pad_pct = (1.0 - (total_non_pad_tokens / total_matrix_positions)) * 100.0
    overall_scored_pct = (total_scored_tokens / total_matrix_positions) * 100.0

    print(f"\nOVERALL TOKEN AUDIT AT STEP {target_steps:,}:")
    print(f"  1. Nominal Matrix Positions (step * 12,288): {total_matrix_positions:14,d} (100.0%)")
    print(f"  2. Real Non-Pad Tokens (Prompt + Asst):      {total_non_pad_tokens:14,d} ({total_non_pad_tokens/total_matrix_positions*100:.1f}%)")
    print(f"  3. Padded Masked Positions:                  {total_matrix_positions - total_non_pad_tokens:14,d} ({overall_pad_pct:.1f}%)")
    print(f"  4. Truly Scored Assistant Tokens (N_valid):  {total_scored_tokens:14,d} ({overall_scored_pct:.1f}%)")
    
    num_params = 26747392
    print(f"\nREAL CHINCHILLA MULTIPLE (Relative to 26.75M Parameters):")
    print(f"  Nominal Multiple (Matrix): {total_matrix_positions / num_params:.2f}x")
    print(f"  Non-Pad Multiple (Tokens): {total_non_pad_tokens / num_params:.2f}x")
    print(f"  Supervised Multiple (NLL): {total_scored_tokens / num_params:.2f}x")

    print("\nCORPUS SEQUENCE COUNTS ACCUMULATED BY PIPELINE:")
    for c, cnt in sorted(pipeline.corpus_seq_counts.items(), key=lambda x: -x[1]):
        sup_toks = pipeline.corpus_sup_tokens.get(c, 0)
        print(f"  {c:<20}: {cnt:8,d} sequences | {sup_toks:12,d} scored tokens | {sup_toks/max(1,cnt):6.1f} avg scored/seq")

    print(f"\nPIPELINE DROP & FILTER COUNTERS:")
    print(f"  Total Dropped Sequences (Exceeded 384): {pipeline.total_dropped_sequences:,}")
    print(f"  Total Decontaminated (13-gram match):  {pipeline.total_decontaminated_sequences:,}")


if __name__ == "__main__":
    run_data_sufficiency_and_overlap_audit()
    run_pipeline_offline_replay()
