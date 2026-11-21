"""
Step 0: Hardware Benchmark Suite
Measures actual training throughput (forward + cross-entropy + backward + AdamW optimizer step)
across Pico (~6.5M), Small (~26.7M), and Medium (~107M) rungs on available CPU hardware.
Reports tokens/second and exact estimated wall-clock hours for 20x Chinchilla token budgets.
"""

import os
import sys
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from aris.configs.model_config import ArisConfig
from aris.models.aris_lm import ArisLM

def benchmark_rung(
    name: str,
    embed_dim: int,
    num_layers: int,
    num_query_heads: int,
    num_kv_heads: int,
    ff_dim: int,
    vocab_size: int = 8192,
    max_seq_len: int = 2048,
    train_seq_len: int = 384,
    batch_size: int = 4,
    duration_sec: float = 60.0,
):
    print(f"\nBenchmarking {name} rung...")
    cfg = ArisConfig(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        ff_dim=ff_dim,
        tie_word_embeddings=True,
    )
    model = ArisLM(cfg)
    total_params = cfg.expected_param_count(tied_embeddings=True)
    model.train()
    optimizer = AdamW(model.parameters(), lr=1e-4)

    # Pre-generate synthetic batch
    x = torch.randint(0, vocab_size, (batch_size, train_seq_len), dtype=torch.long)
    y = torch.randint(0, vocab_size, (batch_size, train_seq_len), dtype=torch.long)

    # Warmup
    for _ in range(3):
        optimizer.zero_grad()
        logits, loss, _ = model(x, targets=y)
        loss.backward()
        optimizer.step()

    # Timed run
    tokens_per_step = batch_size * train_seq_len
    total_tokens = 0
    step_count = 0
    start_time = time.perf_counter()

    while True:
        optimizer.zero_grad()
        logits, loss, _ = model(x, targets=y)
        loss.backward()
        optimizer.step()
        
        step_count += 1
        total_tokens += tokens_per_step
        elapsed = time.perf_counter() - start_time
        if elapsed >= duration_sec:
            break

    throughput = total_tokens / elapsed
    budget_20x = total_params * 20
    hours_20x = (budget_20x / throughput) / 3600.0
    days_20x = hours_20x / 24.0

    pilot_budget_5pct = budget_20x * 0.05
    pilot_hours = (pilot_budget_5pct / throughput) / 3600.0

    return {
        "name": name,
        "total_params": total_params,
        "step_count": step_count,
        "elapsed_sec": elapsed,
        "tokens_per_sec": throughput,
        "budget_20x_tokens": budget_20x,
        "hours_20x": hours_20x,
        "days_20x": days_20x,
        "pilot_5pct_hours": pilot_hours,
    }


def main():
    print("=" * 95)
    print("STEP 0: HARDWARE BENCHMARK & 20x CHINCHILLA TRAINING FEASIBILITY")
    print(f"Device:            CPU ({os.cpu_count()} logical cores)")
    print(f"PyTorch Threads:   {torch.get_num_threads()}")
    print("Benchmark Mode:    Full Forward + CrossEntropy + Backward + AdamW Step (seq_len=384, batch_size=4)")
    print("=" * 95)

    rungs = [
        ("Pico", 256, 6, 4, 2, 704),
        ("Small", 512, 8, 8, 2, 1408),
        ("Medium", 768, 16, 12, 4, 2048),
    ]

    results = []
    for name, embed_dim, num_layers, num_query_heads, num_kv_heads, ff_dim in rungs:
        res = benchmark_rung(
            name=name,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            ff_dim=ff_dim,
            duration_sec=60.0,
        )
        results.append(res)

    print("\n" + "=" * 95)
    print("HARDWARE BENCHMARK SUMMARY & ESTIMATED 20x TRAINING TIME")
    print("=" * 95)
    print(f"{'Rung':8} | {'Params':12} | {'Throughput':14} | {'20x Token Budget':18} | {'Est. 20x Hours':16} | {'Est. 20x Days':14} | {'5% Pilot Hours':14}")
    print("-" * 105)

    for r in results:
        b_str = f"{r['budget_20x_tokens'] / 1e6:.2f}M" if r['budget_20x_tokens'] < 1e9 else f"{r['budget_20x_tokens'] / 1e9:.2f}B"
        print(f"{r['name']:8} | {r['total_params']:12,d} | {r['tokens_per_sec']:10.1f} tok/s | {b_str:18} | {r['hours_20x']:14.1f} h | {r['days_20x']:12.1f} d | {r['pilot_5pct_hours']:12.2f} h")


if __name__ == "__main__":
    main()
