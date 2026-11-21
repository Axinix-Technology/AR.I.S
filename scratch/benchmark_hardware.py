"""
Step 0: Hardware Benchmark Suite (CPU + CUDA Supported)
Measures actual training throughput (forward + cross-entropy + backward + AdamW optimizer step)
across Pico (~6.5M), Small (~26.7M), Medium (~107M), and Base (~279M) rungs.
Supports both local CPU and Kaggle/Colab CUDA GPU execution.
Reports tokens/second and exact estimated wall-clock hours for 20x Chinchilla token budgets.
"""

import os
import sys
import time
import argparse
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
    device: torch.device,
    vocab_size: int = 8192,
    max_seq_len: int = 2048,
    train_seq_len: int = 384,
    batch_size: int = 4,
    duration_sec: float = 30.0,
):
    print(f"\nBenchmarking {name} rung on {device}...")
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
    model = ArisLM(cfg).to(device)
    total_params = cfg.expected_param_count(tied_embeddings=True)
    model.train()
    optimizer = AdamW(model.parameters(), lr=1e-4)

    # Pre-generate synthetic batch on target device
    x = torch.randint(0, vocab_size, (batch_size, train_seq_len), dtype=torch.long, device=device)
    y = torch.randint(0, vocab_size, (batch_size, train_seq_len), dtype=torch.long, device=device)

    # Warmup
    for _ in range(3):
        optimizer.zero_grad()
        logits, loss, _ = model(x, targets=y)
        loss.backward()
        optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()

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
        
        if device.type == "cuda":
            torch.cuda.synchronize()
            
        elapsed = time.perf_counter() - start_time
        if elapsed >= duration_sec:
            break

    throughput = total_tokens / elapsed
    budget_20x = total_params * 20
    hours_20x = (budget_20x / throughput) / 3600.0
    days_20x = hours_20x / 24.0

    pilot_budget_5pct = budget_20x * 0.05
    pilot_hours = (pilot_budget_5pct / throughput) / 3600.0

    print(f"  Completed {step_count} steps ({total_tokens:,} tokens) in {elapsed:.2f}s => {throughput:.1f} tok/s")

    # Clean memory between rungs
    del model, optimizer, x, y
    if device.type == "cuda":
        torch.cuda.empty_cache()

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
    parser = argparse.ArgumentParser(description="AR.I.S Step 0 Hardware Benchmark")
    parser.add_argument("--duration", type=float, default=15.0, help="Benchmark duration in seconds per rung")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Compute device")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training step")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 105)
    print("STEP 0: HARDWARE BENCHMARK & 20x CHINCHILLA TRAINING FEASIBILITY")
    print(f"Selected Device:   {device}")
    if device.type == "cuda":
        print(f"GPU Model:         {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM:        {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        print(f"CUDA Version:      {torch.version.cuda}")
    else:
        print(f"CPU Cores:         {os.cpu_count()} logical cores")
        print(f"PyTorch Threads:   {torch.get_num_threads()}")
    print(f"Benchmark Mode:    Full Forward + CrossEntropy + Backward + AdamW Step (seq_len=384, batch_size={args.batch_size})")
    print(f"Duration Per Rung: {args.duration} seconds")
    print("=" * 105)

    rungs = [
        ("Pico",   256,  6,  4, 2,  704),
        ("Small",  512,  8,  8, 2, 1408),
        ("Medium", 768, 16, 12, 4, 2048),
        ("Base",  1024, 24, 16, 4, 2816),
    ]

    results = []
    for name, embed_dim, num_layers, num_query_heads, num_kv_heads, ff_dim in rungs:
        # On CPU, run Base for half duration if user didn't specify long duration to avoid excessive wait
        run_dur = args.duration if (device.type == "cuda" or name != "Base") else min(args.duration, 10.0)
        res = benchmark_rung(
            name=name,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            ff_dim=ff_dim,
            device=device,
            batch_size=args.batch_size,
            duration_sec=run_dur,
        )
        results.append(res)

    print("\n" + "=" * 105)
    print("HARDWARE BENCHMARK SUMMARY & ESTIMATED 20x TRAINING TIME")
    print(f"Device: {device} " + (f"({torch.cuda.get_device_name(0)})" if device.type == 'cuda' else f"({os.cpu_count()} CPU cores)"))
    print("=" * 105)
    print(f"{'Rung':8} | {'Params':12} | {'Throughput':14} | {'20x Token Budget':18} | {'Est. 20x Hours':16} | {'Est. 20x Days':14} | {'5% Pilot Hours':14}")
    print("-" * 105)

    for r in results:
        b_str = f"{r['budget_20x_tokens'] / 1e6:.2f}M" if r['budget_20x_tokens'] < 1e9 else f"{r['budget_20x_tokens'] / 1e9:.2f}B"
        print(f"{r['name']:8} | {r['total_params']:12,d} | {r['tokens_per_sec']:10.1f} tok/s | {b_str:18} | {r['hours_20x']:14.1f} h | {r['days_20x']:12.1f} d | {r['pilot_5pct_hours']:12.2f} h")


if __name__ == "__main__":
    main()
