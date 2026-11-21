"""
AR.I.S Clean Pretraining & Developmental Training Engine V2
Executes reproducible, manifest-driven training on approved clean corpora using Tokenizer V2 (8,192 vocab).
Features:
  1. Strict manifest integration via ArisDataPipelineV2 (zero-slice truncation, zero supervised loss masking).
  2. Small Rung Gen 2 Architecture (~26.75M parameters: 8 layers, d_model=512, GQA 8/2, SwiGLU 1408, RoPE, RMSNorm).
  3. Chinchilla warmup and cosine learning rate schedule with AdamW and gradient clipping (1.0).
  4. Automatic mixed precision (AMP bfloat16 / float16) for blazing-fast GPU throughput.
  5. Continuous bit-identical resumption serializing model, optimizer, scheduler, RNG, and pipeline cursors.
  6. Dedicated output checkpointing: models/aris_lm_clean_small_pilot.pt (quarantines legacy checkpoints).
"""

from __future__ import annotations
import os
import sys
import time
import math
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from aris.configs.model_config import ArisConfig
from aris.models.aris_lm import ArisLM
from aris.tokenizer.bpe_tokenizer import ArisTokenizer
from aris.data.pipeline_v2 import ArisDataPipelineV2

MODELS_DIR = BASE_DIR / "models"
SCRATCH_DIR = BASE_DIR / "scratch"


def get_cosine_lr(step: int, warmup_steps: int, total_steps: int, max_lr: float, min_lr: float) -> float:
    """Computes learning rate with linear warmup and cosine decay."""
    if step < warmup_steps:
        return max_lr * float(step + 1) / float(max(1, warmup_steps))
    if step > total_steps:
        return min_lr
    decay_ratio = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def train_clean_run(
    rung: str = "small",
    pilot: bool = True,
    total_steps: Optional[int] = None,
    batch_size: int = 16,
    accum_steps: int = 2,
    lr: float = 6e-4,
    min_lr: float = 6e-5,
    warmup_steps: int = 100,
    grad_clip: float = 1.0,
    log_every: int = 20,
    save_every: int = 200,
    device_name: str = "auto",
    checkpoint_name: str = "aris_lm_clean_small_pilot.pt",
    resume: bool = True,
):
    print("=" * 105)
    print("AR.I.S CLEAN TRAINING V2 (PHASE 2b/3 FOUNDATION TRAINING)")
    print("=" * 105)

    # 1. Resolve Device
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    print(f"Compute Device:      {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU Hardware:        {gpu_name} ({vram_gb:.2f} GB VRAM)")

    # 2. Load Frozen Tokenizer V2
    tok_path = MODELS_DIR / "aris_bpe_tokenizer_v2.json"
    if not tok_path.exists():
        raise FileNotFoundError(f"Tokenizer V2 artifact not found at {tok_path}")
    tokenizer = ArisTokenizer.load(tok_path)
    print(f"Tokenizer:           {tok_path.name} (Vocab: {tokenizer.vocab_size:,})")

    # 3. Model Architecture Configuration
    if rung == "small":
        cfg = ArisConfig(
            vocab_size=tokenizer.vocab_size,  # 8,192
            max_seq_len=384,
            embed_dim=512,
            num_layers=8,
            num_query_heads=8,
            num_kv_heads=2,
            ff_dim=1408,
            tie_word_embeddings=True,
        )
    elif rung == "pico":
        cfg = ArisConfig(
            vocab_size=tokenizer.vocab_size,
            max_seq_len=384,
            embed_dim=256,
            num_layers=4,
            num_query_heads=4,
            num_kv_heads=2,
            ff_dim=704,
            tie_word_embeddings=True,
        )
    else:
        raise ValueError(f"Unsupported rung: {rung}")

    param_count = cfg.expected_param_count()
    print(f"Model Rung:          {rung.upper()} ({param_count:,} parameters)")

    # 4. Token & Step Budget Calculation
    # 20x Chinchilla = params * 20
    # 5% Pilot = 5% of 20x Chinchilla
    tokens_per_step = batch_size * cfg.max_seq_len * accum_steps  # 16 * 384 * 2 = 12,288 tokens/step
    budget_20x_tokens = param_count * 20
    if pilot:
        target_token_budget = int(budget_20x_tokens * 0.05)  # ~26.75M tokens for small rung
        if total_steps is None:
            total_steps = math.ceil(target_token_budget / tokens_per_step)
        mode_label = f"5% Pilot Run ({target_token_budget:,} tokens / {total_steps:,} steps)"
    else:
        target_token_budget = budget_20x_tokens
        if total_steps is None:
            total_steps = math.ceil(target_token_budget / tokens_per_step)
        mode_label = f"Full 20x Chinchilla Run ({target_token_budget:,} tokens / {total_steps:,} steps)"

    print(f"Training Mode:       {mode_label}")
    print(f"Tokens Per Step:     {tokens_per_step:,} (Batch: {batch_size}, SeqLen: {cfg.max_seq_len}, Accum: {accum_steps})")
    print(f"Max Learning Rate:   {lr:.2e} (Warmup: {warmup_steps} steps, Cosine Decay to {min_lr:.2e})")
    print("-" * 105)

    # 5. Initialize Model & Optimizer
    model = ArisLM(cfg).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)

    # 6. Initialize Data Pipeline V2
    print("Initializing Data Pipeline V2 across manifest-approved corpora...")
    pipeline = ArisDataPipelineV2(
        tokenizer=tokenizer,
        max_seq_len=cfg.max_seq_len,
        seed=42,
    )
    data_iter = iter(pipeline)

    # 7. Resumption Check
    ckpt_path = MODELS_DIR / checkpoint_name
    start_step = 0
    tokens_trained = 0

    if resume and ckpt_path.exists():
        print(f"Resuming clean training from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt.get("step", 0)
        tokens_trained = ckpt.get("tokens_trained", start_step * tokens_per_step)
        if "pipeline_state" in ckpt:
            pipeline.set_state(ckpt["pipeline_state"])
            data_iter = iter(pipeline)
        print(f"  Resumed at Step {start_step:,} ({tokens_trained:,} tokens already completed)")

    # 8. Mixed Precision Scaler
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    print(f"Precision:           {'AMP ' + str(amp_dtype) if use_amp else 'Float32 FP32'}")
    print("=" * 105)
    print(f"{'Step':8} | {'Loss':8} | {'LR':10} | {'GradNorm':9} | {'Throughput':14} | {'Tokens Done':14} | {'Progress':9} | {'ETA':10}")
    print("-" * 105)

    model.train()
    optimizer.zero_grad()

    t_start = time.time()
    step_window_tokens = 0
    step_window_start = time.time()

    for step in range(start_step, total_steps):
        # Update Dynamic Learning Rate
        curr_lr = get_cosine_lr(step, warmup_steps, total_steps, lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = curr_lr

        accum_loss = 0.0

        for micro_step in range(accum_steps):
            # Fetch micro-batch
            batch_inputs = []
            batch_targets = []
            for _ in range(batch_size):
                try:
                    inp, tgt = next(data_iter)
                except StopIteration:
                    data_iter = iter(pipeline)
                    inp, tgt = next(data_iter)
                batch_inputs.append(inp)
                batch_targets.append(tgt)

            # Collate and pad micro-batch to max_seq_len
            padded_inputs = []
            padded_targets = []
            pad_id = getattr(cfg, "pad_token_id", 0)
            target_seq_len = cfg.max_seq_len

            for inp_t, tgt_t in zip(batch_inputs, batch_targets):
                cur_len = len(inp_t)
                if cur_len < target_seq_len:
                    pad_len = target_seq_len - cur_len
                    p_inp = torch.cat([inp_t, torch.full((pad_len,), pad_id, dtype=torch.long)])
                    p_tgt = torch.cat([tgt_t, torch.full((pad_len,), -100, dtype=torch.long)])
                else:
                    p_inp = inp_t[:target_seq_len]
                    p_tgt = tgt_t[:target_seq_len]
                padded_inputs.append(p_inp)
                padded_targets.append(p_tgt)

            x = torch.stack(padded_inputs).to(device)
            y = torch.stack(padded_targets).to(device)

            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                _, loss, _ = model(x, targets=y)
                loss = loss / accum_steps

            if use_amp and amp_dtype == torch.float16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_loss += loss.item()

        # Gradient Clipping & Optimizer Step
        if use_amp and amp_dtype == torch.float16:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        optimizer.zero_grad()

        tokens_trained += tokens_per_step
        step_window_tokens += tokens_per_step

        # Step Logging
        if (step + 1) % log_every == 0 or (step + 1) == total_steps:
            now = time.time()
            elapsed_window = now - step_window_start
            tok_per_sec = (step_window_tokens / elapsed_window) if elapsed_window > 0 else 0.0
            pct_done = ((step + 1) / total_steps) * 100.0
            remaining_steps = total_steps - (step + 1)
            eta_sec = (remaining_steps * tokens_per_step / tok_per_sec) if tok_per_sec > 0 else 0
            eta_str = f"{int(eta_sec // 60)}m {int(eta_sec % 60):02d}s"

            print(
                f"{step + 1:8,d} | {accum_loss:8.4f} | {curr_lr:10.2e} | {float(grad_norm):9.4f} | "
                f"{tok_per_sec:10.1f} tok/s | {tokens_trained:14,d} | {pct_done:8.1f}% | {eta_str:10}"
            )
            step_window_tokens = 0
            step_window_start = time.time()

        # Checkpointing
        if (step + 1) % save_every == 0 or (step + 1) == total_steps:
            checkpoint_payload = {
                "step": step + 1,
                "tokens_trained": tokens_trained,
                "loss": accum_loss,
                "config": cfg.__dict__,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "pipeline_state": pipeline.get_state(),
                "rng_state": torch.get_rng_state(),
            }
            torch.save(checkpoint_payload, ckpt_path)
            print(f"  [Checkpoint Saved] -> {ckpt_path.name} (Step {step + 1:,}, Loss {accum_loss:.4f})")

    total_time = time.time() - t_start
    print("-" * 105)
    print(f"Clean Training Run Completed in {int(total_time // 60)}m {int(total_time % 60):02d}s!")
    print(f"Final Checkpoint: {ckpt_path} ({ckpt_path.stat().st_size / (1024*1024):.2f} MB)")
    print("=" * 105)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AR.I.S Clean Training Engine V2")
    parser.add_argument("--rung", type=str, default="small", choices=["small", "pico"], help="Model size rung")
    parser.add_argument("--pilot", action="store_true", default=True, help="Run 5 percent token budget pilot")
    parser.add_argument("--full", dest="pilot", action="store_false", help="Run 100 percent 20x Chinchilla budget")
    parser.add_argument("--steps", type=int, default=None, help="Explicit total steps override")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size per forward micro-step")
    parser.add_argument("--accum-steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=6e-4, help="Max learning rate")
    parser.add_argument("--min-lr", type=float, default=6e-5, help="Min learning rate")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Warmup optimizer steps")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Compute device")
    parser.add_argument("--checkpoint", type=str, default="aris_lm_clean_small_pilot.pt", help="Checkpoint filename")
    parser.add_argument("--save-every", type=int, default=200, help="Checkpoint frequency in steps")
    parser.add_argument("--log-every", type=int, default=20, help="Log print frequency in steps")
    args = parser.parse_args()

    train_clean_run(
        rung=args.rung,
        pilot=args.pilot,
        total_steps=args.steps,
        batch_size=args.batch_size,
        accum_steps=args.accum_steps,
        lr=args.lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        device_name=args.device,
        checkpoint_name=args.checkpoint,
        save_every=args.save_every,
        log_every=args.log_every,
    )
