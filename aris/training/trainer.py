"""
AR.I.S — Unified Neural Model Trainer with Proven Continuous Resumption
Features:
  - Manifest-Driven Streaming: Ingests solely verified corpora via ArisDataPipelineV2.
  - Zero-Slice Truncation: 100% of samples terminate with <|im_end|>; context shrinks to fit responses.
  - Response-Only Loss Masking: Prompts masked with -100; zero double-shift causal alignment.
  - Hardware Efficiency: Automatic Mixed Precision (AMP bfloat16 / float16) + Gradient Accumulation.
  - Dynamic LR Schedule: Linear warmup with cosine decay to min_lr.
  - Milestone Archiving: Preserves checkpoint milestones every 2,500 steps.
  - Cloud Auto-Backup: Automatic mirroring to Google Drive on Google Colab.
"""

from __future__ import annotations
import os
import sys
import math
import time
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import asdict

import torch
import torch.nn as nn
from torch.optim import AdamW

from ..config import BASE_DIR, DATA_DIR, MODELS_DIR
from ..configs.model_config import ArisConfig
from ..models.aris_lm import ArisLM
from ..tokenizer.bpe_tokenizer import ArisTokenizer
from ..data.pipeline_v2 import ArisDataPipelineV2


def get_cosine_lr(step: int, warmup_steps: int, total_steps: int, max_lr: float, min_lr: float) -> float:
    """Computes learning rate with linear warmup and cosine decay."""
    if step < warmup_steps:
        return max_lr * float(step + 1) / float(max(1, warmup_steps))
    if step > total_steps:
        return min_lr
    decay_ratio = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


class ArisTrainer:
    """
    Unified Production Trainer for Modern ArisLM Architecture.
    """

    def __init__(
        self,
        model: Optional[ArisLM] = None,
        tokenizer: Optional[ArisTokenizer] = None,
        device: Optional[torch.device] = None,
        checkpoint_dir: Optional[Path] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device) if model is not None else None
        self.tokenizer = tokenizer
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else MODELS_DIR
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _set_windows_sleep_state(prevent_sleep: bool):
        """Prevents Windows from entering sleep mode during active training."""
        if sys.platform == "win32":
            try:
                import ctypes
                if prevent_sleep:
                    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
                else:
                    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
            except Exception:
                pass

    @classmethod
    def train_proven(
        cls,
        rung: str = "small",
        pilot: bool = False,
        budget_mult: float = 10.0,
        total_steps: Optional[int] = None,
        batch_size: int = 16,
        accum_steps: int = 2,
        lr: float = 6e-4,
        min_lr: float = 6e-5,
        warmup_steps: int = 500,
        grad_clip: float = 1.0,
        log_every: int = 50,
        save_every: int = 500,
        device_name: str = "auto",
        checkpoint_name: Optional[str] = None,
        resume_from: Optional[str] = None,
        resume: bool = True,
        backup_dir: Optional[Path] = None,
    ) -> Path:
        """
        Executes the proven clean training pipeline across all corpora with AMP bfloat16.
        """
        cls._set_windows_sleep_state(True)
        try:
            print("=" * 105)
            print("AR.I.S UNIFIED TRAINING ENGINE (PROVEN PIPELINE V2)")
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

            # 2. Load Frozen Tokenizer (V3 for Gen 3/Medium if present, otherwise V2)
            if rung.lower() in ("medium", "gen3") and (MODELS_DIR / "aris_bpe_tokenizer_v3.json").exists():
                tok_path = MODELS_DIR / "aris_bpe_tokenizer_v3.json"
            elif (MODELS_DIR / "aris_bpe_tokenizer_v2.json").exists():
                tok_path = MODELS_DIR / "aris_bpe_tokenizer_v2.json"
            else:
                tok_path = MODELS_DIR / "aris_bpe_tokenizer.json"
            if not tok_path.exists():
                raise FileNotFoundError(f"Tokenizer artifact not found at {tok_path}")
            tokenizer = ArisTokenizer.load(tok_path)
            print(f"Tokenizer:           {tok_path.name} (Vocab: {tokenizer.vocab_size:,})")

            # 3. Model Architecture Configuration from Rung
            target_seq_len = 1024 if rung.lower() in ("medium", "gen3") else 384
            cfg = ArisConfig.from_rung(rung=rung, vocab_size=tokenizer.vocab_size, max_seq_len=target_seq_len)
            param_count = cfg.expected_param_count()
            print(f"Model Rung:          {rung.upper()} ({param_count:,} parameters)")

            # 4. Token & Step Budget Calculation
            tokens_per_step = batch_size * cfg.max_seq_len * accum_steps
            b_mult = 1.0 if pilot else budget_mult
            target_token_budget = int(param_count * b_mult)
            mode_label = f"Pilot Run ({target_token_budget:,} tokens)" if pilot else f"{b_mult:.1f}x Chinchilla Scaling Run ({target_token_budget:,} tokens)"

            if total_steps is None:
                total_steps = math.ceil(target_token_budget / tokens_per_step)

            print(f"Training Mode:       {mode_label} -> {total_steps:,} total optimizer steps")
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

            # 7. Checkpoint Resolution
            if checkpoint_name is None:
                if pilot:
                    ckpt_name = f"aris_lm_clean_{rung}_pilot.pt"
                else:
                    mult_str = f"{int(b_mult)}" if b_mult.is_integer() else f"{b_mult}"
                    ckpt_name = f"aris_lm_clean_{rung}_{mult_str}x.pt"
            else:
                ckpt_name = checkpoint_name

            ckpt_path = MODELS_DIR / ckpt_name
            start_step = 0
            tokens_trained = 0

            load_source = None
            if resume_from:
                candidate = Path(resume_from)
                if not candidate.is_absolute():
                    candidate = BASE_DIR / resume_from if (BASE_DIR / resume_from).exists() else MODELS_DIR / resume_from
                if candidate.exists():
                    load_source = candidate
                else:
                    print(f"Warning: --resume-from path '{resume_from}' not found. Checking {ckpt_path}...")

            if load_source is None and resume and ckpt_path.exists():
                load_source = ckpt_path

            if load_source and load_source.exists():
                print(f"Resuming clean training from {load_source}...")
                ckpt = torch.load(load_source, map_location=device, weights_only=False)
                model.load_state_dict(ckpt["model_state_dict"])
                if "optimizer_state_dict" in ckpt:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                start_step = ckpt.get("step", 0)
                tokens_trained = ckpt.get("tokens_trained", start_step * tokens_per_step)
                if "pipeline_state" in ckpt:
                    pipeline.set_state(ckpt["pipeline_state"])
                    data_iter = iter(pipeline)
                if "rng_state" in ckpt:
                    try:
                        g_rng = ckpt["rng_state"]
                        if isinstance(g_rng, torch.Tensor):
                            g_rng = g_rng.to(device="cpu", dtype=torch.uint8)
                        torch.set_rng_state(g_rng)
                    except Exception:
                        pass
                print(f"  Resumed at Step {start_step:,} ({tokens_trained:,} tokens already completed)")
                if load_source != ckpt_path:
                    print(f"  Target Save Checkpoint: {ckpt_path.name}")

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
                curr_lr = get_cosine_lr(step, warmup_steps, total_steps, lr, min_lr)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = curr_lr

                accum_loss = 0.0

                for micro_step in range(accum_steps):
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
                    step_window_tokens += int((y != -100).sum().item())

                tokens_trained += tokens_per_step

                if use_amp and amp_dtype == torch.float16:
                    scaler.unscale_(optimizer)
                    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                    optimizer.step()

                optimizer.zero_grad()

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
                        "model_config": cfg.__dict__,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "pipeline_state": pipeline.get_state(),
                        "rng_state": torch.get_rng_state(),
                    }
                    torch.save(checkpoint_payload, ckpt_path)
                    print(f"  [Checkpoint Saved] -> {ckpt_path.name} (Step {step + 1:,}, Loss {accum_loss:.4f})")

                    # Milestone Archiving (every 2,500 steps)
                    if (step + 1) % 2500 == 0 or (step + 1) == total_steps:
                        milestone_path = ckpt_path.with_name(f"{ckpt_path.stem}_step_{step + 1}{ckpt_path.suffix}")
                        torch.save(checkpoint_payload, milestone_path)
                        print(f"  [Milestone Checkpoint Archived] -> {milestone_path.name}")

                    # Mirror to Google Drive if backup_dir provided
                    if backup_dir:
                        try:
                            backup_dir.mkdir(parents=True, exist_ok=True)
                            dest = backup_dir / ckpt_path.name
                            shutil.copy2(ckpt_path, dest)
                            print(f"  --> [Cloud Auto-Backup] Mirrored to Google Drive: {dest}")
                        except Exception as e:
                            print(f"  --> [Cloud Auto-Backup Warning] Could not mirror to {backup_dir}: {e}")

            total_time = time.time() - t_start
            print("-" * 105)
            print(f"Clean Training Run Completed in {int(total_time // 60)}m {int(total_time % 60):02d}s!")
            print(f"Final Checkpoint: {ckpt_path} ({ckpt_path.stat().st_size / (1024*1024):.2f} MB)")
            print("=" * 105)
            return ckpt_path

        finally:
            cls._set_windows_sleep_state(False)

    @classmethod
    def load_model(cls, checkpoint_path: Path, device: Optional[torch.device] = None) -> Tuple[ArisLM, int]:
        """Loads ArisLM and returns (model, cumulative_step)."""
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        cfg_dict = checkpoint.get("model_config") or checkpoint.get("config", {})

        config = ArisConfig(
            vocab_size=cfg_dict["vocab_size"],
            max_seq_len=cfg_dict["max_seq_len"],
            embed_dim=cfg_dict["embed_dim"],
            num_layers=cfg_dict["num_layers"],
            num_query_heads=cfg_dict["num_query_heads"],
            num_kv_heads=cfg_dict["num_kv_heads"],
            ff_dim=cfg_dict.get("ff_dim"),
            tie_word_embeddings=cfg_dict.get("tie_word_embeddings", True),
        )

        actual_sd_elements = sum(v.numel() for v in checkpoint["model_state_dict"].values())
        expected_sd_elements = config.expected_state_dict_count()
        if actual_sd_elements != expected_sd_elements:
            raise ValueError(
                f"Checkpoint state_dict count mismatch: actual {actual_sd_elements} != expected {expected_sd_elements}"
            )

        model = ArisLM(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        return model, checkpoint.get("step", 0)
