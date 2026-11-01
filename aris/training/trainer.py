"""
AR.I.S — Neural Model Trainer with Incremental Resumption
Features:
  - Phase-by-phase incremental training (e.g. today 500, tomorrow 1,000 -> cumulative 1,500)
  - Restores model weights, optimizer states, learning rate schedules, and step counts
  - Preserves previous learning progress without resetting weights or checkpoints
"""

from __future__ import annotations
import sys
import math
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from ..models.aris_lm import ArisLM
from ..tokenizer.bpe_tokenizer import ArisTokenizer
from ..data.dataset import EnterpriseDataLoader


class ArisTrainer:
    """
    Continuous & Incremental Training Engine for ArisLM.
    """

    def __init__(
        self,
        model: ArisLM,
        tokenizer: ArisTokenizer,
        device: Optional[torch.device] = None,
        checkpoint_dir: Optional[Path] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("models")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _set_windows_sleep_state(prevent_sleep: bool):
        """Prevents Windows from entering sleep mode during active training."""
        if sys.platform == "win32":
            try:
                import ctypes
                # ES_CONTINUOUS = 0x80000000, ES_SYSTEM_REQUIRED = 0x00000001
                if prevent_sleep:
                    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
                else:
                    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
            except Exception:
                pass

    def train(
        self,
        parquet_dir: Path,
        ud_archive_path: Optional[Path] = None,
        qasrl_dir: Optional[Path] = None,
        hotpotqa_dir: Optional[Path] = None,
        additional_steps: int = 500,
        batch_size: int = 8,
        accum_steps: int = 1,
        max_threads: Optional[int] = None,
        lr: float = 3e-4,
        warmup_steps: int = 50,
        grad_clip: float = 1.0,
        save_every: int = 100,
        log_every: int = 10,
        max_seq_len: int = 384,
        checkpoint_name: str = "aris_lm_checkpoint.pt",
        resume: bool = True,
        focus: str = "all",
        rank: int = 0,
        world_size: int = 1,
    ) -> Dict[str, Any]:
        """
        Executes cumulative incremental training combining all 4 foundational corpora:
        UltraChat + Universal Dependencies + QA-SRL + HotpotQA.
        Supports Protocol 1 (Thermal Management via Gradient Accumulation & Thread Limiter)
        and Windows Keep-Awake kernel lock.
        Supports Modular Cognitive Focus ('all', 'language', 'reasoning') and Distributed Sharding (rank/world_size).
        """
        if max_threads and max_threads > 0:
            torch.set_num_threads(max_threads)
            print(f"[Protocol 1 Cool Training] Capped CPU Threads to {max_threads}")

        self._set_windows_sleep_state(True)
        print("[Protocol 1 Windows Keep-Awake] System sleep locked for unattended training.")

        try:
            return self._execute_train_loop(
                parquet_dir=parquet_dir,
                ud_archive_path=ud_archive_path,
                qasrl_dir=qasrl_dir,
                hotpotqa_dir=hotpotqa_dir,
                additional_steps=additional_steps,
                batch_size=batch_size,
                accum_steps=accum_steps,
                lr=lr,
                warmup_steps=warmup_steps,
                grad_clip=grad_clip,
                save_every=save_every,
                log_every=log_every,
                max_seq_len=max_seq_len,
                checkpoint_name=checkpoint_name,
                resume=resume,
                focus=focus,
                rank=rank,
                world_size=world_size,
            )
        finally:
            self._set_windows_sleep_state(False)
            print("[Protocol 1 Windows Keep-Awake] System sleep lock released.")

    def _execute_train_loop(
        self,
        parquet_dir: Path,
        ud_archive_path: Optional[Path],
        qasrl_dir: Optional[Path],
        hotpotqa_dir: Optional[Path],
        additional_steps: int,
        batch_size: int,
        accum_steps: int,
        lr: float,
        warmup_steps: int,
        grad_clip: float,
        save_every: int,
        log_every: int,
        max_seq_len: int,
        checkpoint_name: str,
        resume: bool,
        focus: str = "all",
        rank: int = 0,
        world_size: int = 1,
    ) -> Dict[str, Any]:

        save_path = self.checkpoint_dir / checkpoint_name
        current_step = 0
        optimizer_state = None

        if resume and save_path.exists():
            print(f"\n[Incremental Training] Found existing checkpoint: {save_path}")
            checkpoint = torch.load(save_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            current_step = checkpoint.get("step", 0)
            optimizer_state = checkpoint.get("optimizer_state_dict")
            print(f"[Incremental Training] Resumed from cumulative step: {current_step}")
        else:
            print(f"\n[New Training] Starting fresh from step 0.")

        target_total_steps = current_step + additional_steps

        optimizer = AdamW(self.model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
        if optimizer_state:
            try:
                optimizer.load_state_dict(optimizer_state)
            except Exception:
                print("  Note: Initialized fresh optimizer state for new learning phase.")

        scheduler = CosineAnnealingLR(optimizer, T_max=max(1, additional_steps), eta_min=lr * 0.1)

        loader = EnterpriseDataLoader.get_loader(
            ultrachat_dir=parquet_dir,
            ud_archive_path=ud_archive_path,
            qasrl_dir=qasrl_dir,
            hotpotqa_dir=hotpotqa_dir,
            tokenizer=self.tokenizer,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            split="train",
            focus=focus,
            rank=rank,
            world_size=world_size,
        )

        print(f"\n=======================================================")
        print(f"  AR.I.S Incremental Neural Training Session")
        print(f"=======================================================")
        print(f"Device:             {self.device}")
        print(f"Model Parameters:   {self.model.num_parameters:,}")
        print(f"Starting Step:      {current_step}")
        print(f"Steps this run:     +{additional_steps}")
        print(f"Target Total Step:  {target_total_steps}")
        print(f"Batch Size:         {batch_size}")
        print(f"Cognitive Focus:    {focus.upper()}")
        if world_size > 1:
            print(f"Data Sharding:      Rank {rank} / World Size {world_size}")
        print(f"Learning Rate:      {lr}\n")

        self.model.train()
        loader_iter = iter(loader)
        running_loss = 0.0
        session_step = 0
        start_time = time.time()
        history: List[Dict[str, Any]] = []

        optimizer.zero_grad()

        while current_step < target_total_steps:
            try:
                batch_inputs, batch_targets = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch_inputs, batch_targets = next(loader_iter)

            batch_inputs = batch_inputs.to(self.device)
            batch_targets = batch_targets.to(self.device)

            _, loss, _ = self.model(batch_inputs, targets=batch_targets)
            if loss is None or torch.isnan(loss):
                continue

            # Scale loss for gradient accumulation
            scaled_loss = loss / accum_steps
            scaled_loss.backward()
            running_loss += loss.item()

            # Execute optimizer step after accum_steps micro-steps
            if (session_step + 1) % accum_steps == 0 or (current_step + 1) >= target_total_steps:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)
                optimizer.step()

                if session_step < warmup_steps and current_step < warmup_steps:
                    warmup_lr = lr * float(session_step + 1) / float(max(1, warmup_steps))
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = warmup_lr
                else:
                    scheduler.step()

                optimizer.zero_grad()
                current_step += 1

            session_step += 1

            if session_step % log_every == 0:
                avg_loss = running_loss / log_every
                elapsed = time.time() - start_time
                steps_per_sec = session_step / max(elapsed, 1e-5)
                current_lr = optimizer.param_groups[0]["lr"]

                print(f"[Step {current_step:05d} (Cumulative)] Loss: {avg_loss:.4f} | LR: {current_lr:.6f} | Speed: {steps_per_sec:.2f} st/s")
                history.append({"step": current_step, "loss": round(avg_loss, 4), "lr": current_lr})
                running_loss = 0.0

            if session_step % save_every == 0 or current_step == target_total_steps:
                self.save_checkpoint(save_path, step=current_step, optimizer=optimizer, loss=loss.item())
                print(f"  --> Checkpoint saved! Cumulative trained steps: {current_step}")

        total_time = time.time() - start_time
        print(f"\n[Training Phase Complete] Successfully completed +{session_step} steps in {total_time:.1f}s.")
        print(f"Cumulative Model Knowledge: {current_step} total steps stored in {save_path}\n")

        return {
            "cumulative_steps": current_step,
            "session_steps": session_step,
            "total_time_seconds": round(total_time, 2),
            "final_checkpoint": str(save_path),
            "history": history,
        }

    def save_checkpoint(
        self,
        filepath: Path,
        step: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
        loss: Optional[float] = None,
    ) -> None:
        """Saves weights, config, and cumulative training state."""
        checkpoint = {
            "step": step,
            "loss": loss,
            "model_state_dict": self.model.state_dict(),
            "model_config": {
                "vocab_size": self.model.config.vocab_size,
                "max_seq_len": self.model.config.max_seq_len,
                "embed_dim": self.model.config.embed_dim,
                "num_layers": self.model.config.num_layers,
                "num_query_heads": self.model.config.num_query_heads,
                "num_kv_heads": self.model.config.num_kv_heads,
                "ff_dim": self.model.config.ff_dim,
            },
        }
        if optimizer:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        torch.save(checkpoint, filepath)

    @classmethod
    def load_model(cls, checkpoint_path: Path, device: Optional[torch.device] = None) -> Tuple[ArisLM, int]:
        """Loads ArisLM and returns (model, cumulative_step)."""
        from ..configs.model_config import ArisConfig
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        cfg_dict = checkpoint["model_config"]

        config = ArisConfig(
            vocab_size=cfg_dict["vocab_size"],
            max_seq_len=cfg_dict["max_seq_len"],
            embed_dim=cfg_dict["embed_dim"],
            num_layers=cfg_dict["num_layers"],
            num_query_heads=cfg_dict["num_query_heads"],
            num_kv_heads=cfg_dict["num_kv_heads"],
            ff_dim=cfg_dict.get("ff_dim"),
        )
        model = ArisLM(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        return model, checkpoint.get("step", 0)
