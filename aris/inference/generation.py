"""
AR.I.S — Production Inference Engine with KV Cache
Features:
  - Vectorized KV-Cache generation (O(1) incremental step computation)
  - True batch generation with per-sequence EOS stopping
  - Nucleus (Top-P) and Top-K sampling with temperature control
"""

from __future__ import annotations
from typing import Optional, List
import torch
import torch.nn.functional as F
from ..models.aris_lm import ArisLM


class ArisInferenceEngine:
    """
    High-performance inference runner for ArisLM.
    """

    @classmethod
    @torch.no_grad()
    def generate(
        cls,
        model: ArisLM,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        eos_id: int = 3,
        pad_id: int = 0,
    ) -> torch.Tensor:
        """
        Generates tokens using KV Caching and per-sequence EOS tracking.
        Args:
            input_ids: (Batch, PromptLen) LongTensor
        Returns:
            (Batch, PromptLen + GeneratedLen) LongTensor
        """
        was_training = model.training
        model.eval()

        device = input_ids.device
        batch_size, prompt_len = input_ids.shape

        # Per-sequence finished mask
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # 1. Prefill Step: Process full prompt and build initial KV Caches
        logits, _, kv_caches = model(
            input_ids=input_ids,
            use_cache=True,
            offset=0,
        )

        # Next token from prefill
        next_token_logits = logits[:, -1, :] / max(temperature, 1e-5)
        next_tokens = cls._sample_next_token(next_token_logits, top_k=top_k, top_p=top_p)

        # Update finished states
        finished = finished | (next_tokens.squeeze(-1) == eos_id)

        generated_ids = [input_ids, next_tokens]
        current_len = prompt_len

        # 2. Decoding Steps with KV Cache: Process only 1 token per step (O(1) compute)
        for _ in range(max_new_tokens - 1):
            if finished.all():
                break

            current_len += 1
            step_logits, _, kv_caches = model(
                input_ids=next_tokens,
                kv_caches=kv_caches,
                use_cache=True,
                offset=current_len - 1,
            )

            next_token_logits = step_logits[:, -1, :] / max(temperature, 1e-5)
            next_tokens = cls._sample_next_token(next_token_logits, top_k=top_k, top_p=top_p)

            # Mask out tokens for already finished sequences
            next_tokens = torch.where(finished.unsqueeze(-1), torch.full_like(next_tokens, pad_id), next_tokens)
            finished = finished | (next_tokens.squeeze(-1) == eos_id)

            generated_ids.append(next_tokens)

        if was_training:
            model.train()

        return torch.cat(generated_ids, dim=1)

    @staticmethod
    def _sample_next_token(logits: torch.Tensor, top_k: int = 40, top_p: float = 0.9) -> torch.Tensor:
        """Applies Top-K and Nucleus (Top-P) sampling."""
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")

        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -float("Inf")

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

