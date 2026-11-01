"""
AR.I.S — Rotary Positional Embeddings (RoPE)
Purpose:
  Encodes relative token positions by rotating Query and Key vectors in complex space.
  Enables extrapolation to longer context windows and seamless incremental decoding with KV Caches.
"""

from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn


class ArisRotaryEmbedding(nn.Module):
    """
    Precomputes and applies rotary position embeddings (Su et al., 2021).
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Inverse frequency: 1.0 / (base ** (2i / dim))
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cosine and sine tables for max_seq_len
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)  # (max_seq_len, dim // 2)
        emb = torch.cat((freqs, freqs), dim=-1) # (max_seq_len, dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _extend_cache(self, required_len: int, device: torch.device):
        """Dynamically extends the cosine/sine precomputed cache to support arbitrary context lengths."""
        new_max = max(required_len, self.max_seq_len * 2)
        t = torch.arange(new_max, dtype=torch.float32, device=device)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()
        self.sin_cached = emb.sin()
        self.max_seq_len = new_max

    def forward(self, seq_len: int, offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns cos and sin for positions [offset : offset + seq_len]."""
        required = offset + seq_len
        if required > self.max_seq_len:
            self._extend_cache(required, self.inv_freq.device)
        cos = self.cos_cached[offset : offset + seq_len]
        sin = self.sin_cached[offset : offset + seq_len]
        return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Rotary Position Embedding to Query and Key tensors.
    q: (Batch, Heads, SeqLen, HeadDim)
    k: (Batch, KVHeads, SeqLen, HeadDim)
    cos, sin: (SeqLen, HeadDim)
    """
    # Reshape cos and sin for broadcasting: (1, 1, SeqLen, HeadDim)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed

