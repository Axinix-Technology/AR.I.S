"""
AR.I.S — Modern LLM Configuration (2026 Architecture)
Purpose:
  Configurable hyperparameters for RoPE, GQA, SwiGLU, RMSNorm, and KV Caching.
  Optimized for CPU efficiency and scalable foundation pre-training.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ArisConfig:
    # Vocabulary & Sequence
    vocab_size: int = 8192
    max_seq_len: int = 512
    
    # Model Dimensions (Sweet spot for CPU: ~15M - 30M parameters)
    embed_dim: int = 384
    num_layers: int = 6
    num_query_heads: int = 6
    num_kv_heads: int = 2          # GQA (3 query heads per KV head)
    ff_dim: Optional[int] = None   # SwiGLU intermediate dimension (default: 8/3 * embed_dim)
    
    # RoPE (Rotary Position Embeddings)
    rope_theta: float = 10000.0
    
    # Normalization & Regularization
    rms_norm_eps: float = 1e-6
    dropout: float = 0.0           # Modern LLMs use 0.0 dropout during pretraining
    
    # Special Token IDs
    pad_token_id: int = 0
    unk_token_id: int = 1
    bos_token_id: int = 2
    eos_token_id: int = 3
    
    def __post_init__(self):
        assert self.embed_dim % self.num_query_heads == 0, "embed_dim must be divisible by num_query_heads"
        assert self.num_query_heads % self.num_kv_heads == 0, "num_query_heads must be divisible by num_kv_heads"
        
        # SwiGLU standard dimension scaling: ~2/3 of 4x embed_dim rounded to multiple of 64
        if self.ff_dim is None:
            hidden = int(2 * (4 * self.embed_dim) / 3)
            self.ff_dim = ((hidden + 63) // 64) * 64

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_query_heads

