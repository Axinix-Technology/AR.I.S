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
    vocab_size: int = 4096
    max_seq_len: int = 384
    
    # Model Dimensions (Gen 2 Cloud Architecture: ~24.6M parameters, 2x Doubling)
    embed_dim: int = 512
    num_layers: int = 8
    num_query_heads: int = 8
    num_kv_heads: int = 2          # GQA (4 query heads per KV head)
    ff_dim: Optional[int] = None   # SwiGLU intermediate dimension (default: 8/3 * embed_dim)
    tie_word_embeddings: bool = True  # Weight tying between tok_emb and lm_head
    
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

    def expected_param_count(self, tied_embeddings: Optional[bool] = None) -> int:
        """
        Derives exact parameter count mathematically from config architecture.
        Accounts for tied embeddings (tok_emb tied to lm_head).
        """
        if tied_embeddings is None:
            tied_embeddings = self.tie_word_embeddings
        head_dim = self.head_dim
        tok_emb_params = self.vocab_size * self.embed_dim
        lm_head_params = 0 if tied_embeddings else (self.vocab_size * self.embed_dim)
        norm_f_params = self.embed_dim

        # Per-block parameter calculation:
        # 1. attn_norm: embed_dim
        # 2. q_proj: embed_dim * (num_query_heads * head_dim)
        # 3. k_proj: embed_dim * (num_kv_heads * head_dim)
        # 4. v_proj: embed_dim * (num_kv_heads * head_dim)
        # 5. out_proj: (num_query_heads * head_dim) * embed_dim
        # 6. mlp_norm: embed_dim
        # 7. w_gate: embed_dim * ff_dim
        # 8. w_up: embed_dim * ff_dim
        # 9. w_down: ff_dim * embed_dim
        block_params = (
            self.embed_dim  # attn_norm
            + self.embed_dim * (self.num_query_heads * head_dim)  # q_proj
            + self.embed_dim * (self.num_kv_heads * head_dim)     # k_proj
            + self.embed_dim * (self.num_kv_heads * head_dim)     # v_proj
            + (self.num_query_heads * head_dim) * self.embed_dim  # out_proj
            + self.embed_dim  # mlp_norm
            + self.embed_dim * self.ff_dim  # w_gate
            + self.embed_dim * self.ff_dim  # w_up
            + self.ff_dim * self.embed_dim  # w_down
        )

        return tok_emb_params + lm_head_params + norm_f_params + (self.num_layers * block_params)

    def expected_state_dict_count(self) -> int:
        """
        Derives total serialized tensor element count in state_dict.
        When weights are tied, PyTorch stores both tok_emb.weight and lm_head.weight.
        """
        return self.expected_param_count(tied_embeddings=False)

