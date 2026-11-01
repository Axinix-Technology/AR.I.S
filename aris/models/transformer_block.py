"""
AR.I.S — Modern Transformer Decoder Block
Architecture:
  x -> RMSNorm -> RoPE GQA Attention -> Residual (+)
    -> RMSNorm -> SwiGLU Gated MLP    -> Residual (+)
"""

from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
from .normalization import ArisRMSNorm
from .attention import ArisGQAWithKVCache
from .mlp import ArisSwiGLU


class ArisTransformerBlock(nn.Module):
    """
    Decoder block with Pre-RMSNorm, RoPE Grouped-Query Attention, and SwiGLU.
    """

    def __init__(
        self,
        embed_dim: int,
        num_query_heads: int,
        num_kv_heads: int,
        ff_dim: int,
        rms_norm_eps: float = 1e-6,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attn_norm = ArisRMSNorm(embed_dim, eps=rms_norm_eps)
        self.attn = ArisGQAWithKVCache(
            embed_dim=embed_dim,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            dropout=dropout,
        )
        self.mlp_norm = ArisRMSNorm(embed_dim, eps=rms_norm_eps)
        self.mlp = ArisSwiGLU(embed_dim=embed_dim, ff_dim=ff_dim)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Pre-RMSNorm Attention
        attn_out, new_cache = self.attn(
            self.attn_norm(x),
            cos=cos,
            sin=sin,
            kv_cache=kv_cache,
            use_cache=use_cache,
        )
        x = x + attn_out

        # Pre-RMSNorm SwiGLU
        x = x + self.mlp(self.mlp_norm(x))
        return x, new_cache

