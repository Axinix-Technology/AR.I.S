"""
AR.I.S — Grouped-Query Attention (GQA) with KV-Cache
Purpose:
  Modern attention module supporting:
  - Grouped-Query Attention (GQA): Multiple query heads share one KV head.
  - FlashAttention / PyTorch SDPA enable_gqa integration.
  - Efficient dynamic KV Caching for fast, O(1) incremental generation.
"""

from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .rope import apply_rotary_pos_emb


class ArisGQAWithKVCache(nn.Module):
    """
    Grouped-Query Attention with Rotary Embeddings and dynamic KV Caching.
    """

    def __init__(
        self,
        embed_dim: int,
        num_query_heads: int,
        num_kv_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_query_heads
        self.num_queries_per_kv = num_query_heads // num_kv_heads
        self.dropout = dropout

        # Projections: Q uses full heads, K and V use reduced KV heads
        self.q_proj = nn.Linear(embed_dim, self.num_query_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: Input tensor of shape (Batch, SeqLen, EmbedDim)
            cos, sin: Precomputed rotary embeddings
            kv_cache: Optional tuple of (cached_k, cached_v)
            use_cache: If True, returns updated (k, v) cache
        """
        B, T, _ = x.size()

        # Project Q, K, V
        q = self.q_proj(x).view(B, T, self.num_query_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Update or retrieve KV Cache
        if kv_cache is not None:
            prev_k, prev_v = kv_cache
            k = torch.cat([prev_k, k], dim=-2)
            v = torch.cat([prev_v, v], dim=-2)

        new_kv_cache = (k, v) if use_cache else None

        # Repeat KV heads for GQA if needed by PyTorch SDPA
        if self.num_queries_per_kv > 1:
            k_expanded = k.repeat_interleave(self.num_queries_per_kv, dim=1)
            v_expanded = v.repeat_interleave(self.num_queries_per_kv, dim=1)
        else:
            k_expanded = k
            v_expanded = v

        # During incremental decoding (T=1 with cache), causality is maintained automatically
        is_causal = (T > 1) and (kv_cache is None)

        out = F.scaled_dot_product_attention(
            q, k_expanded, v_expanded,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        # Recombine heads and project
        out = out.transpose(1, 2).contiguous().view(B, T, self.embed_dim)
        return self.out_proj(out), new_kv_cache

