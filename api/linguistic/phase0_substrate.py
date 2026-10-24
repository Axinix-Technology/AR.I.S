"""
AR.I.S — Phase 0 PyTorch Neural Sequence Substrate

Constitutional Invariants:
  - PURE TENSOR SUBSTRATE:
    Consumes discrete Token ID tensors produced by the canonical tokenizer.
    Does NOT accept raw strings or hand-crafted heuristic features.
  - MEASURABLE REPRESENTATION & GEOMETRY:
    Combines learnable discrete token embeddings with explicit positional embeddings.
    Transforms sequence representations through a transparent multi-head self-attention block.
  - TRANSPARENT OBSERVABILITY:
    Exposes all intermediate tensor representations, embedding geometries, attention maps,
    and gradients for scientific measurement.
"""

from __future__ import annotations
import math
from typing import Dict, Any, Optional, Tuple, List
import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    """A standard Transformer encoder block with Multi-Head Self-Attention and Feed-Forward Network."""

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.0):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_out, weights = self.attention(
            x, x, x,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        x = self.norm1(x + attn_out)
        ff_out = self.ffn(x)
        x = self.norm2(x + ff_out)
        return x, weights


class ArisPhase0Model(nn.Module):
    """
    Minimal PyTorch sequence substrate for Phase 0.
    Maps discrete token sequences -> continuous embeddings -> relational self-attention -> predictions.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 32,
        max_seq_len: int = 16,
        num_heads: int = 2,
        ff_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.num_layers = num_layers

        # 1. Discrete Token Embedding Table: E_tok in R^(vocab_size x embed_dim)
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        # 2. Discrete Positional Embedding Table: E_pos in R^(max_seq_len x embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)

        self.norm_input = nn.LayerNorm(embed_dim)

        # 3. Sequence Relational Transformation Layers (Stacked Transformer Blocks)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        # 4. Output Projection Head to Discrete Vocabulary Logits
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Tie weights: lm_head shares weights with token_embedding for tighter representation learning
        self.lm_head.weight = self.token_embedding.weight

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass over token IDs.
        Args:
          input_ids: LongTensor of shape (batch_size, seq_len)
          attention_mask: Optional LongTensor of shape (batch_size, seq_len) (1 = active, 0 = pad)
        """
        batch_size, seq_len = input_ids.shape
        assert seq_len <= self.max_seq_len, (
            f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}"
        )

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)

        tok_emb = self.token_embedding(input_ids)          # (B, L, D)
        pos_emb = self.position_embedding(positions)       # (B, L, D)

        x = self.norm_input(tok_emb + pos_emb)            # (B, L, D)

        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)

        all_attn_weights = []
        for block in self.blocks:
            x, weights = block(x, key_padding_mask=key_padding_mask)
            all_attn_weights.append(weights)

        logits = self.lm_head(x)                          # (B, L, vocab_size)

        return {
            "logits": logits,
            "hidden_states": x,
            "token_embeddings": tok_emb,
            "pos_embeddings": pos_emb,
            "attention_weights": all_attn_weights[-1],
        }
