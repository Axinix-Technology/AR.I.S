"""
AR.I.S — Causal Autoregressive Transformer Language Model (ArisLM)
Purpose:
  A clean, production-grade PyTorch Causal Transformer Decoder for autoregressive
  language modeling, conversational chat, and instruction following.
  Implements multi-head causal self-attention, pre-layer normalization,
  tied token embeddings, and nucleus/top-k generation.
"""

from __future__ import annotations
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArisCausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention block using PyTorch 2.0 scaled dot-product attention.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Key, Query, Value projections in a single linear layer
        self.c_attn = nn.Linear(embed_dim, 3 * embed_dim)
        # Output projection
        self.c_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        # Compute Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.embed_dim, dim=2)

        # Reshape for multi-head attention: (B, num_heads, T, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # PyTorch 2.0 fast causal attention with flash/memory efficient backend
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        # Recombine heads: (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class ArisMLP(nn.Module):
    """
    Feed-Forward Network with GELU non-linearity.
    """

    def __init__(self, embed_dim: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.c_fc = nn.Linear(embed_dim, ff_dim)
        self.act = nn.GELU()
        self.c_proj = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        return self.dropout(x)


class ArisTransformerBlock(nn.Module):
    """
    Standard Pre-LayerNorm Transformer Decoder Block.
    """

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = ArisCausalSelfAttention(embed_dim, num_heads, dropout)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.mlp = ArisMLP(embed_dim, ff_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN residual connections
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class ArisLM(nn.Module):
    """
    AR.I.S Autoregressive Foundation Language Model.
    """

    def __init__(
        self,
        vocab_size: int = 8192,
        max_seq_len: int = 512,
        embed_dim: int = 384,
        num_heads: int = 6,
        num_layers: int = 6,
        ff_dim: int = 1536,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim

        # Token & Positional Embeddings
        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)
        self.drop = nn.Dropout(dropout)

        # Transformer Decoder Stack
        self.blocks = nn.ModuleList([
            ArisTransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        # Final LayerNorm & LM Head (projecting hidden states to vocabulary logits)
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Weight tying (Press & Wolf, 2017)
        self.tok_emb.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        Args:
            input_ids: (Batch, SeqLen) LongTensor
            targets: (Batch, SeqLen) Optional target token IDs for computing loss
        Returns:
            (logits, loss)
        """
        device = input_ids.device
        b, t = input_ids.size()
        assert t <= self.max_seq_len, f"Sequence length {t} exceeds maximum {self.max_seq_len}"

        positions = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)

        # Token + positional embeddings
        x = self.tok_emb(input_ids) + self.pos_emb(positions)
        x = self.drop(x)

        # Pass through decoder blocks
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # Flatten tokens for Cross-Entropy loss calculation
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1), ignore_index=-100)

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9,
        eos_id: int = 3,
    ) -> torch.Tensor:
        """
        Autoregressive next-token generation with Top-K and Nucleus (Top-P) sampling.
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop to context window if needed
            idx_cond = input_ids if input_ids.size(1) <= self.max_seq_len else input_ids[:, -self.max_seq_len:]

            logits, _ = self(idx_cond)
            # Take logits of the last token
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            # Optional Top-K truncation
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            # Optional Top-P (Nucleus) truncation
            if 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift mask right to keep at least first token above threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float("Inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat((input_ids, next_token), dim=1)

            if next_token.item() == eos_id:
                break

        return input_ids
