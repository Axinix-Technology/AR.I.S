"""
AR.I.S — Causal Autoregressive Language Model (ArisLM 2026)
Architecture Summary:
  - Token Embeddings (with weight tying to LM Head)
  - RoPE (Rotary Position Embeddings)
  - Pre-RMSNorm Layers
  - GQA (Grouped-Query Attention with 3:1 head ratio)
  - SwiGLU Gated Feed-Forward Networks
  - Dynamic KV Cache for O(1) incremental generation
"""

from __future__ import annotations
from typing import Optional, Tuple, List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..configs.model_config import ArisConfig
from .rope import ArisRotaryEmbedding
from .normalization import ArisRMSNorm
from .transformer_block import ArisTransformerBlock


class ArisLM(nn.Module):
    """
    Modern 2026 Causal Decoder-Only Foundation Language Model.
    """

    def __init__(self, config: Optional[ArisConfig] = None):
        super().__init__()
        self.config = config or ArisConfig()

        # Token Embedding (No absolute positional embeddings — RoPE handles position)
        self.tok_emb = nn.Embedding(self.config.vocab_size, self.config.embed_dim)

        # RoPE embedding provider
        self.rope = ArisRotaryEmbedding(
            dim=self.config.head_dim,
            max_seq_len=self.config.max_seq_len,
            base=self.config.rope_theta,
        )

        # Stack of Transformer Decoder Blocks
        self.blocks = nn.ModuleList([
            ArisTransformerBlock(
                embed_dim=self.config.embed_dim,
                num_query_heads=self.config.num_query_heads,
                num_kv_heads=self.config.num_kv_heads,
                ff_dim=self.config.ff_dim,
                rms_norm_eps=self.config.rms_norm_eps,
                dropout=self.config.dropout,
            )
            for _ in range(self.config.num_layers)
        ])

        # Final RMSNorm & LM Head
        self.norm_f = ArisRMSNorm(self.config.embed_dim, eps=self.config.rms_norm_eps)
        self.lm_head = nn.Linear(self.config.embed_dim, self.config.vocab_size, bias=False)

        # Weight Tying (Embeddings tied to LM Head)
        self.tok_emb.weight = self.lm_head.weight

        # Weight Initialization
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
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        offset: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Forward pass with optional KV Cache and targets.
        """
        B, T = input_ids.size()
        cos, sin = self.rope(seq_len=T, offset=offset)

        x = self.tok_emb(input_ids)

        new_kv_caches = [] if use_cache else None

        for idx, block in enumerate(self.blocks):
            layer_cache = kv_caches[idx] if kv_caches is not None else None
            x, updated_cache = block(
                x,
                cos=cos,
                sin=sin,
                kv_cache=layer_cache,
                use_cache=use_cache,
            )
            if use_cache and updated_cache is not None:
                new_kv_caches.append(updated_cache)

        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # Shift so that tokens < n predict n (standard causal next-token prediction)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_targets = targets[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_targets.view(-1),
                ignore_index=-100,
            )

        return logits, loss, new_kv_caches

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

