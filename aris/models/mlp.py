"""
AR.I.S — SwiGLU Gated Feed-Forward Network
Purpose:
  Modern replacement for standard GELU MLP (Shazeer, 2020).
  Computes element-wise product of a SiLU (Swish) gated linear projection and a value projection:
  SwiGLU(x) = (SiLU(x * W_gate) * (x * W_up)) * W_down
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArisSwiGLU(nn.Module):
    """
    Gated SwiGLU Feed-Forward Network.
    """

    def __init__(self, embed_dim: int, ff_dim: int):
        super().__init__()
        # Gate and Up projections in separate linear maps
        self.w_gate = nn.Linear(embed_dim, ff_dim, bias=False)
        self.w_up = nn.Linear(embed_dim, ff_dim, bias=False)
        self.w_down = nn.Linear(ff_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.silu(w_gate(x)) * w_up(x)
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

