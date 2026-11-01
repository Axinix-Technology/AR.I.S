"""
AR.I.S — Root Mean Square Normalization (RMSNorm)
Purpose:
  Modern replacement for LayerNorm used in LLaMA, Mistral, and Gemma.
  Achieves numerical stability and higher computational efficiency by
  enforcing invariant scaling without calculating mean activation offsets.
"""

from __future__ import annotations
import torch
import torch.nn as nn


class ArisRMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    formula: y = (x / sqrt(mean(x^2) + eps)) * weight
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate root mean square over the last dimension in float32 for stability
        input_dtype = x.dtype
        x_fp32 = x.float()
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        normed = x_fp32 * torch.rsqrt(variance + self.eps)
        return (normed.to(input_dtype)) * self.weight

