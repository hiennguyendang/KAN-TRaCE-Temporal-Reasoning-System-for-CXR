"""Neck: Linear projection 512→128 + LayerNorm + GELU.

The neck produces the shared representation `region_feat[29,128]` used by both
the disease head and T-MLP.  Justification for keeping the neck (even though
MLP can eat 512 directly): it keeps T-MLP input compact (412 vs 1564 dim)
and provides normalisation stability.
"""

from __future__ import annotations

import torch.nn as nn

from .constants import ENCODER_DIM, PROJ_DIM


class Neck(nn.Module):
    """Project encoder features down to a compact representation.

    ``Linear(encoder_dim → proj_dim) → LayerNorm → GELU``
    """

    def __init__(
        self,
        encoder_dim: int = ENCODER_DIM,
        proj_dim: int = PROJ_DIM,
    ):
        super().__init__()
        self.proj = nn.Linear(encoder_dim, proj_dim)
        self.ln = nn.LayerNorm(proj_dim)
        self.act = nn.GELU()

    def forward(self, x):
        """x: [..., encoder_dim] → [..., proj_dim]."""
        return self.act(self.ln(self.proj(x)))
