"""Region pooling: Attention-pool and mean-ROI-pool.

Attention-pool replaces the naive mean-ROI-pool from the original design:
- 29 learnable region queries attend to the 196 grid cells, masked by bbox.
- Produces region_pooled [B,29,D], region_mask [B,29], alpha [B,29,196].
- alpha is a *faithful* attribution (it IS the pooling weight, not an approx).

Mean-ROI-pool is kept as ablation baseline.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import ENCODER_DIM, GRID_H, GRID_W, NUM_PATCHES, NUM_REGIONS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _boxes_to_mask(
    boxes: torch.Tensor,       # [B, 29, 4]  normalised x1y1x2y2 (0–1)
    grid_h: int = GRID_H,
    grid_w: int = GRID_W,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert normalised boxes to a binary cell-inclusion mask.

    Returns
    -------
    cell_mask : BoolTensor [B, 29, grid_h*grid_w]
        True if the centre of grid cell c falls inside box r.
    region_mask : BoolTensor [B, 29]
        True if box r is valid (not a sentinel (0,0,0,0) and has positive area).
    """
    B = boxes.size(0)
    device = boxes.device

    # Centre coordinates of each grid cell, normalised to [0,1].
    cy = (torch.arange(grid_h, device=device, dtype=boxes.dtype) + 0.5) / grid_h
    cx = (torch.arange(grid_w, device=device, dtype=boxes.dtype) + 0.5) / grid_w
    # [grid_h, grid_w] → flatten to [HW]
    grid_cy, grid_cx = torch.meshgrid(cy, cx, indexing="ij")
    grid_cx = grid_cx.reshape(1, 1, -1)  # [1, 1, HW]
    grid_cy = grid_cy.reshape(1, 1, -1)

    x1 = boxes[:, :, 0:1]  # [B, 29, 1]
    y1 = boxes[:, :, 1:2]
    x2 = boxes[:, :, 2:3]
    y2 = boxes[:, :, 3:4]

    in_x = (grid_cx >= x1) & (grid_cx <= x2)  # [B, 29, HW]
    in_y = (grid_cy >= y1) & (grid_cy <= y2)
    cell_mask = in_x & in_y                     # [B, 29, HW]

    # Region valid = box has positive width AND height AND is not sentinel.
    w = (x2 - x1).squeeze(-1)  # [B, 29]
    h = (y2 - y1).squeeze(-1)
    region_mask = (w > 1e-6) & (h > 1e-6)  # [B, 29]

    return cell_mask, region_mask


# ---------------------------------------------------------------------------
# Attention-pool
# ---------------------------------------------------------------------------

class AttentionPool(nn.Module):
    """Cross-attention region pooling (DETR-style).

    29 learnable region queries attend to 196 grid-cell features, masked by
    bounding box.  Single-head for interpretability of alpha.

    Parameters
    ----------
    d_model : int
        Feature dimension of encoder patches (default 512).
    num_regions : int
        Number of anatomical regions (29).
    num_patches : int
        Number of grid cells (196 = 14×14).
    dropout : float
        Dropout on attention weights (entropy regularisation).
    """

    def __init__(
        self,
        d_model: int = ENCODER_DIM,
        num_regions: int = NUM_REGIONS,
        num_patches: int = NUM_PATCHES,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_regions = num_regions
        self.scale = math.sqrt(d_model)

        # Learnable region queries  [29, d]
        self.query = nn.Parameter(torch.randn(num_regions, d_model) * 0.02)

        # Key / Value projections for grid cells
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        self.attn_drop = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        """Initialise so that attention starts ≈ uniform (like mean-pool)."""
        # Small Q so scores start near 0 → softmax ≈ uniform.
        nn.init.normal_(self.query, std=0.02)
        # K/V near-identity: small perturbation around identity-like init.
        for proj in [self.W_k, self.W_v]:
            nn.init.xavier_uniform_(proj.weight, gain=0.1)

    def forward(
        self,
        feats: torch.Tensor,   # [B, 196, 512]
        boxes: torch.Tensor,   # [B, 29, 4]  normalised x1y1x2y2
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        region_pooled : [B, 29, D]
            Pooled features per region (zeroed for invalid regions).
        region_mask : [B, 29]  bool
            True where region is valid.
        alpha : [B, 29, 196]
            Attention weights (faithful attribution within each region).
        """
        B = feats.size(0)

        K = self.W_k(feats)     # [B, 196, D]
        V = self.W_v(feats)     # [B, 196, D]
        Q = self.query.unsqueeze(0).expand(B, -1, -1)  # [B, 29, D]

        # Scaled dot-product scores  [B, 29, 196]
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale

        # Mask: only attend to cells inside the box
        cell_mask, region_mask = _boxes_to_mask(boxes)  # [B,29,196], [B,29]

        # Where cell_mask is False, set score to -inf (before softmax)
        scores = scores.masked_fill(~cell_mask, float("-inf"))

        # Guard: if a region has NO valid cells, all scores are -inf.
        # softmax(-inf, ...) = nan. Replace with 0 after softmax.
        alpha = F.softmax(scores, dim=-1)  # [B, 29, 196]
        alpha = alpha.nan_to_num(0.0)      # NaN → 0 for empty regions
        alpha = self.attn_drop(alpha)

        # Weighted sum  [B, 29, D]
        region_pooled = torch.bmm(alpha, V)

        # Zero out invalid regions
        region_pooled = region_pooled * region_mask.unsqueeze(-1).float()

        # LayerNorm
        region_pooled = self.ln(region_pooled)

        return region_pooled, region_mask, alpha


# ---------------------------------------------------------------------------
# Mean-ROI-pool (ablation baseline)
# ---------------------------------------------------------------------------

class MeanROIPool(nn.Module):
    """Simple mean pooling within each bounding box (baseline / fallback)."""

    def __init__(self, d_model: int = ENCODER_DIM):
        super().__init__()
        self.d_model = d_model
        self.ln = nn.LayerNorm(d_model)

    def forward(
        self,
        feats: torch.Tensor,   # [B, 196, D]
        boxes: torch.Tensor,   # [B, 29, 4]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cell_mask, region_mask = _boxes_to_mask(boxes)  # [B,29,196], [B,29]

        # Uniform alpha within box
        counts = cell_mask.float().sum(dim=-1, keepdim=True).clamp(min=1.0)  # [B,29,1]
        alpha = cell_mask.float() / counts  # [B, 29, 196]

        # Weighted sum (mean inside box)
        region_pooled = torch.bmm(alpha, feats)  # [B, 29, D]
        region_pooled = region_pooled * region_mask.unsqueeze(-1).float()
        region_pooled = self.ln(region_pooled)

        return region_pooled, region_mask, alpha


def build_pool(pool_type: str = "attn", **kwargs) -> nn.Module:
    """Factory: 'attn' → AttentionPool, 'mean' → MeanROIPool."""
    if pool_type == "attn":
        return AttentionPool(**kwargs)
    elif pool_type == "mean":
        return MeanROIPool(**{k: v for k, v in kwargs.items() if k == "d_model"})
    else:
        raise ValueError(f"Unknown pool type: {pool_type!r}")
