"""Loss functions for Phase 3 (C-MLP) disease classification.

Key design decisions (from phase_3_4_mlp_update.md §7):
- Masked BCE: entries with label == -100 are excluded from loss.
- pos_weight log-scale (RADAR formula): α_i = log(1 + N / w_i).
- Optional focal loss: down-weight easy examples (γ=2).
- Imbalance handling is the #1 lever for quality.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import MASK_VALUE, NUM_DISEASES


def compute_pos_weight_log(
    labels: torch.Tensor,
    mask_value: int = MASK_VALUE,
) -> torch.Tensor:
    """Compute RADAR-style log pos_weight from training labels.

    Formula: α_i = log(1 + N_valid_i / pos_count_i)
    where N_valid_i = number of non-masked entries for disease i.

    Parameters
    ----------
    labels : Tensor [N, 14]
        All training labels (may contain mask_value for unknowns).

    Returns
    -------
    pos_weight : Tensor [14]
    """
    valid = labels != mask_value                    # [N, 14]
    pos = (labels == 1) & valid                     # [N, 14]
    n_valid = valid.float().sum(dim=0).clamp(min=1) # [14]
    n_pos = pos.float().sum(dim=0).clamp(min=1)     # [14]
    return torch.log(1.0 + n_valid / n_pos)


def compute_pos_weight_inverse(
    labels: torch.Tensor,
    mask_value: int = MASK_VALUE,
) -> torch.Tensor:
    """Simple inverse-frequency pos_weight."""
    valid = labels != mask_value
    pos = (labels == 1) & valid
    n_valid = valid.float().sum(dim=0).clamp(min=1)
    n_pos = pos.float().sum(dim=0).clamp(min=1)
    n_neg = (n_valid - n_pos).clamp(min=1)
    return n_neg / n_pos


class MaskedBCELoss(nn.Module):
    """BCE with logits, ignoring entries where target == mask_value.

    Parameters
    ----------
    pos_weight : Optional[Tensor[14]]
        Per-class positive weight.
    mask_value : int
        Label value to ignore (default -100).
    """

    def __init__(
        self,
        pos_weight: torch.Tensor | None = None,
        mask_value: int = MASK_VALUE,
    ):
        super().__init__()
        self.mask_value = mask_value
        # Register as buffer so it moves with the model to device.
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight.float())
        else:
            self.pos_weight = None

    def forward(
        self,
        logits: torch.Tensor,   # [B, 14] or [B, 29, 14]
        targets: torch.Tensor,  # same shape, values in {0, 1, mask_value}
    ) -> torch.Tensor:
        mask = targets != self.mask_value
        if not mask.any():
            return logits.new_tensor(0.0, requires_grad=True)

        logits_m = logits[mask]
        targets_m = targets[mask].float()

        pw = self.pos_weight
        if pw is not None:
            # Expand pos_weight to match the last dim
            # mask flattens; need to broadcast carefully.
            # Simpler: compute full loss then mask.
            pass

        # Full element-wise loss, then mask
        loss_full = F.binary_cross_entropy_with_logits(
            logits, targets.float().clamp(0, 1),  # clamp mask_value → 0 temporarily
            pos_weight=pw,
            reduction="none",
        )
        loss_full = loss_full * mask.float()
        return loss_full.sum() / mask.float().sum()


class FocalBCELoss(nn.Module):
    """Sigmoid focal loss with optional alpha-balancing and pos_weight.

    Focal loss = -α_t * (1 - p_t)^γ * log(p_t)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        pos_weight: torch.Tensor | None = None,
        mask_value: int = MASK_VALUE,
    ):
        super().__init__()
        self.gamma = gamma
        self.mask_value = mask_value
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight.float())
        else:
            self.pos_weight = None

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        mask = targets != self.mask_value
        if not mask.any():
            return logits.new_tensor(0.0, requires_grad=True)

        targets_f = targets.float().clamp(0, 1)
        p = torch.sigmoid(logits)
        p_t = p * targets_f + (1 - p) * (1 - targets_f)
        focal_weight = (1 - p_t) ** self.gamma

        # BCE element-wise
        bce = F.binary_cross_entropy_with_logits(
            logits, targets_f,
            reduction="none",
        )

        # Apply pos_weight manually
        if self.pos_weight is not None:
            pw = self.pos_weight
            alpha_t = pw * targets_f + (1 - targets_f)
            bce = bce * alpha_t

        loss = focal_weight * bce
        loss = loss * mask.float()
        return loss.sum() / mask.float().sum()


def build_loss(
    loss_type: str = "bce",
    pos_weight: torch.Tensor | None = None,
    focal_gamma: float = 2.0,
    mask_value: int = MASK_VALUE,
) -> nn.Module:
    """Factory for disease classification loss."""
    if loss_type == "bce":
        return MaskedBCELoss(pos_weight=pos_weight, mask_value=mask_value)
    elif loss_type == "focal":
        return FocalBCELoss(
            gamma=focal_gamma,
            pos_weight=pos_weight,
            mask_value=mask_value,
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type!r}")
