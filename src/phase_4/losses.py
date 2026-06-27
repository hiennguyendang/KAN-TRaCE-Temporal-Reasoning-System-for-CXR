"""Loss functions for Phase 4 T-MLP (3-class progression).

Masked CE with class-weight (inverse frequency or effective number).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from phase_3.constants import MASK_VALUE, NUM_PROG


def compute_class_weight_inverse(
    labels: torch.Tensor,
    num_classes: int = NUM_PROG,
    mask_value: int = MASK_VALUE,
) -> torch.Tensor:
    """Inverse-frequency class weights for 3-class CE.

    Parameters
    ----------
    labels : Tensor  (any shape, values in {0,1,2, mask_value})

    Returns
    -------
    weights : Tensor [num_classes]
    """
    flat = labels.view(-1)
    valid = flat[flat != mask_value]
    if valid.numel() == 0:
        return torch.ones(num_classes)

    counts = torch.bincount(valid, minlength=num_classes).float().clamp(min=1)
    total = counts.sum()
    weights = total / (num_classes * counts)
    return weights


def compute_class_weight_effective(
    labels: torch.Tensor,
    num_classes: int = NUM_PROG,
    beta: float = 0.9999,
    mask_value: int = MASK_VALUE,
) -> torch.Tensor:
    """Effective number class weights (Cui et al. 2019)."""
    flat = labels.view(-1)
    valid = flat[flat != mask_value]
    if valid.numel() == 0:
        return torch.ones(num_classes)

    counts = torch.bincount(valid, minlength=num_classes).float().clamp(min=1)
    effective = 1.0 - beta ** counts
    weights = (1.0 - beta) / effective
    weights = weights / weights.sum() * num_classes  # normalise
    return weights


class MaskedCELoss(nn.Module):
    """Cross-entropy loss for 3-class progression, masking invalid entries.

    Parameters
    ----------
    weight : Optional[Tensor[3]]
        Per-class weight.
    mask_value : int
        Ignore index (default -100 = PyTorch native ignore_index).
    """

    def __init__(
        self,
        weight: torch.Tensor | None = None,
        mask_value: int = MASK_VALUE,
    ):
        super().__init__()
        self.mask_value = mask_value
        if weight is not None:
            self.register_buffer("weight", weight.float())
        else:
            self.weight = None

    def forward(
        self,
        logits: torch.Tensor,   # [B, 29, 14, 3]
        targets: torch.Tensor,  # [B, 29, 14]  values in {0,1,2, mask_value}
        pair_mask: torch.Tensor | None = None,  # [B, 29, 14] bool
    ) -> torch.Tensor:
        B, R, D, C = logits.shape

        # Flatten for CE
        logits_flat = logits.reshape(-1, C)  # [B*29*14, 3]
        targets_flat = targets.reshape(-1)   # [B*29*14]

        # Additionally mask where pair_mask is False
        if pair_mask is not None:
            invalid = ~pair_mask.reshape(-1)
            targets_flat = targets_flat.clone()
            targets_flat[invalid] = self.mask_value

        return F.cross_entropy(
            logits_flat, targets_flat,
            weight=self.weight,
            ignore_index=self.mask_value,
        )


def build_loss(
    weight_strategy: str = "inverse",
    train_labels: torch.Tensor | None = None,
) -> MaskedCELoss:
    """Build T-MLP loss with optional class weighting."""
    weight = None
    if train_labels is not None and weight_strategy != "none":
        if weight_strategy == "inverse":
            weight = compute_class_weight_inverse(train_labels)
        elif weight_strategy == "effective":
            weight = compute_class_weight_effective(train_labels)
    return MaskedCELoss(weight=weight)
