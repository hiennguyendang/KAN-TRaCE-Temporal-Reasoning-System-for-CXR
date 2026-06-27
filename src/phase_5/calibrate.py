"""Temperature scaling calibration for M3 and M4 logits.

Fits a single temperature T (or per-disease T[14]) on the validation set
to minimise NLL of σ(logit/T).  This is a lightweight post-hoc step —
does NOT retrain the model.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from phase_3.constants import MASK_VALUE, NUM_DISEASES, NUM_PROG

logger = logging.getLogger(__name__)


class TemperatureScaler(nn.Module):
    """Learnable temperature for Platt scaling.

    For M3 (disease classification): σ(logit / T) → calibrated probability.
    For M4 (progression): softmax(logit / T) → calibrated 3-class prob.
    """

    def __init__(self, per_class: bool = False, num_classes: int = 1):
        super().__init__()
        n = num_classes if per_class else 1
        # Init T=1.0 (identity)
        self.temperature = nn.Parameter(torch.ones(n))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by learned temperature."""
        T = self.temperature.clamp(min=0.01)
        return logits / T


def fit_temperature_disease(
    logits: torch.Tensor,    # [N, 14]  — val set M3 logits
    targets: torch.Tensor,   # [N, 14]  — val set labels
    per_disease: bool = False,
    mask_value: int = MASK_VALUE,
    max_iter: int = 200,
    lr: float = 0.01,
) -> TemperatureScaler:
    """Fit temperature scaling for M3 disease classification.

    Minimises NLL: -Σ [y·log σ(l/T) + (1-y)·log(1 - σ(l/T))] on valid entries.
    """
    n_cls = NUM_DISEASES if per_disease else 1
    scaler = TemperatureScaler(per_class=per_disease, num_classes=n_cls)
    optimizer = torch.optim.LBFGS(scaler.parameters(), lr=lr, max_iter=max_iter)

    mask = targets != mask_value

    def closure():
        optimizer.zero_grad()
        scaled = scaler(logits)
        # Masked BCE
        loss_full = F.binary_cross_entropy_with_logits(
            scaled, targets.float().clamp(0, 1), reduction="none",
        )
        loss = (loss_full * mask.float()).sum() / mask.float().sum().clamp(min=1)
        loss.backward()
        return loss

    optimizer.step(closure)

    T = scaler.temperature.detach()
    logger.info("Fitted M3 temperature: %s", T.tolist())
    return scaler


def fit_temperature_progression(
    logits: torch.Tensor,    # [N, 29, 14, 3]  — val set M4 logits
    targets: torch.Tensor,   # [N, 29, 14]  — val set progression labels
    pair_mask: torch.Tensor | None = None,
    mask_value: int = MASK_VALUE,
    max_iter: int = 200,
    lr: float = 0.01,
) -> TemperatureScaler:
    """Fit temperature for M4 progression (3-class softmax)."""
    scaler = TemperatureScaler(per_class=False, num_classes=1)
    optimizer = torch.optim.LBFGS(scaler.parameters(), lr=lr, max_iter=max_iter)

    # Flatten
    B, R, D, C = logits.shape
    logits_flat = logits.reshape(-1, C)
    targets_flat = targets.reshape(-1)

    valid = targets_flat != mask_value
    if pair_mask is not None:
        valid = valid & pair_mask.reshape(-1)

    def closure():
        optimizer.zero_grad()
        scaled = scaler(logits_flat)
        loss = F.cross_entropy(scaled[valid], targets_flat[valid])
        loss.backward()
        return loss

    optimizer.step(closure)

    T = scaler.temperature.detach()
    logger.info("Fitted M4 temperature: %s", T.tolist())
    return scaler


def expected_calibration_error(
    probs: torch.Tensor,   # [N] — calibrated probabilities
    targets: torch.Tensor, # [N] — binary labels
    n_bins: int = 15,
) -> float:
    """Compute Expected Calibration Error (ECE)."""
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = probs.numel()

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (probs >= lo) & (probs < hi)
        if not in_bin.any():
            continue
        avg_conf = probs[in_bin].mean().item()
        avg_acc = targets[in_bin].float().mean().item()
        ece += abs(avg_conf - avg_acc) * in_bin.sum().item() / total

    return ece
