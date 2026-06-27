"""Evaluation metrics for Phase 3 (C-MLP).

Design decision (phase_3_4_mlp_update.md §7.4):
- Primary: macro-F1 + per-class F1 (NOT accuracy — it lies for imbalanced data).
- Secondary: AUROC (multilabel).
- Checkpoint selection: macro-F1.
"""

from __future__ import annotations

from typing import Any

import torch

from .constants import LABEL_NAMES, MASK_VALUE, NUM_DISEASES


def _apply_mask(
    preds: torch.Tensor,
    targets: torch.Tensor,
    mask_value: int = MASK_VALUE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return masked preds, targets, and the mask itself (per-element)."""
    mask = targets != mask_value
    return preds, targets, mask


def per_class_f1(
    logits: torch.Tensor,     # [N, 14]
    targets: torch.Tensor,    # [N, 14]  values in {0, 1, mask_value}
    thresholds: torch.Tensor | None = None,  # [14] per-disease threshold
    mask_value: int = MASK_VALUE,
) -> dict[str, float]:
    """Compute per-class F1 and macro-F1.

    Returns dict: {'macro_f1': float, '<disease_name>_f1': float, ...}
    """
    if thresholds is None:
        thresholds = torch.full((NUM_DISEASES,), 0.5)

    probs = torch.sigmoid(logits)  # [N, 14]
    results: dict[str, float] = {}
    f1_scores = []

    for d in range(NUM_DISEASES):
        mask = targets[:, d] != mask_value
        if not mask.any():
            results[f"{LABEL_NAMES[d]}_f1"] = float("nan")
            continue

        pred_d = (probs[mask, d] >= thresholds[d]).long()
        true_d = targets[mask, d].long()

        tp = ((pred_d == 1) & (true_d == 1)).sum().float()
        fp = ((pred_d == 1) & (true_d == 0)).sum().float()
        fn = ((pred_d == 0) & (true_d == 1)).sum().float()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        results[f"{LABEL_NAMES[d]}_f1"] = f1.item()
        f1_scores.append(f1.item())

    results["macro_f1"] = (
        sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    )
    return results


def multilabel_auroc(
    logits: torch.Tensor,     # [N, 14]
    targets: torch.Tensor,    # [N, 14]
    mask_value: int = MASK_VALUE,
) -> dict[str, float]:
    """Compute per-class AUROC (using manual trapezoid if torchmetrics unavailable).

    Returns dict: {'macro_auroc': float, '<disease_name>_auroc': float, ...}
    """
    probs = torch.sigmoid(logits)
    results: dict[str, float] = {}
    aurocs = []

    for d in range(NUM_DISEASES):
        mask = targets[:, d] != mask_value
        if not mask.any():
            results[f"{LABEL_NAMES[d]}_auroc"] = float("nan")
            continue

        p = probs[mask, d]
        t = targets[mask, d].long()

        # Need both classes present
        if t.sum() == 0 or t.sum() == t.numel():
            results[f"{LABEL_NAMES[d]}_auroc"] = float("nan")
            continue

        # Sort by descending predicted probability
        sorted_indices = torch.argsort(p, descending=True)
        t_sorted = t[sorted_indices]

        # TPR and FPR at each threshold
        n_pos = t.sum().float()
        n_neg = (t.numel() - t.sum()).float()

        tp_cumsum = t_sorted.cumsum(0).float()
        fp_cumsum = (1 - t_sorted).cumsum(0).float()

        tpr = tp_cumsum / n_pos
        fpr = fp_cumsum / n_neg

        # Prepend (0,0)
        tpr = torch.cat([torch.zeros(1, device=tpr.device), tpr])
        fpr = torch.cat([torch.zeros(1, device=fpr.device), fpr])

        # Trapezoid rule
        auc = torch.trapezoid(tpr, fpr).item()
        results[f"{LABEL_NAMES[d]}_auroc"] = auc
        aurocs.append(auc)

    results["macro_auroc"] = sum(aurocs) / len(aurocs) if aurocs else 0.0
    return results


def compute_all_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    thresholds: torch.Tensor | None = None,
) -> dict[str, float]:
    """Compute all Phase 3 metrics: F1 + AUROC."""
    m = {}
    m.update(per_class_f1(logits, targets, thresholds))
    m.update(multilabel_auroc(logits, targets))
    return m
