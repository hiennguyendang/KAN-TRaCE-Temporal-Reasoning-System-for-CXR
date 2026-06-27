"""Metrics for Phase 4 T-MLP (3-class progression).

Primary: macro-F1 across 3 classes + per-class F1.
Warning flag: accuracy ≈ 0.51 ≈ always-predict-stable → useless.
"""

from __future__ import annotations

import torch

from phase_3.constants import MASK_VALUE, NUM_DISEASES, PROG_CLASSES, NUM_PROG


def progression_metrics(
    logits: torch.Tensor,     # [N, 29, 14, 3] or flattened
    targets: torch.Tensor,    # [N, 29, 14]
    pair_mask: torch.Tensor | None = None,  # [N, 29, 14] bool
    mask_value: int = MASK_VALUE,
) -> dict[str, float]:
    """Compute macro-F1 and per-class F1 for 3-class progression.

    Returns dict with: macro_f1, improved_f1, stable_f1, worsened_f1,
    accuracy, support_improved, support_stable, support_worsened.
    """
    if logits.dim() == 4:
        preds = logits.argmax(dim=-1)  # [N, 29, 14]
    else:
        preds = logits

    # Flatten
    preds_flat = preds.reshape(-1)
    targets_flat = targets.reshape(-1)

    # Build valid mask
    valid = targets_flat != mask_value
    if pair_mask is not None:
        valid = valid & pair_mask.reshape(-1)

    if not valid.any():
        return {"macro_f1": 0.0, "accuracy": 0.0}

    p = preds_flat[valid]
    t = targets_flat[valid]

    results: dict[str, float] = {}
    f1_scores = []

    for c in range(NUM_PROG):
        tp = ((p == c) & (t == c)).sum().float()
        fp = ((p == c) & (t != c)).sum().float()
        fn = ((p != c) & (t == c)).sum().float()
        support = (t == c).sum().item()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        cls_name = PROG_CLASSES[c]
        results[f"{cls_name}_f1"] = f1.item()
        results[f"support_{cls_name}"] = support
        f1_scores.append(f1.item())

    results["macro_f1"] = sum(f1_scores) / len(f1_scores)
    results["accuracy"] = (p == t).float().mean().item()

    # Warning: if accuracy ≈ stable-class proportion → model is trivial
    stable_prop = (t == 1).float().mean().item()
    results["stable_proportion"] = stable_prop
    results["is_trivial"] = abs(results["accuracy"] - stable_prop) < 0.02

    return results


def confusion_matrix_3class(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pair_mask: torch.Tensor | None = None,
    mask_value: int = MASK_VALUE,
) -> torch.Tensor:
    """3×3 confusion matrix: rows = true, cols = predicted."""
    preds = logits.argmax(dim=-1) if logits.dim() == 4 else logits
    p = preds.reshape(-1)
    t = targets.reshape(-1)

    valid = t != mask_value
    if pair_mask is not None:
        valid = valid & pair_mask.reshape(-1)

    p = p[valid]
    t = t[valid]

    cm = torch.zeros(NUM_PROG, NUM_PROG, dtype=torch.long)
    for true_c in range(NUM_PROG):
        for pred_c in range(NUM_PROG):
            cm[true_c, pred_c] = ((t == true_c) & (p == pred_c)).sum()

    return cm
