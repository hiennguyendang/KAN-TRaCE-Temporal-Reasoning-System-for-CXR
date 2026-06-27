"""Assemble structured findings from M3/M4 outputs.

Implements Tầng A→D of phase_5_new.md:
- Tầng A: Structured findings core (calibrate → choose findings)
- Tầng B: Grounding "ở đâu" (region attribution)
- Tầng C: Calibration + abstention (assert/hedge/omit bands)
- Tầng D: Temporal guard (hard gate on has_prior)
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from .attribution import region_attribution, region_counterfactual
from .constants import (
    CRITICAL_DISEASES,
    DEFAULT_THRESHOLDS,
    LABEL_NAMES,
    NUM_DISEASES,
    NUM_PROG,
    NUM_REGIONS,
    PROG_CLASSES,
    REGION_NAMES,
)
from .report import Finding, M5Report

logger = logging.getLogger(__name__)


def assemble_report(
    region_logits: torch.Tensor,      # [29, 14]
    image_logits: torch.Tensor,       # [14]
    region_mask: torch.Tensor,        # [29] bool
    boxes: torch.Tensor,              # [29, 4]  resized space
    boxes_original: torch.Tensor,     # [29, 4]  original pixel space
    has_prior: bool = False,
    prog_logits: torch.Tensor | None = None,  # [29, 14, 3]
    pair_mask: torch.Tensor | None = None,    # [29, 14] bool
    temp_disease: float = 1.0,        # temperature for M3
    temp_prog: float = 1.0,           # temperature for M4
    thresholds: dict[str, float] | None = None,
) -> M5Report:
    """Assemble an M5Report from M3/M4 raw outputs.

    This is the core pipeline: Tầng A → B → C → D.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    tau_assert = thresholds.get("tau_assert", 0.7)
    tau_uncertain = thresholds.get("tau_uncertain", 0.3)
    tau_prog = thresholds.get("tau_prog", 0.6)

    # ---- Tầng A: Calibrated probabilities ----
    p_image = torch.sigmoid(image_logits / temp_disease)        # [14]
    p_region = torch.sigmoid(region_logits / temp_disease)      # [29, 14]

    # ---- Tầng B: Region attribution ----
    attr = region_attribution(region_logits, region_mask)        # [29, 14]
    hinge = region_counterfactual(region_logits, region_mask)    # [29, 14] bool

    findings: list[Finding] = []
    review_regions: set[str] = set()

    for d in range(NUM_DISEASES):
        p_cal = p_image[d].item()
        disease_name = LABEL_NAMES[d]

        # ---- Tầng C: Assert / hedge / omit ----
        if p_cal < tau_uncertain:
            continue  # omit

        status = "assert" if p_cal >= tau_assert else "hedge"

        # Representative region = argmax region_logits for this disease
        # (only among valid regions)
        valid_logits = region_logits[:, d].clone()
        valid_logits[~region_mask] = float("-inf")
        rep_r = int(valid_logits.argmax().item())

        if not region_mask[rep_r]:
            continue  # no valid region at all

        region_name = REGION_NAMES[rep_r]
        bbox = boxes[rep_r].tolist()
        bbox_orig = boxes_original[rep_r].tolist()
        r_attr = attr[rep_r, d].item()
        is_hinge = hinge[rep_r, d].item()

        # Flag for review: hedge + critical disease
        flag = (status == "hedge" and disease_name in CRITICAL_DISEASES)

        # ---- Tầng D: Temporal guard ----
        progression = None
        prog_conf = None

        if has_prior and prog_logits is not None and pair_mask is not None:
            if pair_mask[rep_r, d]:
                # Calibrated softmax
                p3 = torch.softmax(
                    prog_logits[rep_r, d, :] / temp_prog, dim=0
                )
                cls = int(p3.argmax().item())
                conf = p3.max().item()

                if conf >= tau_prog:
                    progression = PROG_CLASSES[cls]
                    prog_conf = conf
                # else: not confident enough → don't mention progression
        # If has_prior is False → progression stays None (no temporal language)

        findings.append(Finding(
            disease=disease_name,
            region=region_name,
            bbox=bbox,
            bbox_original=bbox_orig,
            p_cal=p_cal,
            status=status,
            region_attr=r_attr,
            is_hinge=bool(is_hinge),
            progression=progression,
            prog_conf=prog_conf,
            flag_review=flag,
        ))

        if flag:
            review_regions.add(region_name)

    # Sort findings by confidence (highest first)
    findings.sort(key=lambda f: f.p_cal, reverse=True)

    return M5Report(
        findings=findings,
        review_regions=sorted(review_regions),
        has_prior=has_prior,
        prose=None,        # set by realize.py
        verify_passed=True,  # set by verify.py
    )
