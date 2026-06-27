"""Region attribution and counterfactual analysis.

Two faithful grounding methods (from phase_5_new.md §3-B):

1. Region attribution (exact, mandatory):
   attr[r,d] = exp(region_logit[r,d]) / Σ_r' exp(region_logit[r',d])
   This IS the gradient of LSE — not an approximation.

2. Region counterfactual (optional, cheap, faithful):
   Ablate region r → recompute LSE → does the call flip?
   If yes → region r is "hinge" for disease d.
"""

from __future__ import annotations

import torch

from phase_3.constants import NUM_DISEASES, NUM_REGIONS


def region_attribution(
    region_logits: torch.Tensor,    # [29, 14]
    region_mask: torch.Tensor,      # [29] bool
) -> torch.Tensor:
    """Compute exact region attribution via softmax of region logits.

    Returns
    -------
    attr : [29, 14] — attribution weight per (region, disease).
        attr[r,d] = contribution of region r to the image-level decision for disease d.
        Sums to 1 over valid regions for each disease.
        Invalid regions get attr = 0.
    """
    masked_logits = region_logits.clone()
    masked_logits[~region_mask] = float("-inf")

    # Softmax over regions (dim=0)
    attr = torch.softmax(masked_logits, dim=0)
    attr = attr.nan_to_num(0.0)

    return attr  # [29, 14]


def region_counterfactual(
    region_logits: torch.Tensor,    # [29, 14]
    region_mask: torch.Tensor,      # [29] bool
    threshold: float = 0.0,         # logit threshold for "present"
) -> torch.Tensor:
    """Find hinge regions via leave-one-out counterfactual.

    For each valid region r: ablate it, recompute LSE → if the call flips
    (present → absent), region r is a "hinge" for that disease.

    Returns
    -------
    is_hinge : [29, 14] bool
    """
    R, D = region_logits.shape
    is_hinge = torch.zeros(R, D, dtype=torch.bool)

    # Full image logit (LSE)
    masked = region_logits.clone()
    masked[~region_mask] = float("-inf")
    full_lse = torch.logsumexp(masked, dim=0)  # [14]

    full_present = full_lse > threshold  # [14]

    for r in range(R):
        if not region_mask[r]:
            continue

        # Ablate region r
        ablated_mask = region_mask.clone()
        ablated_mask[r] = False

        if not ablated_mask.any():
            # Only region → definitely hinge
            is_hinge[r] = full_present
            continue

        ablated_logits = region_logits.clone()
        ablated_logits[~ablated_mask] = float("-inf")
        ablated_lse = torch.logsumexp(ablated_logits, dim=0)  # [14]

        ablated_present = ablated_lse > threshold

        # Hinge: present with r, absent without r
        is_hinge[r] = full_present & ~ablated_present

    return is_hinge
