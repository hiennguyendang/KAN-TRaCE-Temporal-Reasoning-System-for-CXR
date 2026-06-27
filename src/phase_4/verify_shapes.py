"""Smoke test: verify T-MLP shapes with dummy data.

Run:  python -m phase_4.verify_shapes
"""

from __future__ import annotations

import torch

from phase_3.constants import NUM_DISEASES, NUM_PROG, NUM_REGIONS, PROJ_DIM
from .model import TMLP


def main():
    print("=" * 60)
    print("Phase 4 — T-MLP shape verification")
    print("=" * 60)

    B = 4
    device = torch.device("cpu")

    # Dummy region_feat from C-MLP
    prior_feat = torch.randn(B, NUM_REGIONS, PROJ_DIM, device=device)
    curr_feat = torch.randn(B, NUM_REGIONS, PROJ_DIM, device=device)

    # Dummy labels [B, 14]
    prior_labels = torch.randint(0, 2, (B, NUM_DISEASES), device=device)
    curr_labels = torch.randint(0, 2, (B, NUM_DISEASES), device=device)

    # Region masks
    prior_mask = torch.ones(B, NUM_REGIONS, dtype=torch.bool, device=device)
    curr_mask = torch.ones(B, NUM_REGIONS, dtype=torch.bool, device=device)
    # Invalidate some regions
    prior_mask[0, 25:] = False
    curr_mask[1, 20:] = False

    print(f"\nprior_feat:   {prior_feat.shape}")
    print(f"curr_feat:    {curr_feat.shape}")
    print(f"prior_labels: {prior_labels.shape}")
    print(f"curr_labels:  {curr_labels.shape}")

    # --- Default T-MLP (labels ON, hadamard OFF) ---
    model = TMLP(use_hadamard=False, use_labels=True)
    model.eval()
    print(f"\nT-MLP in_dim = {model.in_dim} (expected 412 = 128*3 + 14*2)")

    with torch.no_grad():
        out = model(prior_feat, curr_feat, prior_labels, curr_labels,
                     prior_mask, curr_mask)

    print(f"prog_logits:  {out.prog_logits.shape}")
    print(f"pair_mask:    {out.pair_mask.shape}")

    assert out.prog_logits.shape == (B, NUM_REGIONS, NUM_DISEASES, NUM_PROG)
    assert out.pair_mask.shape == (B, NUM_REGIONS, NUM_DISEASES)
    assert model.in_dim == PROJ_DIM * 3 + NUM_DISEASES * 2  # 412

    # --- T-MLP with hadamard ---
    model_h = TMLP(use_hadamard=True, use_labels=True)
    print(f"\nT-MLP (hadamard) in_dim = {model_h.in_dim} (expected {412 + PROJ_DIM})")
    assert model_h.in_dim == PROJ_DIM * 4 + NUM_DISEASES * 2  # 540

    # --- T-MLP no labels ---
    model_nl = TMLP(use_hadamard=False, use_labels=False)
    print(f"T-MLP (no labels) in_dim = {model_nl.in_dim} (expected {PROJ_DIM * 3})")
    assert model_nl.in_dim == PROJ_DIM * 3  # 384

    # --- Param count ---
    n = sum(p.numel() for p in model.parameters())
    print(f"\nT-MLP params (default): {n:,}")

    print("\n[OK] All T-MLP shape checks passed!")


if __name__ == "__main__":
    main()
