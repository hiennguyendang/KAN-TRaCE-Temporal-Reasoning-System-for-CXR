"""Smoke test: verify tensor shapes through the C-MLP pipeline.

Run:  python -m phase_3.verify_shapes
"""

from __future__ import annotations

import torch

from .constants import ENCODER_DIM, NUM_DISEASES, NUM_PATCHES, NUM_REGIONS, PROJ_DIM
from .model import CMLP


def main():
    print("=" * 60)
    print("Phase 3 — C-MLP shape verification")
    print("=" * 60)

    device = torch.device("cpu")
    B = 4

    # Dummy inputs
    feats = torch.randn(B, NUM_PATCHES, ENCODER_DIM, device=device)
    # Random normalised boxes (x1,y1,x2,y2 in [0,1])
    boxes_raw = torch.rand(B, NUM_REGIONS, 4, device=device)
    # Ensure x1 < x2, y1 < y2
    boxes = torch.zeros_like(boxes_raw)
    boxes[:, :, 0] = boxes_raw[:, :, 0].clamp(0, 0.4)
    boxes[:, :, 1] = boxes_raw[:, :, 1].clamp(0, 0.4)
    boxes[:, :, 2] = boxes_raw[:, :, 2].clamp(0.5, 1.0)
    boxes[:, :, 3] = boxes_raw[:, :, 3].clamp(0.5, 1.0)
    # Make a few sentinel boxes
    boxes[0, 25:, :] = 0.0
    boxes[1, 20:, :] = 0.0

    print(f"\nInput feats:  {feats.shape}")
    print(f"Input boxes:  {boxes.shape}")

    # --- Test attention-pool model ---
    model_attn = CMLP(pool_type="attn", head_type="mlp", use_global=True)
    model_attn.eval()
    with torch.no_grad():
        out = model_attn(feats, boxes)

    print(f"\n--- C-MLP (attn pool, global ON) ---")
    print(f"image_logits:         {out.image_logits.shape}")
    print(f"region_logits:        {out.region_logits.shape}")
    print(f"region_feat:          {out.region_feat.shape}")
    print(f"region_mask:          {out.region_mask.shape}  sum={out.region_mask.sum(dim=1).tolist()}")
    print(f"alpha:                {out.alpha.shape}")
    print(f"global_logits:        {out.global_logits.shape if out.global_logits is not None else 'None'}")
    print(f"region_image_logits:  {out.region_image_logits.shape}")

    assert out.image_logits.shape == (B, NUM_DISEASES)
    assert out.region_logits.shape == (B, NUM_REGIONS, NUM_DISEASES)
    assert out.region_feat.shape == (B, NUM_REGIONS, PROJ_DIM)
    assert out.region_mask.shape == (B, NUM_REGIONS)
    assert out.alpha.shape == (B, NUM_REGIONS, NUM_PATCHES)
    assert out.global_logits is not None
    assert out.global_logits.shape == (B, NUM_DISEASES)

    # Check sentinel regions are zeroed
    for b in range(B):
        for r in range(NUM_REGIONS):
            if not out.region_mask[b, r]:
                assert out.region_logits[b, r].abs().sum() == 0, \
                    f"region_logits not zeroed for sentinel (b={b}, r={r})"

    # --- Test mean-pool model (no global) ---
    model_mean = CMLP(pool_type="mean", head_type="mlp", use_global=False)
    model_mean.eval()
    with torch.no_grad():
        out2 = model_mean(feats, boxes)

    print(f"\n--- C-MLP (mean pool, global OFF) ---")
    print(f"image_logits:         {out2.image_logits.shape}")
    print(f"global_logits:        {out2.global_logits}")

    assert out2.global_logits is None
    assert out2.image_logits.shape == (B, NUM_DISEASES)

    # --- Parameter count ---
    n_attn = sum(p.numel() for p in model_attn.parameters())
    n_mean = sum(p.numel() for p in model_mean.parameters())
    print(f"\nParam count (attn+global): {n_attn:,}")
    print(f"Param count (mean, no global): {n_mean:,}")

    print("\n[OK] All shape checks passed!")


if __name__ == "__main__":
    main()
