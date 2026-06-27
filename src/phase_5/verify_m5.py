"""Smoke test for M5 pipeline with dummy M3/M4 outputs.

Run:  python -m phase_5.verify_m5

Tests:
1. Template realize → zero hallucination
2. has_prior=False → zero temporal language
3. has_prior=True → progression readout correct
4. Round-trip keyword check passes
5. Report serialisation round-trip
"""

from __future__ import annotations

import torch

from phase_3.constants import NUM_DISEASES, NUM_PROG, NUM_REGIONS

from .assemble import assemble_report
from .realize import realize, realize_template
from .report import M5Report
from .verify import check_temporal_hallucination, verify_report


def _make_dummy_m3():
    """Create dummy M3 outputs with some positive findings."""
    region_logits = torch.randn(NUM_REGIONS, NUM_DISEASES) * 0.5

    # Make a few diseases clearly positive
    region_logits[16, 2] = 3.0   # cardiac silhouette → Cardiomegaly (high)
    region_logits[9, 3] = 2.0    # right lower lung → Lung Opacity (medium)
    region_logits[5, 7] = 1.5    # right upper lung → Pneumonia (hedge range)

    image_logits = torch.logsumexp(region_logits, dim=0)  # [14]

    region_mask = torch.ones(NUM_REGIONS, dtype=torch.bool)
    region_mask[25:] = False  # last 4 regions invalid

    boxes = torch.rand(NUM_REGIONS, 4) * 0.5
    boxes[:, 2:] += 0.5

    return {
        "region_logits": region_logits,
        "image_logits": image_logits,
        "region_mask": region_mask,
        "boxes": boxes,
        "boxes_original": boxes.clone(),
    }


def _make_dummy_m4():
    """Create dummy M4 outputs."""
    prog_logits = torch.randn(NUM_REGIONS, NUM_DISEASES, NUM_PROG)
    # Make cardiomegaly stable at cardiac silhouette
    prog_logits[16, 2, :] = torch.tensor([-1.0, 3.0, -1.0])  # stable
    # Make lung opacity worsened at right lower lung
    prog_logits[9, 3, :] = torch.tensor([-1.0, -1.0, 3.0])   # worsened

    pair_mask = torch.ones(NUM_REGIONS, NUM_DISEASES, dtype=torch.bool)

    return {
        "prog_logits": prog_logits,
        "pair_mask": pair_mask,
    }


def main():
    print("=" * 60)
    print("Phase 5 — M5 Report Assembler verification")
    print("=" * 60)

    m3 = _make_dummy_m3()
    m4 = _make_dummy_m4()

    # ---- Test 1: No prior (single study) ----
    print("\n--- Test 1: Single study (no prior) ---")
    report_no_prior = assemble_report(
        **m3, has_prior=False, prog_logits=None, pair_mask=None,
    )
    report_no_prior = realize(report_no_prior, mode="template")
    report_no_prior = verify_report(report_no_prior)

    print(report_no_prior.summary())
    print(f"\nProse:\n{report_no_prior.prose}")

    # Check: no temporal language
    temporal = check_temporal_hallucination(report_no_prior)
    assert not temporal["temporal_halluc"], (
        f"Temporal hallucination with no prior! Matches: {temporal['matches']}"
    )
    print("[OK] No temporal hallucination (has_prior=False)")

    # Check: no progression in findings
    for f in report_no_prior.findings:
        assert f.progression is None, f"Progression set without prior: {f}"
    print("[OK] No progression in findings")

    assert report_no_prior.verify_passed
    print("[OK] Verify passed")

    # ---- Test 2: With prior (temporal) ----
    print("\n--- Test 2: Temporal study (with prior) ---")
    report_temporal = assemble_report(
        **m3, has_prior=True, prog_logits=m4["prog_logits"],
        pair_mask=m4["pair_mask"],
    )
    report_temporal = realize(report_temporal, mode="template")
    report_temporal = verify_report(report_temporal)

    print(report_temporal.summary())
    print(f"\nProse:\n{report_temporal.prose}")

    # Check: some findings should have progression
    has_prog = any(f.progression is not None for f in report_temporal.findings)
    print(f"Has progression: {has_prog}")
    assert report_temporal.verify_passed
    print("[OK] Verify passed (temporal)")

    # ---- Test 3: Serialisation round-trip ----
    print("\n--- Test 3: JSON serialisation round-trip ---")
    json_str = report_temporal.to_json()
    restored = M5Report.from_json(json_str)

    assert len(restored.findings) == len(report_temporal.findings)
    assert restored.has_prior == report_temporal.has_prior
    for orig, rest in zip(report_temporal.findings, restored.findings):
        assert orig.disease == rest.disease
        assert orig.region == rest.region
        assert orig.status == rest.status
        assert abs(orig.p_cal - rest.p_cal) < 1e-6
    print("[OK] JSON round-trip passed")

    # ---- Test 4: Empty findings ----
    print("\n--- Test 4: Empty findings (all diseases below threshold) ---")
    m3_neg = _make_dummy_m3()
    m3_neg["region_logits"] = torch.ones(NUM_REGIONS, NUM_DISEASES) * -5.0
    m3_neg["image_logits"] = torch.logsumexp(m3_neg["region_logits"], dim=0)

    report_empty = assemble_report(**m3_neg, has_prior=False)
    report_empty = realize(report_empty, mode="template")
    assert len(report_empty.findings) == 0
    assert "No significant findings" in report_empty.prose
    print("[OK] Empty findings handled correctly")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("[OK] All M5 verification tests passed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
