"""End-to-end M5 pipeline: load M3/M4 outputs → assemble → realize → verify.

CLI entry point for running M5 on saved model outputs.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from .assemble import assemble_report
from .realize import realize
from .verify import verify_report

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="M5 Report Assembler")

    p.add_argument("--m3-output", type=Path, required=True,
                   help="Path to M3 output .pt file (dict with region_logits, etc.)")
    p.add_argument("--m4-output", type=Path, default=None,
                   help="Path to M4 output .pt file (optional, for temporal)")
    p.add_argument("--output-dir", type=Path, default=Path("./m5_reports"))
    p.add_argument("--temp-disease", type=float, default=1.0,
                   help="Temperature for M3 calibration")
    p.add_argument("--temp-prog", type=float, default=1.0,
                   help="Temperature for M4 calibration")
    p.add_argument("--tau-assert", type=float, default=0.7)
    p.add_argument("--tau-uncertain", type=float, default=0.3)
    p.add_argument("--tau-prog", type=float, default=0.6)
    p.add_argument("--realize-mode", choices=["template", "llm"], default="template")

    return p


def run_single(
    m3_data: dict[str, torch.Tensor],
    m4_data: dict[str, torch.Tensor] | None = None,
    temp_disease: float = 1.0,
    temp_prog: float = 1.0,
    thresholds: dict[str, float] | None = None,
    realize_mode: str = "template",
    chexbert_fn=None,
) -> dict:
    """Run M5 on a single study.

    Parameters
    ----------
    m3_data : dict with keys:
        'region_logits' [29,14], 'image_logits' [14], 'region_mask' [29],
        'boxes' [29,4], 'boxes_original' [29,4]
    m4_data : optional dict with keys:
        'prog_logits' [29,14,3], 'pair_mask' [29,14]

    Returns
    -------
    dict with 'report' (M5Report), 'report_json' (str), 'prose' (str).
    """
    has_prior = m4_data is not None

    report = assemble_report(
        region_logits=m3_data["region_logits"],
        image_logits=m3_data["image_logits"],
        region_mask=m3_data["region_mask"],
        boxes=m3_data["boxes"],
        boxes_original=m3_data.get("boxes_original", m3_data["boxes"]),
        has_prior=has_prior,
        prog_logits=m4_data["prog_logits"] if m4_data else None,
        pair_mask=m4_data["pair_mask"] if m4_data else None,
        temp_disease=temp_disease,
        temp_prog=temp_prog,
        thresholds=thresholds,
    )

    # Realize
    report = realize(report, mode=realize_mode)

    # Verify
    report = verify_report(report, chexbert_fn=chexbert_fn)

    # If verify failed and using LLM, fall back to template
    if not report.verify_passed and realize_mode == "llm":
        logger.warning("LLM output failed verification — falling back to template")
        report = realize(report, mode="template")
        report = verify_report(report, chexbert_fn=chexbert_fn)

    return {
        "report": report,
        "report_json": report.to_json(),
        "prose": report.prose,
        "verify_passed": report.verify_passed,
    }


def main():
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load M3 output
    m3_data = torch.load(args.m3_output, map_location="cpu", weights_only=True)

    # Load M4 output (optional)
    m4_data = None
    if args.m4_output and args.m4_output.exists():
        m4_data = torch.load(args.m4_output, map_location="cpu", weights_only=True)

    thresholds = {
        "tau_assert": args.tau_assert,
        "tau_uncertain": args.tau_uncertain,
        "tau_prog": args.tau_prog,
    }

    result = run_single(
        m3_data=m3_data,
        m4_data=m4_data,
        temp_disease=args.temp_disease,
        temp_prog=args.temp_prog,
        thresholds=thresholds,
        realize_mode=args.realize_mode,
    )

    # Save
    report_path = output_dir / "report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(result["report_json"])

    prose_path = output_dir / "report.txt"
    with open(prose_path, "w", encoding="utf-8") as f:
        f.write(result["prose"] or "")

    print(f"\n{'='*60}")
    print("M5 Report Summary")
    print(f"{'='*60}")
    print(result["report"].summary())
    print(f"\nVerify passed: {result['verify_passed']}")
    print(f"Saved to: {report_path}")

    if result["prose"]:
        print(f"\n--- Prose ---\n{result['prose']}")


if __name__ == "__main__":
    main()
