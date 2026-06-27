"""Configuration for Phase 4 T-MLP training."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

ON_KAGGLE = Path("/kaggle/input").exists()
_REPO = Path(__file__).resolve().parents[2]
WORK_ROOT = Path("/kaggle/working") if ON_KAGGLE else (_REPO / "runs" / "phase_4")


def env_path(name: str, default: Path) -> Path:
    val = os.environ.get(name)
    return Path(val) if val else default


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="T-MLP Phase 4 training")

    # --- Data paths ---
    p.add_argument("--feature-cache", type=Path,
                   default=env_path("FEATURE_CACHE",
                                    Path("/kaggle/input/biovilt-features")
                                    if ON_KAGGLE else _REPO / "data" / "feature_cache"))
    p.add_argument("--metadata", type=Path,
                   default=env_path("METADATA",
                                    Path("/kaggle/input/mimic-metadata/mimic_metadata_final.jsonl")
                                    if ON_KAGGLE else _REPO / "metadata" / "mimic_metadata_final.jsonl"))
    p.add_argument("--bbox-dir", type=Path,
                   default=env_path("BBOX_DIR",
                                    Path("/kaggle/input/yolo-bbox-labels")
                                    if ON_KAGGLE else _REPO / "data" / "yolo_labels"))
    p.add_argument("--splits", type=Path,
                   default=env_path("SPLITS_JSON",
                                    Path("/kaggle/input/mimic-metadata/selected_patient_splits.json")
                                    if ON_KAGGLE else _REPO / "selected_patient_splits.json"))
    p.add_argument("--cmlp-checkpoint", type=Path,
                   default=env_path("CMLP_CKPT",
                                    Path("/kaggle/working/cmlp_best.pt")
                                    if ON_KAGGLE else _REPO / "runs" / "phase_3" / "cmlp_best.pt"),
                   help="C-MLP checkpoint for extracting region_feat")
    p.add_argument("--output-dir", type=Path,
                   default=env_path("TMLP_OUTPUT", WORK_ROOT))

    # --- Architecture ---
    p.add_argument("--head", choices=["mlp", "kan"], default="mlp")
    p.add_argument("--use-hadamard", action="store_true",
                   help="Add element-wise product to concat (adds +128 dim)")
    p.add_argument("--no-labels", action="store_true",
                   help="Omit label input from concat (removes -28 dim)")
    p.add_argument("--label-input", choices=["gt", "pred"], default="gt",
                   help="Use GT labels (teacher forcing) or C-MLP predictions")
    p.add_argument("--hidden-dim", type=int, default=128)

    # --- Loss ---
    p.add_argument("--class-weight", choices=["none", "inverse", "effective"],
                   default="inverse", help="Class weight strategy for CE 3-class")

    # --- Training ---
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")

    p.add_argument("--resume", type=Path, default=None)

    return p


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()
