"""Configuration for Phase 3 C-MLP training.

Kaggle-friendly defaults: reads from /kaggle/input, writes to /kaggle/working.
All paths overridable via CLI or environment variables.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .constants import ENCODER_DIM, NUM_DISEASES, NUM_PATCHES, NUM_REGIONS, PROJ_DIM

# ---------------------------------------------------------------------------
# Path defaults — Kaggle vs local.
# ---------------------------------------------------------------------------
ON_KAGGLE = Path("/kaggle/input").exists()

_REPO = Path(__file__).resolve().parents[2]  # project root
WORK_ROOT = Path("/kaggle/working") if ON_KAGGLE else (_REPO / "runs" / "phase_3")

# Feature cache: directory of per-study .pt files  (each [196, 512] float16)
DEFAULT_FEATURE_CACHE = (
    Path("/kaggle/input/biovilt-features")
    if ON_KAGGLE
    else (_REPO / "data" / "feature_cache")
)

# Metadata JSONL (has labels, split, scene_path, dicom_id, …)
DEFAULT_METADATA = (
    Path("/kaggle/input/mimic-metadata/mimic_metadata_final.jsonl")
    if ON_KAGGLE
    else (_REPO / "metadata" / "mimic_metadata_final.jsonl")
)

# Bbox source: directory of YOLO .txt label files (29-class, normalised xywh)
DEFAULT_BBOX_DIR = (
    Path("/kaggle/input/yolo-bbox-labels")
    if ON_KAGGLE
    else (_REPO / "data" / "yolo_labels")
)

# Patient splits JSON (already exists in repo root)
DEFAULT_SPLITS = (
    Path("/kaggle/input/mimic-metadata/selected_patient_splits.json")
    if ON_KAGGLE
    else (_REPO / "selected_patient_splits.json")
)


def env_path(name: str, default: Path) -> Path:
    val = os.environ.get(name)
    return Path(val) if val else default


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="C-MLP Phase 3 training")

    # --- Data paths ---
    p.add_argument("--feature-cache", type=Path,
                   default=env_path("FEATURE_CACHE", DEFAULT_FEATURE_CACHE))
    p.add_argument("--metadata", type=Path,
                   default=env_path("METADATA", DEFAULT_METADATA))
    p.add_argument("--bbox-dir", type=Path,
                   default=env_path("BBOX_DIR", DEFAULT_BBOX_DIR))
    p.add_argument("--splits", type=Path,
                   default=env_path("SPLITS_JSON", DEFAULT_SPLITS))
    p.add_argument("--output-dir", type=Path,
                   default=env_path("CMLP_OUTPUT", WORK_ROOT))

    # --- Architecture ---
    p.add_argument("--head", choices=["mlp", "kan"], default="mlp",
                   help="Disease head type (default: mlp; kan = ablation)")
    p.add_argument("--pool", choices=["attn", "mean"], default="attn",
                   help="Region pooling strategy (default: attn = attention pool)")
    p.add_argument("--global-branch", choices=["on", "off"], default="on",
                   help="Global branch for relation-based findings")
    p.add_argument("--proj-dim", type=int, default=PROJ_DIM,
                   help=f"Neck projection dim (default: {PROJ_DIM})")
    p.add_argument("--head-hidden", type=int, default=64,
                   help="Hidden dim in disease MLPHead (default: 64)")

    # --- Loss ---
    p.add_argument("--loss", choices=["bce", "focal"], default="bce",
                   help="Loss type for disease classification")
    p.add_argument("--pos-weight", choices=["none", "log", "inverse"], default="log",
                   help="Positive weight strategy (log = RADAR log-scale)")
    p.add_argument("--focal-gamma", type=float, default=2.0)

    # --- Training ---
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=10,
                   help="Early stopping patience (on val macro-F1)")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")

    # --- Misc ---
    p.add_argument("--resume", type=Path, default=None,
                   help="Resume from checkpoint")

    return p


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()
