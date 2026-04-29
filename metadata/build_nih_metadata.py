from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from metadata.nih_metadata import build_nih_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build standardized NIH metadata JSONL")
    parser.add_argument(
        "--metadata-json",
        type=Path,
        default=Path(r"C:\Users\dhint\CHEX-DATA\NIH\metadata.json"),
        help="Path to NIH metadata JSON",
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=Path(r"C:\Users\dhint\CHEX-DATA\NIH\Data_Entry_2017.csv"),
        help="Path to NIH Data_Entry_2017.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"C:\Users\dhint\CHEX-DATA\NIH\metadata"),
        help="Output directory for JSONL and helper files",
    )
    parser.add_argument(
        "--output-filename",
        type=str,
        default="nih_metadata.jsonl",
        help="Output JSONL filename",
    )
    return parser.parse_args()


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("nih_metadata")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    return logger


def main() -> int:
    args = parse_args()
    logger = setup_logger()

    build_nih_metadata(
        metadata_json=args.metadata_json,
        labels_csv=args.labels_csv,
        output_dir=args.output_dir,
        output_filename=args.output_filename,
        logger=logger,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())