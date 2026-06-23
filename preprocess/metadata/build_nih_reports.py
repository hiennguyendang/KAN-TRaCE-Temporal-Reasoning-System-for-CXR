from __future__ import annotations

import argparse
import json
from pathlib import Path

LABEL_TERM_MAP = {
    0: "atelectasis",
    1: "cardiomegaly",
    2: "pleural effusion",
    3: "infiltration",
    4: "mass",
    5: "nodule",
    6: "pneumonia",
    7: "pneumothorax",
    8: "consolidation",
    9: "edema",
    10: "emphysema",
    11: "fibrosis",
    12: "pleural thickening",
    13: "hernia",
}

NORMAL_REPORT = (
    "The chest X-ray is normal. "
    "No significant findings or acute cardiopulmonary abnormalities are seen."
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic NIH reports from label vectors")
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=Path(r"C:\Users\dhint\CHEX-DATA\NIH\metadata\nih_metadata.jsonl"),
        help="Input NIH metadata JSONL path",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path(r"C:\Users\dhint\CHEX-DATA\NIH\metadata\nih_metadata_with_reports.jsonl"),
        help="Output NIH metadata JSONL path with generated reports",
    )
    return parser.parse_args()


def labels_to_report(labels: list[int]) -> str:
    if sum(labels) == 0:
        return compose_report("", NORMAL_REPORT)

    findings = [LABEL_TERM_MAP[idx] for idx, value in enumerate(labels) if value == 1 and idx in LABEL_TERM_MAP]
    if not findings:
        return compose_report("", NORMAL_REPORT)

    if len(findings) == 1:
        impression_text = f"The chest X-ray shows evidence of {findings[0]}."
    elif len(findings) == 2:
        impression_text = f"The chest X-ray shows evidence of {findings[0]} and {findings[1]}."
    else:
        finding_list = ", ".join(findings[:-1]) + f", and {findings[-1]}"
        impression_text = f"The chest X-ray shows evidence of {finding_list}."

    return compose_report("", impression_text)


def compose_report(findings: str, impression: str) -> str:
    return f"FINDINGS: {findings}; IMPRESSION: {impression};"


def process_jsonl(input_jsonl: Path, output_jsonl: Path) -> tuple[int, int]:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0

    with input_jsonl.open("r", encoding="utf-8") as src, output_jsonl.open("w", encoding="utf-8") as dst:
        for raw_line in src:
            line = raw_line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            labels = row.get("labels")
            if not isinstance(labels, list):
                skipped += 1
                continue

            normalized_labels = [1 if int(value) > 0 else 0 for value in labels[:14]]
            if len(normalized_labels) < 14:
                normalized_labels.extend([0] * (14 - len(normalized_labels)))

            row["findings"] = ""
            row["report"] = labels_to_report(normalized_labels)
            dst.write(json.dumps(row, ensure_ascii=False))
            dst.write("\n")
            processed += 1

    return processed, skipped


def main() -> int:
    args = parse_args()
    processed, skipped = process_jsonl(args.input_jsonl.resolve(), args.output_jsonl.resolve())
    print(f"Processed rows: {processed}")
    print(f"Skipped rows: {skipped}")
    print(f"Output: {args.output_jsonl.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
