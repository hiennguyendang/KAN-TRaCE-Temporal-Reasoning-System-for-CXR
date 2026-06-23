from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


STANDARD_REPORT_RE = re.compile(
    r"^FINDINGS:\s*(?P<findings>.*?)\s*;\s*IMPRESSION:\s*(?P<impression>.*?)\s*;\s*$",
    re.IGNORECASE | re.DOTALL,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize existing metadata JSONL files")
    parser.add_argument(
        "--mimic-input",
        type=Path,
        default=Path(r"C:\Users\dhint\CHEX-DATA\MIMIC-CXR\metadata\mimic_metadata.jsonl"),
        help="Existing MIMIC metadata JSONL",
    )
    parser.add_argument(
        "--mimic-output",
        type=Path,
        default=Path(r"C:\Users\dhint\CHEX-DATA\MIMIC-CXR\metadata\mimic_metadata_normalized.jsonl"),
        help="Normalized MIMIC output JSONL",
    )
    parser.add_argument(
        "--chexplus-input",
        type=Path,
        default=Path(r"C:\Users\dhint\CHEX-DATA\CHEXPLUS\metadata\chexplus_metadata.jsonl"),
        help="Existing CheXplus metadata JSONL",
    )
    parser.add_argument(
        "--chexplus-output",
        type=Path,
        default=Path(r"C:\Users\dhint\CHEX-DATA\CHEXPLUS\metadata\chexplus_metadata_normalized.jsonl"),
        help="Normalized CheXplus output JSONL",
    )
    parser.add_argument(
        "--nih-input",
        type=Path,
        default=Path(r"C:\Users\dhint\CHEX-DATA\NIH\metadata\nih_metadata_with_reports_v2.jsonl"),
        help="Existing NIH metadata-with-reports JSONL",
    )
    parser.add_argument(
        "--nih-output",
        type=Path,
        default=Path(r"C:\Users\dhint\CHEX-DATA\NIH\metadata\nih_metadata_with_reports_normalized.jsonl"),
        help="Normalized NIH output JSONL",
    )
    return parser.parse_args()


def normalize_mimic_record(row: dict[str, Any]) -> dict[str, Any]:
    findings, impression = extract_sections_from_raw_report(row.get("report", ""))
    return rebuild_record(
        row,
        findings=findings,
        report=compose_standard_report(findings, impression),
    )


def normalize_chexplus_record(row: dict[str, Any]) -> dict[str, Any]:
    findings, impression = parse_report_sections(row.get("report", ""))
    return rebuild_record(
        row,
        findings=findings,
        report=compose_standard_report(findings, impression),
    )


def normalize_nih_record(row: dict[str, Any]) -> dict[str, Any]:
    impression = labels_to_impression(row.get("labels", []))

    return rebuild_record(
        row,
        findings="",
        report=compose_standard_report("", impression),
    )


def rebuild_record(
    row: dict[str, Any],
    *,
    findings: str,
    report: str,
) -> dict[str, Any]:
    normalized: dict[str, Any] = {}

    for key, value in row.items():
        if key in {"finding", "findings", "report"}:
            continue
        normalized[key] = value

    normalized["findings"] = findings
    normalized["report"] = report
    return normalized


def process_jsonl(input_path: Path, output_path: Path, normalizer) -> tuple[int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0

    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for raw_line in src:
            line = raw_line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            if not isinstance(row, dict):
                skipped += 1
                continue

            normalized = normalizer(row)
            dst.write(json.dumps(normalized, ensure_ascii=False))
            dst.write("\n")
            processed += 1

    return processed, skipped


def parse_standard_report(report: Any) -> tuple[str, str]:
    return parse_report_sections(report)


def parse_report_sections(report: Any) -> tuple[str, str]:
    raw_text = "" if report is None else str(report)
    if not raw_text.strip():
        return "", ""

    semicolon_match = STANDARD_REPORT_RE.match(normalize_report_string(raw_text))
    if semicolon_match:
        findings = normalize_report_string(semicolon_match.group("findings"))
        impression = normalize_report_string(semicolon_match.group("impression"))
        return findings, impression

    multiline_match = re.search(
        r"findings:\s*(?P<findings>.*?)(?=\n\s*impression:\s*|$)\s*\n\s*impression:\s*(?P<impression>.*)$",
        raw_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if multiline_match:
        findings = normalize_report_string(multiline_match.group("findings"))
        impression = normalize_report_string(multiline_match.group("impression"))
        return findings, impression

    return "", normalize_report_string(raw_text)


def extract_sections_from_raw_report(raw_report: Any) -> tuple[str, str]:
    if not isinstance(raw_report, str) or not raw_report.strip():
        return "", ""

    text = raw_report
    lower_text = text.lower()

    findings_match = re.search(r"findings:\s*(.*?)(?=(?:\n\s*[a-z][a-z\s]*:)|$)", lower_text, re.DOTALL)
    impression_match = re.search(r"impression:\s*(.*?)(?=(?:\n\s*[a-z][a-z\s]*:)|$)", lower_text, re.DOTALL)

    findings = normalize_report_string(findings_match.group(1)) if findings_match else ""
    impression = normalize_report_string(impression_match.group(1)) if impression_match else ""

    if findings or impression:
        return findings, impression

    cleaned = normalize_report_string(text)
    return cleaned, ""


def compose_standard_report(findings: str, impression: str) -> str:
    return f"FINDINGS: {findings}; IMPRESSION: {impression};"


def labels_to_impression(labels: Any) -> str:
    if not isinstance(labels, list):
        return ""

    normalized = [1 if int(value) > 0 else 0 for value in labels[:14]]
    if len(normalized) < 14:
        normalized.extend([0] * (14 - len(normalized)))

    if sum(normalized) == 0:
        return (
            "The chest X-ray is normal. "
            "No significant findings or acute cardiopulmonary abnormalities are seen."
        )

    terms = [
        "atelectasis",
        "cardiomegaly",
        "pleural effusion",
        "infiltration",
        "mass",
        "nodule",
        "pneumonia",
        "pneumothorax",
        "consolidation",
        "edema",
        "emphysema",
        "fibrosis",
        "pleural thickening",
        "hernia",
    ]
    findings = [term for idx, term in enumerate(terms) if normalized[idx] == 1]
    if len(findings) == 1:
        return f"The chest X-ray shows evidence of {findings[0]}."
    if len(findings) == 2:
        return f"The chest X-ray shows evidence of {findings[0]} and {findings[1]}."

    return f"The chest X-ray shows evidence of {', '.join(findings[:-1])}, and {findings[-1]}."


def normalize_report_string(value: Any | None) -> str:
    if value is None:
        return ""
    text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main() -> int:
    args = parse_args()

    mimic_processed, mimic_skipped = process_jsonl(args.mimic_input.resolve(), args.mimic_output.resolve(), normalize_mimic_record)
    chexplus_processed, chexplus_skipped = process_jsonl(
        args.chexplus_input.resolve(),
        args.chexplus_output.resolve(),
        normalize_chexplus_record,
    )
    nih_processed, nih_skipped = process_jsonl(args.nih_input.resolve(), args.nih_output.resolve(), normalize_nih_record)

    print(f"MIMIC processed: {mimic_processed}, skipped: {mimic_skipped}")
    print(f"CheXplus processed: {chexplus_processed}, skipped: {chexplus_skipped}")
    print(f"NIH processed: {nih_processed}, skipped: {nih_skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())