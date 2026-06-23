from __future__ import annotations

import csv
import dbm.dumb
import json
import logging
import re
from pathlib import Path
from typing import Any


LABEL_NAMES_14 = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
]

IGNORE_INDEX = -100.0
MIMIC_TO_14 = {index: index for index in range(14)}

INPUT_METADATA_FILES = {
    "mimic": r"C:\Users\dhint\CHEX-DATA\MIMIC-CXR\metadata\mimic_metadata_normalized.jsonl",
    "chexplus": r"C:\Users\dhint\CHEX-DATA\CHEXPLUS\metadata\chexplus_metadata_normalized.jsonl",
}

SOURCE_LABEL_FILES = {
    "mimic": r"C:\Users\dhint\CHEX-DATA\MIMIC-CXR\mimic-cxr-2.0.0-chexpert.csv",
    "chexplus": r"C:\Users\dhint\CHEX-DATA\CHEXPLUS\chexbert_labels\findings_fixed.json",
}

OUTPUT_FILES = {
    "mimic": r"C:\Users\dhint\CHEX-DATA\MIMIC-CXR\metadata\mimic_metadata_unified_14dim.jsonl",
    "chexplus": r"C:\Users\dhint\CHEX-DATA\CHEXPLUS\metadata\chexplus_metadata_unified_14dim.jsonl",
}

PATIENT_TOKEN_RE = re.compile(r"patient\d+", re.IGNORECASE)
STUDY_TOKEN_RE = re.compile(r"study\d+", re.IGNORECASE)
VIEW_TOKEN_RE = re.compile(r"view\d+_(?:frontal|lateral)", re.IGNORECASE)


def normalize_text(value: Any | None) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_subject_id(value: str) -> str:
    cleaned = normalize_text(value).lower()
    if cleaned.startswith("p") and cleaned[1:].isdigit():
        return cleaned[1:]
    return cleaned


def normalize_study_id(value: str) -> str:
    cleaned = normalize_text(value).lower()
    if cleaned.startswith("s") and cleaned[1:].isdigit():
        return cleaned[1:]
    return cleaned


def normalize_field_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", normalize_text(value).lower())


def parse_tristate_label(value: Any | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)):
        number = int(float(value))
        return number if number in (-1, 0, 1) else None

    text = normalize_text(value)
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"nan", "none", "null", "na", "n/a", ""}:
        return None
    try:
        number = int(float(text))
    except ValueError:
        return None
    return number if number in (-1, 0, 1) else None


def convert_mimic_like_to_14(source_labels: list[int | None]) -> list[float]:
    unified = [IGNORE_INDEX] * len(LABEL_NAMES_14)
    for source_index, unified_index in MIMIC_TO_14.items():
        raw = source_labels[source_index] if source_index < len(source_labels) else None
        if raw == 1:
            unified[unified_index] = 1.0
        elif raw == 0:
            unified[unified_index] = 0.0
        else:
            unified[unified_index] = IGNORE_INDEX
    return unified


def load_json_or_jsonl(path: Path) -> Any:
    raw = path.read_text(encoding="utf-8")
    stripped = raw.lstrip()
    if not stripped:
        return []

    if stripped[0] in "[{":
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

    rows: list[Any] = []
    for raw_line in raw.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def iter_payload_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]

    if isinstance(payload, dict):
        for key in ("data", "labels", "items", "entries", "records"):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
        if payload:
            return [payload]

    return []


def extract_token(pattern: re.Pattern[str], text: str) -> str:
    match = pattern.search(text.replace("\\", "/"))
    return match.group(0).lower() if match else ""


def extract_chexplus_key_from_path(path_text: str) -> tuple[str, str, str]:
    normalized = normalize_text(path_text)
    patient_id = extract_token(PATIENT_TOKEN_RE, normalized)
    study_id = extract_token(STUDY_TOKEN_RE, normalized)
    view = extract_token(VIEW_TOKEN_RE, normalized)
    return patient_id, study_id, view


def extract_chexplus_key_from_row(row: dict[str, Any]) -> tuple[str, str, str]:
    patient_id = normalize_text(row.get("patient_id")).lower()
    study_id = normalize_text(row.get("study_id")).lower()
    view = normalize_text(row.get("view")).lower()
    if patient_id and study_id and view:
        return patient_id, study_id, view

    for key in ("path_to_image", "image_path", "path", "file_path", "image", "filename"):
        value = normalize_text(row.get(key))
        if value:
            return extract_chexplus_key_from_path(value)

    return "", "", ""


def extract_chexplus_label_vector(row: dict[str, Any]) -> list[int | None] | None:
    labels_field = row.get("labels")
    if isinstance(labels_field, list) and len(labels_field) >= 14:
        return [parse_tristate_label(value) for value in labels_field[:14]]

    label_field_map = {normalize_field_name(key): key for key in row.keys()}
    ordered_fields: list[str] = []
    for label_name in LABEL_NAMES_14:
        normalized_label = normalize_field_name(label_name)
        source_field = label_field_map.get(normalized_label)
        if source_field:
            ordered_fields.append(source_field)

    if len(ordered_fields) == 14:
        return [parse_tristate_label(row.get(field)) for field in ordered_fields]

    excluded = {
        "path_to_image",
        "image_path",
        "path",
        "file_path",
        "image",
        "filename",
        "patient_id",
        "study_id",
        "view",
        "dataset",
        "image_id",
        "label",
        "text",
        "findings",
        "finding",
        "report",
    }
    fallback_values: list[int | None] = []
    for key, value in row.items():
        if key in excluded:
            continue
        parsed = parse_tristate_label(value)
        if parsed is None and normalize_text(value):
            continue
        fallback_values.append(parsed)

    if len(fallback_values) >= 14:
        return fallback_values[:14]

    return None


def load_mimic_source_index(labels_csv: Path) -> dict[tuple[str, str], list[int | None]]:
    index: dict[tuple[str, str], list[int | None]] = {}

    with labels_csv.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            subject_id = normalize_subject_id(row.get("subject_id", ""))
            study_id = normalize_study_id(row.get("study_id", ""))
            if not subject_id or not study_id:
                continue

            labels = [parse_tristate_label(row.get(name)) for name in LABEL_NAMES_14]
            index[(subject_id, study_id)] = labels

    return index


def chexplus_db_base_path(db_path: Path) -> Path:
    return db_path


def chexplus_db_paths(db_path: Path) -> list[Path]:
    base = chexplus_db_base_path(db_path)
    return [Path(f"{base}{suffix}") for suffix in (".dat", ".dir", ".bak")]


def remove_chexplus_db_files(db_path: Path) -> None:
    for candidate in chexplus_db_paths(db_path):
        if candidate.exists():
            candidate.unlink()


def load_chexplus_source_index(source_path: Path) -> dict[tuple[str, str, str], list[int | None]]:
    index: dict[tuple[str, str, str], list[int | None]] = {}

    if source_path.suffix.lower() == ".csv":
        with source_path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            for row in reader:
                key = extract_chexplus_key_from_row(row)
                if not all(key):
                    continue
                labels = extract_chexplus_label_vector(row)
                if labels is None:
                    continue
                if key not in index:
                    index[key] = labels
        return index

    with source_path.open("r", encoding="utf-8") as stream:
        for raw_line in stream:
            line = raw_line.strip()
            if not line:
                continue
            parsed = json.loads(line)
            if not isinstance(parsed, dict):
                continue

            key = extract_chexplus_key_from_row(parsed)
            if not all(key):
                continue

            labels = extract_chexplus_label_vector(parsed)
            if labels is None:
                continue

            if key not in index:
                index[key] = labels

    return index


def chexplus_key_to_text(key: tuple[str, str, str]) -> str:
    return "|".join(key)


def build_chexplus_source_db(source_path: Path, db_path: Path, logger: logging.Logger) -> int:
    remove_chexplus_db_files(db_path)

    inserted = 0
    malformed = 0
    batch_size = 5000
    batch: list[tuple[str, str]] = []

    with dbm.dumb.open(str(chexplus_db_base_path(db_path)), "c") as db, source_path.open("r", encoding="utf-8") as stream:
        for raw_line in stream:
            line = raw_line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
                if not isinstance(parsed, dict):
                    continue

                key = extract_chexplus_key_from_row(parsed)
                if not all(key):
                    continue

                labels = extract_chexplus_label_vector(parsed)
                if labels is None:
                    continue

                batch.append((chexplus_key_to_text(key), json.dumps(labels)))

                if len(batch) >= batch_size:
                    for key_text, labels_text in batch:
                        if key_text not in db:
                            db[key_text] = labels_text
                            inserted += 1
                    batch.clear()
                    if inserted and inserted % 50000 == 0:
                        logger.info("  CheXplus source indexed %d rows", inserted)
            except BaseException as exc:
                malformed += 1
                logger.warning("CheXplus source row skipped due to error: %s", exc)
                continue

        if batch:
            for key_text, labels_text in batch:
                if key_text not in db:
                    db[key_text] = labels_text
                    inserted += 1

    if malformed:
        logger.warning("CheXplus source had %d malformed JSON lines (skipped)", malformed)

    return inserted


def process_mimic(
    input_path: Path,
    output_path: Path,
    source_index: dict[tuple[str, str], list[int | None]],
    logger: logging.Logger,
) -> tuple[int, int, int]:
    processed = 0
    skipped = 0
    missing_source = 0

    with input_path.open("r", encoding="utf-8") as input_stream, output_path.open("w", encoding="utf-8") as output_stream:
        for raw_line in input_stream:
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            patient_id = normalize_subject_id(record.get("patient_id", ""))
            study_id = normalize_study_id(record.get("study_id", ""))
            source_labels = source_index.get((patient_id, study_id))

            if source_labels is None:
                missing_source += 1
                unified_labels = [IGNORE_INDEX] * len(LABEL_NAMES_14)
            else:
                unified_labels = convert_mimic_like_to_14(source_labels)

            record["labels"] = unified_labels
            output_stream.write(json.dumps(record, ensure_ascii=False))
            output_stream.write("\n")
            processed += 1

            if processed % 50000 == 0:
                logger.info("  MIMIC processed %d records", processed)

    logger.info("MIMIC done: processed=%d skipped=%d missing_source=%d", processed, skipped, missing_source)
    return processed, skipped, missing_source


def process_chexplus(
    input_path: Path,
    output_path: Path,
    source_db_path: Path,
    logger: logging.Logger,
) -> tuple[int, int, int]:
    processed = 0
    skipped = 0
    missing_source = 0

    with dbm.dumb.open(str(chexplus_db_base_path(source_db_path)), "r") as db:
        with input_path.open("r", encoding="utf-8") as input_stream, output_path.open("w", encoding="utf-8") as output_stream:
            for line_num, raw_line in enumerate(input_stream, start=1):
                try:
                    record = json.loads(raw_line)
                    key = extract_chexplus_key_from_row(record)
                    key_text = chexplus_key_to_text(key) if all(key) else ""
                    source_labels: list[int | None] | None = None
                    if key_text and key_text in db:
                        source_labels = json.loads(db[key_text].decode("utf-8"))

                    if source_labels is None:
                        missing_source += 1
                        unified_labels = [IGNORE_INDEX] * len(LABEL_NAMES_14)
                    else:
                        unified_labels = convert_mimic_like_to_14(source_labels)

                    record["labels"] = unified_labels
                    output_stream.write(json.dumps(record, ensure_ascii=False))
                    output_stream.write("\n")
                    processed += 1

                    if processed % 50000 == 0:
                        logger.info("  CheXplus processed %d records", processed)
                except BaseException as exc:
                    skipped += 1
                    logger.warning("CheXplus row %d skipped due to error: %s", line_num, exc)
                    continue

    logger.info("CheXplus done: processed=%d skipped=%d missing_source=%d", processed, skipped, missing_source)
    return processed, skipped, missing_source


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("REBUILD LABELS FROM RAW SOURCES -> UNIFIED 14 DIM")
    logger.info("=" * 80)

    input_paths = {name: Path(path) for name, path in INPUT_METADATA_FILES.items()}
    source_paths = {name: Path(path) for name, path in SOURCE_LABEL_FILES.items()}
    output_paths = {name: Path(path) for name, path in OUTPUT_FILES.items()}

    enabled_datasets: list[str] = []
    for name in ("mimic", "chexplus"):
        if not input_paths[name].exists():
            logger.warning("Skip %s: missing input metadata file: %s", name, input_paths[name])
            continue
        if not source_paths[name].exists():
            logger.warning("Skip %s: missing source label file: %s", name, source_paths[name])
            continue
        ensure_parent(output_paths[name])
        enabled_datasets.append(name)

    if not enabled_datasets:
        logger.error("No dataset can be processed. Please check INPUT_METADATA_FILES and SOURCE_LABEL_FILES.")
        return

    total_processed = 0
    total_skipped = 0
    total_missing_source = 0

    if "mimic" in enabled_datasets:
        logger.info("Loading source labels for mimic...")
        mimic_source = load_mimic_source_index(source_paths["mimic"])
        logger.info("Loaded mimic source rows: %d", len(mimic_source))
        processed, skipped, missing_source = process_mimic(
            input_paths["mimic"],
            output_paths["mimic"],
            mimic_source,
            logger,
        )
        total_processed += processed
        total_skipped += skipped
        total_missing_source += missing_source

    if "chexplus" in enabled_datasets:
        logger.info("Loading source labels for chexplus...")
        chexplus_db_path = output_paths["chexplus"].with_suffix(".source_labels.dbm")
        chexplus_source_count = build_chexplus_source_db(source_paths["chexplus"], chexplus_db_path, logger)
        logger.info("Loaded chexplus source rows: %d", chexplus_source_count)
        processed, skipped, missing_source = process_chexplus(
            input_paths["chexplus"],
            output_paths["chexplus"],
            chexplus_db_path,
            logger,
        )
        total_processed += processed
        total_skipped += skipped
        total_missing_source += missing_source
        remove_chexplus_db_files(chexplus_db_path)

    logger.info("=" * 80)
    logger.info(
        "TOTAL: processed=%d skipped=%d missing_source=%d",
        total_processed,
        total_skipped,
        total_missing_source,
    )
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
