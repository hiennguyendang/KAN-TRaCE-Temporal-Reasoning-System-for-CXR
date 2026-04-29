from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def setup_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("preprocess")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_dir / "preprocess.log", encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def preprocess_image(image_path: Path, output_size: int) -> Image.Image:
    with Image.open(image_path) as img:
        gray = img.convert("L")
        resized = letterbox_resize(gray, output_size)

        arr = np.asarray(resized, dtype=np.float32) / 255.0
        arr_uint8 = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(arr_uint8, mode="L")


def letterbox_resize(image: Image.Image, output_size: int) -> Image.Image:
    src_w, src_h = image.size

    scale = min(output_size / max(src_w, 1), output_size / max(src_h, 1))
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))

    resized = image.resize((new_w, new_h), Image.BILINEAR)

    canvas = Image.new("L", (output_size, output_size), color=0)
    offset_x = (output_size - new_w) // 2
    offset_y = (output_size - new_h) // 2
    canvas.paste(resized, (offset_x, offset_y))
    return canvas


def load_string_set(path: Path) -> set[str]:
    if not path.exists():
        return set()

    raw_text = path.read_text(encoding="utf-8").strip()
    if not raw_text:
        return set()

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        return set()

    if not isinstance(payload, list):
        return set()

    return {str(item) for item in payload}


def save_string_set(path: Path, values: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(values), indent=2), encoding="utf-8")


def load_metadata(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    raw_text = path.read_text(encoding="utf-8").strip()
    if not raw_text:
        return []

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        return []

    if not isinstance(payload, list):
        return []

    return [row for row in payload if isinstance(row, dict)]


def save_metadata(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def ensure_directories(paths: list[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def build_metadata_entry(
    image_id: str,
    patient_id: str,
    study_id: str,
    output_path: Path,
) -> dict[str, Any]:
    return {
        "image_id": image_id,
        "patient_id": patient_id,
        "study_id": study_id,
        "image_path": str(output_path),
    }
