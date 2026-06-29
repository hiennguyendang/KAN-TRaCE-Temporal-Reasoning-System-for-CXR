"""Dataset for Phase 3 C-MLP training.

Loads pre-computed BioViL-T features ([196, 512] per study) from a cache
directory, plus bounding boxes (YOLO format) and disease labels from metadata.

Feature cache format: one `{dicom_id}.pt` file per study containing a
float16 tensor of shape [196, 512].

BBox format: YOLO `.txt` label files (one per image), each line:
    class_id  cx  cy  w  h  (normalised 0–1)
Converted to x1y1x2y2 format for the model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .constants import (
    ENCODER_DIM,
    LABEL_NAMES,
    MASK_VALUE,
    NUM_DISEASES,
    NUM_PATCHES,
    NUM_REGIONS,
)


def _yolo_to_xyxy(cx: float, cy: float, w: float, h: float):
    """Convert YOLO normalised (cx, cy, w, h) to (x1, y1, x2, y2)."""
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return (x1, y1, x2, y2)


def load_yolo_boxes(
    txt_path: Path,
    num_regions: int = NUM_REGIONS,
) -> torch.Tensor:
    """Load YOLO label file → [29, 4] normalised x1y1x2y2 tensor.

    If a region has no box, its entry is (0,0,0,0) = sentinel.
    If a region has multiple boxes, the largest (by area) is kept.
    """
    boxes = torch.zeros(num_regions, 4)

    if not txt_path.exists():
        return boxes

    best_area: dict[int, float] = {}

    for line in txt_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        if cls_id < 0 or cls_id >= num_regions:
            continue
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        area = w * h
        if cls_id not in best_area or area > best_area[cls_id]:
            best_area[cls_id] = area
            x1, y1, x2, y2 = _yolo_to_xyxy(cx, cy, w, h)
            boxes[cls_id] = torch.tensor([x1, y1, x2, y2])

    return boxes


class CachedFeatureDataset(Dataset):
    """Dataset loading cached BioViL-T features + YOLO boxes + disease labels.

    Parameters
    ----------
    metadata_records : list[dict]
        Each record must have keys: 'dicom_id', 'labels_14' (list of 14 ints,
        with -100 for unknown), and optionally 'image_id' for bbox lookup.
    feature_cache_dir : Path
        Directory containing {dicom_id}.pt files.
    bbox_dir : Path
        Directory containing YOLO {dicom_id}.txt label files.
    """

    def __init__(
        self,
        metadata_records: list[dict[str, Any]],
        feature_cache_dir: Path,
        bbox_dir: Path,
    ):
        self.records = metadata_records
        self.feature_dir = Path(feature_cache_dir)
        self.bbox_dir = Path(bbox_dir)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rec = self.records[idx]
        dicom_id = rec["dicom_id"]

        # --- Features ---
        feat_path = self.feature_dir / f"{dicom_id}.pt"
        if feat_path.exists():
            feats = torch.load(feat_path, map_location="cpu", weights_only=True)
            if feats.dtype == torch.float16:
                feats = feats.float()
            if feats.dim() == 2 and feats.size(0) == 197:
                # Token 0 is CLS token; tokens 1..196 are spatial patch features
                feats = feats[1:, :]
        else:
            # Return zeros if feature not cached (should not happen in practice)
            feats = torch.zeros(NUM_PATCHES, ENCODER_DIM)


        # --- Boxes ---
        bbox_path = self.bbox_dir / f"{dicom_id}.txt"
        boxes = load_yolo_boxes(bbox_path)

        # --- Labels (14-dim, multi-label) ---
        labels_raw = rec.get("labels_14")
        if labels_raw is not None:
            labels = torch.tensor(labels_raw, dtype=torch.long)
        else:
            # Try to build from individual label fields
            labels = torch.full((NUM_DISEASES,), MASK_VALUE, dtype=torch.long)
            for i, name in enumerate(LABEL_NAMES):
                key = name.lower().replace(" ", "_")
                if key in rec:
                    val = rec[key]
                    if val in (0, 1):
                        labels[i] = val

        return {
            "feats": feats,          # [196, 512]
            "boxes": boxes,          # [29, 4]
            "labels": labels,        # [14]
            "dicom_id": dicom_id,
        }


def load_metadata_jsonl(
    metadata_path: Path,
    split: str | None = None,
    splits_json: Path | None = None,
) -> list[dict]:
    """Load metadata JSONL, optionally filtering by split.

    If splits_json is given, use patient-level splits from that file.
    Otherwise use the 'split' field in metadata.
    """
    import json as _json

    records = []
    patient_split_map: dict[str, str] | None = None

    if splits_json is not None and splits_json.exists():
        with open(splits_json) as f:
            raw = _json.load(f)
        # Flatten: {split_name: [patient_ids]} → {patient_id: split_name}
        patient_split_map = {}
        for sname, pids in raw.items():
            for pid in pids:
                patient_split_map[str(pid)] = sname

    with open(metadata_path) as f:
        for line in f:
            rec = _json.loads(line)
            if split is not None:
                if patient_split_map is not None:
                    pid = str(rec.get("patient_id", rec.get("subject_id", "")))
                    rec_split = patient_split_map.get(pid)
                else:
                    rec_split = rec.get("split")
                if rec_split != split:
                    continue
            records.append(rec)

    return records
