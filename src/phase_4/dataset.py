"""Dataset for Phase 4 T-MLP training.

Loads temporal pairs: (prior_study, current_study) with progression labels.
Each sample provides:
- prior_feat [29, 128] + curr_feat [29, 128]  (from C-MLP region_feat)
- prior_labels [14] + curr_labels [14]  (disease labels)
- prog_labels [29, 14]  (progression: 0=improved, 1=stable, 2=worsened, -100=unknown)
- prior_boxes [29, 4] + curr_boxes [29, 4]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from phase_3.constants import MASK_VALUE, NUM_DISEASES, NUM_REGIONS, PROJ_DIM
from phase_3.dataset import load_yolo_boxes


class TemporalPairDataset(Dataset):
    """Dataset for temporal progression training.

    Parameters
    ----------
    pair_records : list[dict]
        Each record has:
        - 'prior_dicom_id', 'curr_dicom_id'
        - 'prior_labels_14' [14], 'curr_labels_14' [14]
        - 'prog_labels' [29, 14] or [14]  (progression per region×disease)
    region_feat_dir : Path
        Directory of {dicom_id}_region_feat.pt files (from C-MLP inference).
        Each file: tensor [29, 128].
    bbox_dir : Path
        YOLO label directory.
    """

    def __init__(
        self,
        pair_records: list[dict[str, Any]],
        region_feat_dir: Path,
        bbox_dir: Path,
    ):
        self.records = pair_records
        self.feat_dir = Path(region_feat_dir)
        self.bbox_dir = Path(bbox_dir)

    def __len__(self) -> int:
        return len(self.records)

    def _load_feat(self, dicom_id: str) -> torch.Tensor:
        """Load region_feat [29, 128] from cache."""
        path = self.feat_dir / f"{dicom_id}_region_feat.pt"
        if path.exists():
            feat = torch.load(path, map_location="cpu", weights_only=True)
            if feat.dtype == torch.float16:
                feat = feat.float()
            return feat
        return torch.zeros(NUM_REGIONS, PROJ_DIM)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rec = self.records[idx]

        prior_id = rec["prior_dicom_id"]
        curr_id = rec["curr_dicom_id"]

        prior_feat = self._load_feat(prior_id)
        curr_feat = self._load_feat(curr_id)

        prior_boxes = load_yolo_boxes(self.bbox_dir / f"{prior_id}.txt")
        curr_boxes = load_yolo_boxes(self.bbox_dir / f"{curr_id}.txt")

        # Labels
        prior_labels = torch.tensor(
            rec.get("prior_labels_14", [MASK_VALUE] * NUM_DISEASES), dtype=torch.long
        )
        curr_labels = torch.tensor(
            rec.get("curr_labels_14", [MASK_VALUE] * NUM_DISEASES), dtype=torch.long
        )

        # Progression labels: ideally [29, 14], but may be [14] (image-level)
        prog_raw = rec.get("prog_labels")
        if prog_raw is not None:
            prog_labels = torch.tensor(prog_raw, dtype=torch.long)
            if prog_labels.dim() == 1:
                # Expand [14] → [29, 14] (same progression for all regions)
                prog_labels = prog_labels.unsqueeze(0).expand(NUM_REGIONS, -1).clone()
        else:
            prog_labels = torch.full(
                (NUM_REGIONS, NUM_DISEASES), MASK_VALUE, dtype=torch.long
            )

        # Region validity masks (derived from boxes)
        prior_mask = (prior_boxes.sum(dim=-1) > 1e-6)  # [29]
        curr_mask = (curr_boxes.sum(dim=-1) > 1e-6)    # [29]

        return {
            "prior_feat": prior_feat,      # [29, 128]
            "curr_feat": curr_feat,        # [29, 128]
            "prior_labels": prior_labels,  # [14]
            "curr_labels": curr_labels,    # [14]
            "prog_labels": prog_labels,    # [29, 14]
            "prior_mask": prior_mask,      # [29]
            "curr_mask": curr_mask,        # [29]
        }


def build_temporal_pairs(
    metadata_path: Path,
    split: str = "train",
    splits_json: Path | None = None,
) -> list[dict]:
    """Build temporal pair records from metadata.

    Groups studies by patient_id, sorted by study_date, and creates pairs
    (consecutive studies).  Progression labels come from comparison_cues in
    scene graph (if available) or are set to MASK_VALUE.
    """
    from phase_3.dataset import load_metadata_jsonl

    records = load_metadata_jsonl(metadata_path, split=split, splits_json=splits_json)

    # Group by patient
    patients: dict[str, list[dict]] = {}
    for rec in records:
        pid = str(rec.get("patient_id", rec.get("subject_id", "")))
        patients.setdefault(pid, []).append(rec)

    pairs = []
    for pid, studies in patients.items():
        # Sort by study date (if available)
        studies.sort(key=lambda r: r.get("study_date", r.get("StudyDate", "")))

        for i in range(len(studies) - 1):
            prior = studies[i]
            curr = studies[i + 1]

            pair = {
                "prior_dicom_id": prior["dicom_id"],
                "curr_dicom_id": curr["dicom_id"],
                "prior_labels_14": prior.get("labels_14"),
                "curr_labels_14": curr.get("labels_14"),
                "prog_labels": None,  # TODO: extract from comparison_cues
                "patient_id": pid,
            }
            pairs.append(pair)

    return pairs
