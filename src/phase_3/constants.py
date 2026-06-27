"""Constants shared across Phase 3, 4, 5.

14 disease labels (CheXbert order) and 29 anatomical regions.
REGION_NAMES is imported from phase_2.constants for single-source-of-truth;
if phase_2 is unavailable (standalone Kaggle run), fall back to a local copy.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 14 CheXbert disease labels — canonical order used in M3/M4/M5.
# ---------------------------------------------------------------------------
LABEL_NAMES: tuple[str, ...] = (
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
)
NUM_DISEASES: int = len(LABEL_NAMES)  # 14
LABEL_TO_ID: dict[str, int] = {n: i for i, n in enumerate(LABEL_NAMES)}

# ---------------------------------------------------------------------------
# 29 anatomical regions (alphabetical order = class-id order).
# ---------------------------------------------------------------------------
try:
    from phase_2.constants import CLASS_NAMES as _P2_NAMES

    REGION_NAMES: tuple[str, ...] = tuple(_P2_NAMES)
except ImportError:
    # Stand-alone / Kaggle: keep a local copy so phase_3 is self-contained.
    REGION_NAMES = tuple(sorted([
        "right lung", "left lung", "mediastinum",
        "right apical zone", "left apical zone",
        "right upper lung zone", "left upper lung zone",
        "right mid lung zone", "left mid lung zone",
        "right lower lung zone", "left lower lung zone",
        "right hilar structures", "left hilar structures",
        "right costophrenic angle", "left costophrenic angle",
        "upper mediastinum", "cardiac silhouette", "trachea",
        "right hemidiaphragm", "left hemidiaphragm",
        "right clavicle", "left clavicle", "spine",
        "right atrium", "cavoatrial junction", "svc",
        "carina", "aortic arch", "abdomen",
    ]))

NUM_REGIONS: int = len(REGION_NAMES)  # 29
REGION_TO_ID: dict[str, int] = {n: i for i, n in enumerate(REGION_NAMES)}

# ---------------------------------------------------------------------------
# Feature dimensions (BioViL-T).
# ---------------------------------------------------------------------------
ENCODER_DIM: int = 512          # BioViL-T patch feature dim
NUM_PATCHES: int = 196          # 14×14 grid
GRID_H: int = 14
GRID_W: int = 14
PROJ_DIM: int = 128             # neck projection dim (default)

# ---------------------------------------------------------------------------
# Progression classes (M4 / T-MLP).
# ---------------------------------------------------------------------------
PROG_CLASSES: tuple[str, ...] = ("improved", "stable", "worsened")
NUM_PROG: int = len(PROG_CLASSES)  # 3

# ---------------------------------------------------------------------------
# Sentinel / masking values.
# ---------------------------------------------------------------------------
MASK_VALUE: int = -100           # ignore index for masked BCE / CE
SENTINEL_BOX: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
