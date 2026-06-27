"""Constants for Phase 5 — Faithful Report Assembler.

Imports shared constants from Phase 3 and adds M5-specific defaults.
"""

from __future__ import annotations

from phase_3.constants import (
    LABEL_NAMES,
    LABEL_TO_ID,
    MASK_VALUE,
    NUM_DISEASES,
    NUM_PROG,
    NUM_REGIONS,
    PROG_CLASSES,
    REGION_NAMES,
    REGION_TO_ID,
)

# Re-export for convenience
__all__ = [
    "LABEL_NAMES", "LABEL_TO_ID", "NUM_DISEASES",
    "REGION_NAMES", "REGION_TO_ID", "NUM_REGIONS",
    "PROG_CLASSES", "NUM_PROG", "MASK_VALUE",
    "DEFAULT_THRESHOLDS",
]

# ---------------------------------------------------------------------------
# Default thresholds for calibration / abstention (§3-C of phase_5_new.md).
# These are STARTING VALUES — tune on validation set per-disease.
# ---------------------------------------------------------------------------
DEFAULT_THRESHOLDS = {
    "tau_assert": 0.7,        # p_cal >= τ_assert → assert (khẳng định)
    "tau_uncertain": 0.3,     # τ_uncertain <= p_cal < τ_assert → hedge
    # p_cal < τ_uncertain → omit (không nhắc)
    "tau_prog": 0.6,          # progression confidence threshold
}

# ---------------------------------------------------------------------------
# Diseases flagged for mandatory review when in "hedge" band.
# (Dangerous / difficult findings that need radiologist attention.)
# ---------------------------------------------------------------------------
CRITICAL_DISEASES: frozenset[str] = frozenset({
    "Pneumothorax",
    "Pneumonia",
    "Consolidation",
    "Fracture",
    "Lung Lesion",
})
