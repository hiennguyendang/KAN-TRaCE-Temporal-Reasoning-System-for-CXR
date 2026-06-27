"""Verify: CheXbert round-trip check (Tầng F).

Ensure the generated report text does not introduce findings
that were not in the M3/M4 structured output (hallucination),
and does not drop findings that were asserted (omission).

Two critical metrics:
- out_of_table_rate ≈ 0  (no hallucinated findings)
- temporal_hallucination_rate = 0  (no temporal language without prior)
"""

from __future__ import annotations

import logging
import re
from typing import Any

from .constants import LABEL_NAMES, PROG_CLASSES
from .report import M5Report

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Temporal language patterns to detect temporal hallucination
# ---------------------------------------------------------------------------
TEMPORAL_PATTERNS = [
    r"\bstable\b",
    r"\bunchanged\b",
    r"\bnew\b",
    r"\bimproved\b",
    r"\bworsened\b",
    r"\bworsening\b",
    r"\bimproving\b",
    r"\bcompared\s+to\s+prior\b",
    r"\bprevious\b",
    r"\bprior\b",
    r"\binterval\b",
    r"\bprogress\w*\b",
    r"\bregress\w*\b",
    r"\bresolved?\b",
    r"\bpersist\w*\b",
]
_TEMPORAL_RE = re.compile("|".join(TEMPORAL_PATTERNS), re.IGNORECASE)


def check_temporal_hallucination(
    report: M5Report,
) -> dict[str, Any]:
    """Check for temporal language when has_prior=False.

    Returns
    -------
    result : dict with keys:
        - temporal_halluc: bool (True = violation found)
        - matches: list of matched temporal words
    """
    if report.has_prior:
        # Temporal language is expected when prior exists
        return {"temporal_halluc": False, "matches": []}

    prose = report.prose or ""
    matches = _TEMPORAL_RE.findall(prose)

    if matches:
        logger.warning(
            "Temporal hallucination detected (has_prior=False): %s", matches
        )

    return {
        "temporal_halluc": len(matches) > 0,
        "matches": matches,
    }


def check_round_trip(
    report: M5Report,
    chexbert_fn=None,
) -> dict[str, Any]:
    """CheXbert round-trip verification (Tầng F).

    Runs CheXbert on the generated prose and compares with the structured
    findings list.  Catches:
    - HALLUCINATION: CheXbert finds a disease in prose that is NOT in findings.
    - OMISSION: A finding is asserted/hedged but CheXbert does not detect it.

    Parameters
    ----------
    chexbert_fn : callable, optional
        Function(report_text: str) → dict[str, str]
        Maps disease names to "positive"/"negative"/"uncertain"/"blank".
        If None, skips CheXbert check (template mode assumed safe).

    Returns
    -------
    result : dict with:
        - round_trip_pass: bool
        - hallucinations: list[str]  (diseases found in text but not in findings)
        - omissions: list[str]  (findings asserted but not detected in text)
    """
    asserted = report.asserted_diseases()
    prose = report.prose or ""

    if chexbert_fn is None:
        # Without CheXbert, do a simple keyword check
        return _keyword_round_trip(prose, asserted)

    # Run CheXbert
    chexbert_labels = chexbert_fn(prose)

    hallucinations = []
    omissions = []

    for disease in LABEL_NAMES:
        in_findings = disease in asserted
        in_text = chexbert_labels.get(disease, "blank") == "positive"

        if in_text and not in_findings:
            hallucinations.append(disease)
        if in_findings and not in_text:
            omissions.append(disease)

    passed = len(hallucinations) == 0

    if hallucinations:
        logger.error("Round-trip HALLUCINATION: %s", hallucinations)
    if omissions:
        logger.warning("Round-trip OMISSION: %s", omissions)

    return {
        "round_trip_pass": passed,
        "hallucinations": hallucinations,
        "omissions": omissions,
    }


def _keyword_round_trip(
    prose: str,
    asserted: set[str],
) -> dict[str, Any]:
    """Simple keyword-based round-trip check (fallback when CheXbert unavailable)."""
    prose_lower = prose.lower()
    hallucinations = []
    omissions = []

    for disease in LABEL_NAMES:
        d_lower = disease.lower()
        mentioned = d_lower in prose_lower
        in_findings = disease in asserted

        # Skip "No Finding" — it's the absence marker
        if disease == "No Finding":
            continue

        if mentioned and not in_findings:
            hallucinations.append(disease)
        if in_findings and not mentioned:
            omissions.append(disease)

    return {
        "round_trip_pass": len(hallucinations) == 0,
        "hallucinations": hallucinations,
        "omissions": omissions,
    }


def verify_report(
    report: M5Report,
    chexbert_fn=None,
) -> M5Report:
    """Full verification pipeline (Tầng F).

    Checks temporal hallucination and round-trip consistency.
    Sets report.verify_passed accordingly.
    """
    # Temporal check
    temporal = check_temporal_hallucination(report)

    # Round-trip check
    rt = check_round_trip(report, chexbert_fn=chexbert_fn)

    # Pass only if both checks pass
    report.verify_passed = (
        not temporal["temporal_halluc"] and rt["round_trip_pass"]
    )

    if not report.verify_passed:
        logger.warning("Report verification FAILED — falling back to template")

    return report
