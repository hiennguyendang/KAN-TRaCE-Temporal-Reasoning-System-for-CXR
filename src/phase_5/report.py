"""Report dataclasses: Finding and M5Report.

Schema from phase_5_new.md §5 — designed for unambiguous serialisation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Finding:
    """A single finding in the structured report."""

    disease: str                     # ∈ LABEL_NAMES (14)
    region: str                      # ∈ REGION_NAMES (29) — representative region
    bbox: list[float]                # [x1, y1, x2, y2] — resized 512 space
    bbox_original: list[float]       # [x1, y1, x2, y2] — original pixel space
    p_cal: float                     # calibrated confidence (image-level)
    status: str                      # "assert" | "hedge"
    region_attr: float               # softmax_r(region_logit[r,d]) for this region
    is_hinge: bool = False           # True if region counterfactual flips the call
    progression: str | None = None   # "improved" | "stable" | "worsened" | None
    prog_conf: float | None = None   # calibrated progression confidence
    flag_review: bool = False        # True → "model uncertain, radiologist please check"


@dataclass
class M5Report:
    """Complete structured report output of Module 5."""

    findings: list[Finding] = field(default_factory=list)
    review_regions: list[str] = field(default_factory=list)  # regions needing attention
    has_prior: bool = False
    prose: str | None = None         # realised text (after verify)
    verify_passed: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialise to JSON-friendly dict."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> M5Report:
        findings = [Finding(**f) for f in data.get("findings", [])]
        return cls(
            findings=findings,
            review_regions=data.get("review_regions", []),
            has_prior=data.get("has_prior", False),
            prose=data.get("prose"),
            verify_passed=data.get("verify_passed", True),
        )

    @classmethod
    def from_json(cls, json_str: str) -> M5Report:
        return cls.from_dict(json.loads(json_str))

    def asserted_diseases(self) -> set[str]:
        """Set of diseases that are asserted or hedged."""
        return {f.disease for f in self.findings if f.status in ("assert", "hedge")}

    def summary(self) -> str:
        """Short human-readable summary."""
        lines = []
        if not self.findings:
            lines.append("No significant findings.")
        for f in self.findings:
            s = f"[{f.status.upper()}] {f.disease} at {f.region} (p={f.p_cal:.2f})"
            if f.progression:
                s += f" — {f.progression}"
            if f.flag_review:
                s += " ⚠ REVIEW"
            lines.append(s)
        return "\n".join(lines)
