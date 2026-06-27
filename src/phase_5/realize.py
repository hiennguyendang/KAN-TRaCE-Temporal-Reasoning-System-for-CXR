"""Realize: Convert structured findings to text.

Tầng E of phase_5_new.md:
- Default: TEMPLATE (faithful, zero hallucination)
- Optional: LLM constrained paraphraser (only paraphrase, no new findings)
"""

from __future__ import annotations

from .report import Finding, M5Report


# ---------------------------------------------------------------------------
# Template realize (default, faithful)
# ---------------------------------------------------------------------------

def _finding_to_template(f: Finding, include_progression: bool) -> str:
    """Convert a single Finding to template text."""
    parts = []

    # Status phrasing
    if f.status == "assert":
        parts.append(f"{f.disease}")
    else:  # hedge
        parts.append(f"Possible {f.disease}, cannot be excluded")

    # Region
    parts.append(f"at {f.region}")

    # Confidence
    parts.append(f"(confidence: {f.p_cal:.0%})")

    # Progression (only if has_prior and progression is set)
    if include_progression and f.progression is not None:
        parts.append(f"— {f.progression} compared to prior")
        if f.prog_conf is not None:
            parts[-1] += f" ({f.prog_conf:.0%})"

    # Review flag
    if f.flag_review:
        parts.append("⚠ radiologist review recommended")

    return ", ".join(parts) + "."


def realize_template(report: M5Report) -> str:
    """Generate template-based report text (faithful, zero hallucination).

    Format:
        FINDINGS:
        1. [ASSERT] Cardiomegaly at cardiac silhouette (confidence: 92%) — stable compared to prior.
        2. [HEDGE] Possible Pneumonia at right lower lung zone, cannot be excluded (confidence: 45%) ⚠ ...

        IMPRESSION:
        N findings identified. ...
    """
    if not report.findings:
        return "FINDINGS:\nNo significant findings identified.\n\nIMPRESSION:\nNo acute cardiopulmonary abnormality."

    lines = ["FINDINGS:"]
    for i, f in enumerate(report.findings, 1):
        status_tag = f"[{f.status.upper()}]"
        text = _finding_to_template(f, include_progression=report.has_prior)
        lines.append(f"{i}. {status_tag} {text}")

    # Impression
    n_assert = sum(1 for f in report.findings if f.status == "assert")
    n_hedge = sum(1 for f in report.findings if f.status == "hedge")

    impression_parts = []
    if n_assert > 0:
        diseases = [f.disease for f in report.findings if f.status == "assert"]
        impression_parts.append(f"{n_assert} finding(s) identified: {', '.join(diseases)}.")
    if n_hedge > 0:
        diseases = [f.disease for f in report.findings if f.status == "hedge"]
        impression_parts.append(
            f"{n_hedge} possible finding(s) requiring radiologist review: {', '.join(diseases)}."
        )
    if report.review_regions:
        impression_parts.append(
            f"Regions flagged for review: {', '.join(report.review_regions)}."
        )

    lines.append("")
    lines.append("IMPRESSION:")
    lines.extend(impression_parts)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM constrained paraphraser (optional, future)
# ---------------------------------------------------------------------------

LLM_SYSTEM_PROMPT = """You are a clinical report writer. You will receive a list of radiological findings in structured format. Your job is to rephrase them into natural clinical prose following radiology reporting conventions.

RULES (ABSOLUTE):
1. Do NOT add any finding not in the list.
2. Do NOT remove any finding from the list.
3. Do NOT change the anatomical region of any finding.
4. Do NOT change progression words (improved/stable/worsened).
5. Only rephrase the language — the clinical content must be identical.
6. Use standard radiology report structure (FINDINGS + IMPRESSION).
"""


def realize_llm(report: M5Report, llm_fn=None) -> str:
    """Generate LLM-paraphrased report text (constrained).

    Parameters
    ----------
    report : M5Report
        Structured report.
    llm_fn : callable, optional
        Function(system_prompt, user_prompt) → str.
        If None, falls back to template.

    Returns
    -------
    prose : str
    """
    if llm_fn is None:
        return realize_template(report)

    # Build user prompt from template output
    template_text = realize_template(report)
    user_prompt = (
        "Rephrase the following structured radiology report into natural clinical prose. "
        "Follow the RULES in your system prompt strictly.\n\n"
        f"{template_text}"
    )

    try:
        prose = llm_fn(LLM_SYSTEM_PROMPT, user_prompt)
    except Exception as e:
        # Fall back to template on LLM failure
        prose = template_text

    return prose


def realize(report: M5Report, mode: str = "template", llm_fn=None) -> M5Report:
    """Add prose to report using specified mode.

    Parameters
    ----------
    mode : 'template' (default) or 'llm'
    """
    if mode == "template":
        report.prose = realize_template(report)
    elif mode == "llm":
        report.prose = realize_llm(report, llm_fn=llm_fn)
    else:
        raise ValueError(f"Unknown realize mode: {mode!r}")

    return report
