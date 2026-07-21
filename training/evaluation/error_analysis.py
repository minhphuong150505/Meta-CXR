"""Per-sample error analysis for generated reports.

Every clinical judgement in this module is a **heuristic over surface text**, not
a radiologist's reading and not a validated clinical metric. That is why every
error flag is named ``possible_*``: a lexicon match is evidence worth triaging,
never proof. Presenting these counts as clinical error rates would overstate
what a regex can know.

The finding lexicon, negation cues, severity and location terms are reused from
``safety/claims.py`` rather than duplicated, so the two subsystems cannot drift
apart on what counts as a mention of pneumothorax.

Temporal hallucination
----------------------
MIMIC-CXR reports routinely compare against a prior study. This pipeline feeds
the model a **single** study with no prior, so any comparative statement in the
output is unsupported by the input. :func:`detect_temporal_hallucination` flags
those. It reports ``possible_temporal_hallucination`` because a phrase like
"unchanged" can occasionally be copied from a reference that legitimately had a
prior -- the flag is a screening signal, not a verdict.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from safety.claims import (
    ABNORMALITY_SYNONYMS,
    NEGATION_CUES,
    SEVERITY_TERMS,
    split_sentences,
)
from training.evaluation.generation_metrics import normalize, tokenize

logger = logging.getLogger(__name__)

#: Phrases that assert a comparison with a prior study.
TEMPORAL_CUES = (
    "compared with",
    "compared to",
    "comparison with",
    "since prior",
    "since the prior",
    "from prior",
    "interval change",
    "in the interval",
    "stable from prior",
    "new since",
    "previously",
    "prior study",
    "prior examination",
    "prior radiograph",
    "unchanged",
    "improved",
    "worsened",
    "increased since",
    "decreased since",
    "resolution of",
    "no interval",
)

#: Words in the *input context* that would make a comparison legitimate.
PRIOR_CONTEXT_CUES = ("prior", "previous", "comparison", "earlier", "baseline")

LATERALITY_TERMS = ("left", "right", "bilateral")

# Error category names. All are screening labels.
POSSIBLE_FALSE_POSITIVE = "possible_false_positive_finding"
POSSIBLE_OMISSION = "possible_omitted_finding"
POSSIBLE_LATERALITY_ERROR = "possible_laterality_error"
POSSIBLE_SEVERITY_ERROR = "possible_severity_error"
POSSIBLE_NEGATION_ERROR = "possible_negation_error"
POSSIBLE_TEMPORAL_HALLUCINATION = "possible_temporal_hallucination"
POSSIBLE_DEVICE_LOCATION_ERROR = "possible_device_location_error"
EMPTY_OUTPUT = "empty_output"
EXCESSIVE_REPETITION = "excessive_repetition"
REPORT_TOO_SHORT = "report_too_short"
REPORT_TOO_LONG = "report_too_long"

ERROR_CATEGORIES = (
    POSSIBLE_FALSE_POSITIVE,
    POSSIBLE_OMISSION,
    POSSIBLE_LATERALITY_ERROR,
    POSSIBLE_SEVERITY_ERROR,
    POSSIBLE_NEGATION_ERROR,
    POSSIBLE_TEMPORAL_HALLUCINATION,
    POSSIBLE_DEVICE_LOCATION_ERROR,
    EMPTY_OUTPUT,
    EXCESSIVE_REPETITION,
    REPORT_TOO_SHORT,
    REPORT_TOO_LONG,
)

DEVICE_TERMS = (
    "endotracheal tube",
    "et tube",
    "central line",
    "picc",
    "nasogastric tube",
    "ng tube",
    "chest tube",
    "pacemaker",
    "catheter",
    "sternotomy wire",
)

DEFAULT_MIN_TOKENS = 5
DEFAULT_MAX_TOKENS = 400
DEFAULT_REPETITION_RATIO = 0.4


def _mentions(text: str) -> dict[str, str]:
    """Map finding name -> polarity, using the shared lexicon.

    Polarity is ``negative`` when a negation cue precedes the mention inside the
    same sentence, otherwise ``positive``. This mirrors ``safety/claims.py``:
    "no pneumothorax" is a *negative claim about pneumothorax*, never the
    absence of a mention.
    """
    found: dict[str, str] = {}
    for sentence in split_sentences(text):
        lowered = normalize(sentence)
        for finding, synonyms in ABNORMALITY_SYNONYMS.items():
            for synonym in synonyms:
                position = lowered.find(synonym)
                if position < 0:
                    continue
                prefix = lowered[:position]
                negated = any(cue in prefix for cue in NEGATION_CUES)
                polarity = "negative" if negated else "positive"
                # A positive mention anywhere wins over a negative one: a report
                # that both denies and asserts a finding is asserting it.
                if found.get(finding) != "positive":
                    found[finding] = polarity
                break
    return found


def detect_temporal_hallucination(
    generated: str, *, context: str | None = None
) -> tuple[bool, list[str]]:
    """Flag comparative statements unsupported by the input.

    Returns ``(flagged, matched_phrases)``. When ``context`` mentions a prior
    study the flag is suppressed, because the comparison may then be grounded.
    """
    lowered = normalize(generated)
    matched = [cue for cue in TEMPORAL_CUES if cue in lowered]
    if not matched:
        return False, []

    if context:
        lowered_context = normalize(context)
        if any(cue in lowered_context for cue in PRIOR_CONTEXT_CUES):
            return False, matched

    return True, matched


def repetition_ratio(text: str) -> float:
    """Fraction of sentences that are exact duplicates of an earlier sentence."""
    sentences = [normalize(s) for s in split_sentences(text) if s.strip()]
    if len(sentences) <= 1:
        return 0.0
    counts = Counter(sentences)
    duplicates = sum(count - 1 for count in counts.values() if count > 1)
    return duplicates / len(sentences)


def _laterality(text: str) -> set[str]:
    tokens = set(tokenize(text))
    return {term for term in LATERALITY_TERMS if term in tokens}


def _severity(text: str) -> set[str]:
    lowered = normalize(text)
    return {term for term in SEVERITY_TERMS if term in lowered}


def _devices(text: str) -> set[str]:
    lowered = normalize(text)
    return {term for term in DEVICE_TERMS if term in lowered}


@dataclass
class SampleErrorReport:
    """Per-sample analysis of one generated report."""

    sample_key: str
    generated: str
    reference: str
    generated_length: int
    reference_length: int
    empty: bool
    repetition_ratio: float
    predicted_findings: dict[str, str]
    reference_findings: dict[str, str]
    false_positive_findings: list[str]
    false_negative_findings: list[str]
    negation_mismatches: list[str]
    temporal_phrases: list[str]
    flags: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self, *, include_text: bool = True) -> dict[str, Any]:
        record: dict[str, Any] = {
            "sample_key": self.sample_key,
            "generated_length": self.generated_length,
            "reference_length": self.reference_length,
            "empty_report": self.empty,
            "repetition_ratio": round(self.repetition_ratio, 4),
            "predicted_findings": self.predicted_findings,
            "reference_findings": self.reference_findings,
            "false_positive_findings": self.false_positive_findings,
            "false_negative_findings": self.false_negative_findings,
            "negation_mismatches": self.negation_mismatches,
            "temporal_phrases": self.temporal_phrases,
            "flags": self.flags,
        }
        record.update({k: round(v, 6) for k, v in self.scores.items()})
        if include_text:
            record["generated"] = self.generated
            record["reference"] = self.reference
        return record


def analyse_sample(
    sample_key: str,
    generated: str,
    reference: str,
    *,
    scores: dict[str, float] | None = None,
    context: str | None = None,
    min_tokens: int = DEFAULT_MIN_TOKENS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    repetition_threshold: float = DEFAULT_REPETITION_RATIO,
) -> SampleErrorReport:
    """Analyse one generated report against its reference."""
    generated_tokens = tokenize(generated)
    reference_tokens = tokenize(reference)

    predicted_findings = _mentions(generated)
    reference_findings = _mentions(reference)

    predicted_positive = {k for k, v in predicted_findings.items() if v == "positive"}
    reference_positive = {k for k, v in reference_findings.items() if v == "positive"}

    false_positive = sorted(predicted_positive - reference_positive)
    false_negative = sorted(reference_positive - predicted_positive)

    # A finding both sides mention, but with opposite polarity.
    negation_mismatches = sorted(
        finding
        for finding in set(predicted_findings) & set(reference_findings)
        if predicted_findings[finding] != reference_findings[finding]
    )

    flagged_temporal, temporal_phrases = detect_temporal_hallucination(
        generated, context=context
    )
    ratio = repetition_ratio(generated)

    flags: list[str] = []
    if not generated_tokens:
        flags.append(EMPTY_OUTPUT)
    if false_positive:
        flags.append(POSSIBLE_FALSE_POSITIVE)
    if false_negative:
        flags.append(POSSIBLE_OMISSION)
    if negation_mismatches:
        flags.append(POSSIBLE_NEGATION_ERROR)
    if flagged_temporal:
        flags.append(POSSIBLE_TEMPORAL_HALLUCINATION)
    if ratio >= repetition_threshold:
        flags.append(EXCESSIVE_REPETITION)
    if generated_tokens and len(generated_tokens) < min_tokens:
        flags.append(REPORT_TOO_SHORT)
    if len(generated_tokens) > max_tokens:
        flags.append(REPORT_TOO_LONG)

    if _laterality(generated) and _laterality(reference):
        if _laterality(generated) != _laterality(reference):
            flags.append(POSSIBLE_LATERALITY_ERROR)
    if _severity(generated) and _severity(reference):
        if _severity(generated) != _severity(reference):
            flags.append(POSSIBLE_SEVERITY_ERROR)
    generated_devices = _devices(generated)
    if generated_devices and generated_devices != _devices(reference):
        flags.append(POSSIBLE_DEVICE_LOCATION_ERROR)

    return SampleErrorReport(
        sample_key=sample_key,
        generated=generated,
        reference=reference,
        generated_length=len(generated_tokens),
        reference_length=len(reference_tokens),
        empty=not generated_tokens,
        repetition_ratio=ratio,
        predicted_findings=predicted_findings,
        reference_findings=reference_findings,
        false_positive_findings=false_positive,
        false_negative_findings=false_negative,
        negation_mismatches=negation_mismatches,
        temporal_phrases=temporal_phrases,
        flags=flags,
        scores=scores or {},
    )


def summarise_errors(reports: list[SampleErrorReport]) -> dict[str, Any]:
    """Aggregate flag rates and length statistics over a split."""
    if not reports:
        raise ValueError("no sample reports to summarise")

    total = len(reports)
    counts = Counter()
    for report in reports:
        counts.update(report.flags)

    lengths = [r.generated_length for r in reports]
    reference_lengths = [r.reference_length for r in reports]

    return {
        "num_samples": total,
        "flag_counts": {category: counts.get(category, 0) for category in ERROR_CATEGORIES},
        "flag_rates": {
            category: counts.get(category, 0) / total for category in ERROR_CATEGORIES
        },
        "empty_output_rate": counts.get(EMPTY_OUTPUT, 0) / total,
        "repetition_rate": counts.get(EXCESSIVE_REPETITION, 0) / total,
        "possible_temporal_hallucination_rate": (
            counts.get(POSSIBLE_TEMPORAL_HALLUCINATION, 0) / total
        ),
        "generated_length": _length_stats(lengths),
        "reference_length": _length_stats(reference_lengths),
        "caveat": (
            "All possible_* rates are lexicon heuristics over surface text, not "
            "radiologist-confirmed clinical error rates."
        ),
    }


def _length_stats(values: list[int]) -> dict[str, float]:
    ordered = sorted(values)
    count = len(ordered)
    return {
        "mean": sum(ordered) / count,
        "min": float(ordered[0]),
        "p25": float(ordered[max(0, count // 4 - 1)]),
        "median": float(ordered[count // 2]),
        "p75": float(ordered[min(count - 1, (3 * count) // 4)]),
        "max": float(ordered[-1]),
    }
