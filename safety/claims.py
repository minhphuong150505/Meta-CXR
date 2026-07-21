"""Claim schema and the baseline parser that turns a draft report into claims.

A generated report is scored as a whole by BLEU/BERTScore, which cannot say
*which* statement is wrong. The safety pipeline works at claim level instead:
one sentence about one finding, with a polarity, so it can be checked against
the classifier, against grounding evidence, and against measurement rules.

Design constraints that are easy to get wrong and are enforced here:

* **Negation is detected, never deleted.** "No pneumothorax" is a *negative*
  claim about pneumothorax, not the absence of a claim, and certainly not a
  positive one. Stripping negation cues -- which some report-processing code
  does to "clean" text -- inverts the clinical meaning.
* **Hedging is its own polarity.** "cannot exclude pneumonia" is neither
  positive nor negative; collapsing it into either direction is a factual error.
* **An unmatched sentence produces no claim rather than a guess.** Silence is
  reported downstream as coverage, so it cannot be mistaken for verification.

stdlib only: no torch, no transformers, no network.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

POSITIVE = "positive"
NEGATIVE = "negative"
UNCERTAIN = "uncertain"
POLARITIES = (POSITIVE, NEGATIVE, UNCERTAIN)

SUPPORTED = "supported"
UNSUPPORTED = "unsupported"
STATUS_UNCERTAIN = "uncertain"
CLAIM_STATUSES = (SUPPORTED, UNSUPPORTED, STATUS_UNCERTAIN)

EVIDENCE_HEATMAP = "heatmap"
EVIDENCE_BOX = "box"
EVIDENCE_MASK = "mask"
EVIDENCE_NONE = "none"
EVIDENCE_TYPES = (EVIDENCE_HEATMAP, EVIDENCE_BOX, EVIDENCE_MASK, EVIDENCE_NONE)

# The 14 CheXpert labels the Stage-1 classifier predicts, with the surface forms
# radiologists actually write. Keys must match the classifier's label names.
ABNORMALITY_SYNONYMS: dict[str, tuple[str, ...]] = {
    "No Finding": ("no finding", "no acute cardiopulmonary abnormality", "unremarkable"),
    "Enlarged Cardiomediastinum": ("enlarged cardiomediastinum", "mediastinal widening", "widened mediastinum"),
    "Cardiomegaly": ("cardiomegaly", "enlarged heart", "cardiac enlargement", "enlarged cardiac silhouette"),
    "Lung Opacity": ("lung opacity", "opacity", "opacities", "airspace disease", "airspace opacity"),
    "Lung Lesion": ("lung lesion", "nodule", "nodules", "mass", "masses"),
    "Edema": ("edema", "pulmonary edema", "vascular congestion", "interstitial edema"),
    "Consolidation": ("consolidation", "consolidative"),
    "Pneumonia": ("pneumonia", "infection", "infectious process"),
    "Atelectasis": ("atelectasis", "atelectatic", "volume loss"),
    "Pneumothorax": ("pneumothorax", "pneumothoraces"),
    "Pleural Effusion": ("pleural effusion", "effusion", "effusions"),
    "Pleural Other": ("pleural thickening", "pleural scarring", "pleural plaque"),
    "Fracture": ("fracture", "fractures", "fractured"),
    "Support Devices": (
        "endotracheal tube", "et tube", "central line", "picc", "pacemaker",
        "chest tube", "nasogastric tube", "ng tube", "catheter", "sternotomy wires",
    ),
}

# Cues that make a mention negative. Matched as whole phrases before the finding.
NEGATION_CUES = (
    "no evidence of", "no evidence for", "without evidence of", "no radiographic evidence of",
    "no significant", "no acute", "no focal", "no new", "no residual",
    "there is no", "there are no", "no ", "not ", "without ", "free of",
    "resolved", "clear of", "negative for", "absence of", "ruled out",
)

# Cues that make a mention hedged. Checked before negation, because
# "cannot exclude" contains "not" and must not be read as a denial.
UNCERTAINTY_CUES = (
    "cannot exclude", "can not exclude", "cannot be excluded", "not excluded",
    "possible", "possibly", "probable", "probably", "may represent", "might represent",
    "could represent", "suspicious for", "suggestive of", "concerning for",
    "questionable", "equivocal", "indeterminate", "versus", "differential",
    "cannot be ruled out", "difficult to exclude", "borderline",
)

# Descriptors kept as structured fields so the reconciler can revise severity
# wording without rewriting the sentence.
SEVERITY_TERMS = (
    "trace", "minimal", "mild", "small", "slight", "moderate", "large",
    "severe", "extensive", "massive", "marked", "significant",
)
LOCATION_TERMS = (
    "right upper", "right middle", "right lower", "left upper", "left lower",
    "bilateral", "biapical", "right basilar", "left basilar", "bibasilar",
    "right apical", "left apical", "retrocardiac", "perihilar", "right", "left",
    "upper", "lower", "basilar", "apical", "lingular",
)

# A measurement is any number with a length unit. Used by MeasurementChecker.
MEASUREMENT_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s?(?:mm|cm|millimeters?|centimeters?)\b", re.IGNORECASE
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.;])\s+")


@dataclass(frozen=True)
class Evidence:
    """Where the visual support for a claim came from, if anywhere."""

    type: str = EVIDENCE_NONE
    coordinates: tuple = ()

    def __post_init__(self) -> None:
        if self.type not in EVIDENCE_TYPES:
            raise ValueError(f"evidence type must be one of {EVIDENCE_TYPES}, got {self.type!r}")

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "coordinates": list(self.coordinates)}


@dataclass
class Claim:
    """One checkable statement extracted from a draft report.

    Scores are ``None`` until a verifier supplies them. ``None`` means "not
    assessed" and is deliberately distinct from ``0.0``, which would mean
    "assessed and found absent". The reconciler relies on that distinction.
    """

    text: str
    finding: str
    polarity: str
    location: str | None = None
    severity: str | None = None
    generator_score: float | None = None
    classifier_score: float | None = None
    grounding_score: float | None = None
    uncertainty: float | None = None
    status: str = STATUS_UNCERTAIN
    evidence: Evidence = field(default_factory=Evidence)
    sentence_index: int = 0

    def __post_init__(self) -> None:
        if self.polarity not in POLARITIES:
            raise ValueError(f"polarity must be one of {POLARITIES}, got {self.polarity!r}")
        if self.status not in CLAIM_STATUSES:
            raise ValueError(f"status must be one of {CLAIM_STATUSES}, got {self.status!r}")

    @property
    def has_measurement(self) -> bool:
        return bool(MEASUREMENT_PATTERN.search(self.text))

    def to_dict(self) -> dict[str, Any]:
        """Serialise to the audit schema. Carries no patient identifiers."""
        return {
            "text": self.text,
            "finding": self.finding,
            "polarity": self.polarity,
            "location": self.location,
            "severity": self.severity,
            "generator_score": self.generator_score,
            "classifier_score": self.classifier_score,
            "grounding_score": self.grounding_score,
            "uncertainty": self.uncertainty,
            "status": self.status,
            "evidence": self.evidence.to_dict(),
        }


class ClaimParser(Protocol):
    """Turns a draft report into checkable claims."""

    def parse(self, report: str) -> list[Claim]:
        ...


def split_sentences(text: str) -> list[str]:
    """Split on sentence terminators, keeping non-empty fragments."""
    return [part.strip() for part in _SENTENCE_SPLIT.split(str(text or "")) if part.strip()]


def _first_term(haystack: str, terms: Sequence[str]) -> str | None:
    """Longest matching term, so 'right lower' beats 'right'."""
    best = None
    for term in terms:
        if re.search(rf"\b{re.escape(term)}\b", haystack):
            if best is None or len(term) > len(best):
                best = term
    return best


def detect_polarity(sentence: str, mention_start: int) -> str:
    """Classify the polarity of one finding mention within its sentence.

    Only the text *before* the mention is considered, so "effusion resolved, no
    pneumothorax" does not mark the effusion negative on account of a cue that
    belongs to the later clause. Hedging is checked first because several
    uncertainty cues contain a negation token.
    """
    lowered = sentence.lower()
    # Clause boundary: a cue in an earlier clause does not reach this mention.
    clause_start = 0
    for separator in (",", ";", " but ", " however ", " although ", " while "):
        position = lowered.rfind(separator, 0, mention_start)
        if position != -1:
            clause_start = max(clause_start, position + len(separator))
    prefix = lowered[clause_start:mention_start]

    for cue in UNCERTAINTY_CUES:
        if cue in prefix:
            return UNCERTAIN
    # A trailing hedge ("... pneumonia cannot be excluded") still hedges.
    tail = lowered[mention_start:]
    for cue in ("cannot be excluded", "cannot be ruled out", "is possible", "not excluded"):
        if cue in tail:
            return UNCERTAIN
    for cue in NEGATION_CUES:
        if cue in prefix:
            return NEGATIVE
    return POSITIVE


class LexiconClaimParser:
    """Baseline claim parser: sentence split, synonym match, polarity detection.

    This is a real, deterministic implementation, not a placeholder. Its limits
    are known and stated rather than hidden:

    * It recognises only the 14 CheXpert findings in ``ABNORMALITY_SYNONYMS``.
      A sentence about anything else yields no claim and is reported as
      unparsed coverage, never as a verified statement.
    * Polarity is cue-based within a clause. Nested constructions
      ("no evidence to suggest that the opacity is not infectious") are not
      resolved; such sentences are marked uncertain by the hedge cues they
      contain rather than being guessed at.

    Replacing it with a trained clinical NLI model requires only implementing
    the ``ClaimParser`` protocol.
    """

    def __init__(self, synonyms: dict[str, tuple[str, ...]] | None = None):
        self.synonyms = synonyms or ABNORMALITY_SYNONYMS

    def parse(self, report: str) -> list[Claim]:
        claims: list[Claim] = []
        for index, sentence in enumerate(split_sentences(report)):
            lowered = sentence.lower()
            for finding, terms in self.synonyms.items():
                match = self._match(lowered, terms)
                if match is None:
                    continue
                claims.append(
                    Claim(
                        text=sentence,
                        finding=finding,
                        polarity=detect_polarity(sentence, match),
                        location=_first_term(lowered, LOCATION_TERMS),
                        severity=_first_term(lowered, SEVERITY_TERMS),
                        sentence_index=index,
                    )
                )
        return claims

    @staticmethod
    def _match(lowered: str, terms: Sequence[str]) -> int | None:
        """Start offset of the earliest whole-word synonym hit, if any."""
        earliest = None
        for term in terms:
            found = re.search(rf"\b{re.escape(term)}\b", lowered)
            if found and (earliest is None or found.start() < earliest):
                earliest = found.start()
        return earliest


def unparsed_sentences(report: str, claims: Sequence[Claim]) -> list[str]:
    """Sentences that produced no claim.

    Reported so a low parse rate is visible: a pipeline that verifies two of a
    report's twelve sentences has not verified the report.
    """
    covered = {claim.sentence_index for claim in claims}
    return [
        sentence
        for index, sentence in enumerate(split_sentences(report))
        if index not in covered
    ]
