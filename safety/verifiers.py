"""Independent checks on a draft report's claims.

Every verifier here returns a result that distinguishes three outcomes:

* **supported / contradicted** -- the verifier ran and reached a conclusion;
* **inconclusive** -- the verifier ran and could not decide;
* **unavailable** -- the verifier could not run at all (no model, no input).

Collapsing "unavailable" into "contradicted" is the failure this module is built
to prevent. If a missing grounding model silently read as "not grounded", the
reconciler would strip correct positive findings from every report and the
pipeline would look like it was working.

stdlib only.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from safety.claims import (
    MEASUREMENT_PATTERN,
    NEGATIVE,
    POSITIVE,
    UNCERTAIN,
    Claim,
    Evidence,
)

SUPPORTED = "supported"
CONTRADICTED = "contradicted"
INCONCLUSIVE = "inconclusive"
UNAVAILABLE = "unavailable"
VERDICTS = (SUPPORTED, CONTRADICTED, INCONCLUSIVE, UNAVAILABLE)


@dataclass(frozen=True)
class VerificationResult:
    """One verifier's opinion about one claim."""

    verdict: str
    score: float | None = None
    detail: str = ""

    def __post_init__(self) -> None:
        if self.verdict not in VERDICTS:
            raise ValueError(f"verdict must be one of {VERDICTS}, got {self.verdict!r}")

    @property
    def ran(self) -> bool:
        """False when the verifier could not run, so callers cannot treat
        absence of evidence as evidence of absence."""
        return self.verdict != UNAVAILABLE


class AbnormalityVerifier(Protocol):
    def verify(self, claim: Claim) -> VerificationResult:
        ...


class PhraseGroundingVerifier(Protocol):
    def verify(self, claim: Claim) -> tuple[VerificationResult, Evidence]:
        ...


class MeasurementChecker(Protocol):
    def check(self, claim: Claim) -> VerificationResult:
        ...


class ClassifierAbnormalityVerifier:
    """Checks a claim's polarity against the Stage-1 classifier's probabilities.

    ``probabilities`` maps an abnormality name to its per-class probabilities,
    ``{"negative": p, "positive": p, "uncertain": p}`` -- exactly the MHCAC
    output. The classifier is an *independent* opinion: it sees the image and
    never sees the draft text, which is what makes the comparison meaningful.

    A finding the classifier does not cover returns ``unavailable``, not
    ``contradicted``.
    """

    def __init__(
        self,
        probabilities: Mapping[str, Mapping[str, float]],
        *,
        positive_threshold: float = 0.5,
        negative_threshold: float = 0.5,
    ):
        self.probabilities = probabilities
        self.positive_threshold = float(positive_threshold)
        self.negative_threshold = float(negative_threshold)

    def verify(self, claim: Claim) -> VerificationResult:
        row = self.probabilities.get(claim.finding)
        if not row:
            return VerificationResult(
                UNAVAILABLE, None, f"classifier has no output for {claim.finding!r}"
            )
        positive = float(row.get(POSITIVE, 0.0))
        negative = float(row.get(NEGATIVE, 0.0))

        if claim.polarity == POSITIVE:
            if positive >= self.positive_threshold:
                return VerificationResult(SUPPORTED, positive, "classifier agrees positive")
            if negative >= self.negative_threshold:
                return VerificationResult(
                    CONTRADICTED, positive, "classifier calls this negative"
                )
            return VerificationResult(INCONCLUSIVE, positive, "classifier below both thresholds")

        if claim.polarity == NEGATIVE:
            if negative >= self.negative_threshold:
                return VerificationResult(SUPPORTED, positive, "classifier agrees negative")
            if positive >= self.positive_threshold:
                # Rule 2 territory: the draft denies something the image supports.
                return VerificationResult(
                    CONTRADICTED, positive, "classifier calls this positive"
                )
            return VerificationResult(INCONCLUSIVE, positive, "classifier below both thresholds")

        # A hedged claim is compatible with either classifier outcome; it can be
        # informed by the probability but not confirmed or refuted by it.
        return VerificationResult(INCONCLUSIVE, positive, "claim is hedged")


class UnavailablePhraseGrounding:
    """The honest default: no phrase-grounding model is integrated.

    This repo contains no phrase-grounding or region-proposal model, and no
    box/mask annotations. Rather than emit fabricated heatmap scores, every
    claim returns ``unavailable`` with ``EVIDENCE_NONE``.

    Missing dependency, stated explicitly: a grounding model
    (e.g. a region-text alignment head trained on MS-CXR bounding boxes) plus
    per-claim box annotations for evaluation. Until one is wired in, the
    reconciler will not apply any rule whose premise requires grounding.
    """

    reason = "no phrase-grounding model is integrated in this repository"

    def verify(self, claim: Claim) -> tuple[VerificationResult, Evidence]:
        return VerificationResult(UNAVAILABLE, None, self.reason), Evidence()


class RegexMeasurementChecker:
    """Confirms measurements in a claim against measurements the pipeline can source.

    A generated "3.2 cm nodule" is unverifiable unless something measured it.
    This checker compares each measurement in the claim text against a supplied
    set of measured values (from a segmentation/measurement tool, or a prior
    report when explicitly permitted). With no measured values available, a
    claim containing a measurement is ``inconclusive`` -- it ran, and it found
    no support -- while a claim containing no measurement is ``supported``,
    because there is nothing to substantiate.

    Missing dependency: an automated measurement source. None is integrated, so
    in the default configuration every measured claim is flagged for review
    rather than silently accepted.
    """

    def __init__(
        self,
        measured_mm: Sequence[float] = (),
        *,
        tolerance_mm: float = 2.0,
    ):
        self.measured_mm = [float(value) for value in measured_mm]
        self.tolerance_mm = float(tolerance_mm)

    @staticmethod
    def to_millimetres(raw: str) -> float | None:
        match = re.match(r"(\d+(?:\.\d+)?)\s?(mm|cm|millimeters?|centimeters?)", raw.strip(), re.I)
        if not match:
            return None
        value = float(match.group(1))
        unit = match.group(2).lower()
        return value * 10.0 if unit.startswith("c") else value

    def check(self, claim: Claim) -> VerificationResult:
        found = MEASUREMENT_PATTERN.findall(claim.text)
        # findall with a non-capturing group returns whole matches.
        raw_values = [m if isinstance(m, str) else m[0] for m in found]
        if not raw_values:
            return VerificationResult(SUPPORTED, None, "claim states no measurement")
        if not self.measured_mm:
            return VerificationResult(
                INCONCLUSIVE, None, "no measurement source available to confirm"
            )
        unconfirmed = []
        for raw in raw_values:
            millimetres = self.to_millimetres(raw)
            if millimetres is None:
                unconfirmed.append(raw)
                continue
            if not any(
                abs(millimetres - measured) <= self.tolerance_mm
                for measured in self.measured_mm
            ):
                unconfirmed.append(raw)
        if unconfirmed:
            return VerificationResult(
                CONTRADICTED, None, f"unconfirmed measurement(s): {', '.join(unconfirmed)}"
            )
        return VerificationResult(SUPPORTED, None, "all measurements confirmed")


class UncertaintyEstimator(Protocol):
    def estimate(self, claim: Claim) -> float | None:
        ...


class EntropyUncertaintyEstimator:
    """Normalised Shannon entropy of the classifier distribution for the finding.

    Returns a value in [0, 1]: 0 when the classifier is certain, 1 when it is
    uniform across the three classes. Returns ``None`` -- not 0.0 -- when the
    classifier has no output for the finding, so "unknown" cannot be read as
    "confident".

    A hedged claim carries an explicit floor: a draft that says "cannot exclude"
    is uncertain by construction regardless of what the classifier thinks.
    """

    def __init__(
        self,
        probabilities: Mapping[str, Mapping[str, float]],
        *,
        hedged_floor: float = 0.5,
    ):
        self.probabilities = probabilities
        self.hedged_floor = float(hedged_floor)

    def estimate(self, claim: Claim) -> float | None:
        row = self.probabilities.get(claim.finding)
        if not row:
            return self.hedged_floor if claim.polarity == UNCERTAIN else None
        values = [max(float(v), 0.0) for v in row.values()]
        total = sum(values)
        if total <= 0:
            return None
        distribution = [v / total for v in values]
        entropy = -sum(p * math.log(p) for p in distribution if p > 0)
        normalised = entropy / math.log(len(distribution)) if len(distribution) > 1 else 0.0
        normalised = min(max(normalised, 0.0), 1.0)
        if claim.polarity == UNCERTAIN:
            return max(normalised, self.hedged_floor)
        return normalised
