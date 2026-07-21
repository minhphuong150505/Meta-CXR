"""Orchestration: draft report -> verified claims -> final report or abstention.

This module only wires components together. It contains no verification logic
of its own, so a real phrase-grounding model or a trained claim parser can be
substituted by passing a different implementation of the same protocol.

The output record deliberately carries no ``subject_id``, ``study_id``,
``dicom_id``, absolute path or reference report. It is safe to persist next to
the other Stage-2 evaluation artifacts.

stdlib only.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from safety.claims import (
    POSITIVE,
    SUPPORTED,
    UNSUPPORTED,
    Claim,
    ClaimParser,
    LexiconClaimParser,
    unparsed_sentences,
)
from safety.reconciler import ReconciliationOutcome, RuleBasedClaimReconciler
from safety.verifiers import (
    AbnormalityVerifier,
    ClassifierAbnormalityVerifier,
    EntropyUncertaintyEstimator,
    MeasurementChecker,
    PhraseGroundingVerifier,
    RegexMeasurementChecker,
    UnavailablePhraseGrounding,
    UncertaintyEstimator,
    VerificationResult,
)


@dataclass
class SafetyReport:
    """Result of running the safety pipeline over one draft."""

    draft: str
    final_report: str
    outcome: ReconciliationOutcome
    unparsed: list[str] = field(default_factory=list)
    verifier_availability: dict[str, bool] = field(default_factory=dict)

    @property
    def parse_coverage(self) -> float:
        """Fraction of sentences that produced at least one claim.

        Surfaced because a pipeline that parsed two of twelve sentences has not
        checked the report, however clean its claim-level numbers look.
        """
        total = len(self.outcome.claims) + len(self.unparsed)
        if total == 0:
            return 0.0
        return len(self.outcome.claims) / total

    def to_dict(self) -> dict[str, Any]:
        return {
            "report": self.final_report,
            "claims": [claim.to_dict() for claim in self.outcome.claims],
            "edits": [edit.to_dict() for edit in self.outcome.edits],
            "unverified_sentences": len(self.unparsed),
            "parse_coverage": round(self.parse_coverage, 4),
            "verifier_availability": self.verifier_availability,
            "safety": {
                "abstained": self.outcome.abstained,
                "requires_review": self.outcome.requires_review,
                "reasons": self.outcome.reasons,
            },
        }


ABSTENTION_TEXT = (
    "This study was not reported automatically: the draft could not be "
    "sufficiently supported by independent verification. Radiologist review required."
)


class SafetyPipeline:
    """draft -> parse -> verify -> ground -> measure -> uncertainty -> reconcile."""

    def __init__(
        self,
        *,
        parser: ClaimParser | None = None,
        grounding: PhraseGroundingVerifier | None = None,
        reconciler: RuleBasedClaimReconciler | None = None,
    ):
        self.parser = parser or LexiconClaimParser()
        self.grounding = grounding or UnavailablePhraseGrounding()
        self.reconciler = reconciler or RuleBasedClaimReconciler()

    def run(
        self,
        draft: str,
        *,
        classifier_probabilities: Mapping[str, Mapping[str, float]] | None = None,
        measured_mm: tuple[float, ...] = (),
        abnormality_verifier: AbnormalityVerifier | None = None,
        measurement_checker: MeasurementChecker | None = None,
        uncertainty_estimator: UncertaintyEstimator | None = None,
    ) -> SafetyReport:
        probabilities = classifier_probabilities or {}
        verifier = abnormality_verifier or ClassifierAbnormalityVerifier(probabilities)
        checker = measurement_checker or RegexMeasurementChecker(measured_mm)
        estimator = uncertainty_estimator or EntropyUncertaintyEstimator(probabilities)

        claims = self.parser.parse(draft)
        unparsed = unparsed_sentences(draft, claims)

        classifier_results: dict[int, VerificationResult] = {}
        grounding_results: dict[int, VerificationResult] = {}
        measurement_results: dict[int, VerificationResult] = {}

        for index, claim in enumerate(claims):
            classifier_results[index] = verifier.verify(claim)
            grounding_result, evidence = self.grounding.verify(claim)
            grounding_results[index] = grounding_result
            claim.evidence = evidence
            measurement_results[index] = checker.check(claim)
            claim.uncertainty = estimator.estimate(claim)

        outcome = self.reconciler.reconcile(
            claims, classifier_results, grounding_results, measurement_results
        )

        availability = {
            "abnormality_verifier": any(r.ran for r in classifier_results.values()),
            "phrase_grounding": any(r.ran for r in grounding_results.values()),
            "measurement_checker": any(r.ran for r in measurement_results.values()),
        }

        final = ABSTENTION_TEXT if outcome.abstained else self._render(outcome)
        return SafetyReport(
            draft=draft,
            final_report=final,
            outcome=outcome,
            unparsed=unparsed,
            verifier_availability=availability,
        )

    @staticmethod
    def _render(outcome: ReconciliationOutcome) -> str:
        """Rebuild the report from the reconciled claim texts.

        Sentences that produced no claim are intentionally dropped rather than
        passed through unchecked: anything in the final report has been through
        the pipeline.
        """
        seen: set[tuple[int, str]] = set()
        parts: list[str] = []
        for claim in sorted(outcome.claims, key=lambda c: c.sentence_index):
            key = (claim.sentence_index, claim.text)
            if key in seen:
                continue
            seen.add(key)
            parts.append(claim.text.strip())
        return " ".join(parts)


def hallucination_rate(claims: list[Claim]) -> float:
    """Asserted findings the evidence does not support, over all asserted findings."""
    asserted = [c for c in claims if c.polarity == POSITIVE or c.status == UNSUPPORTED]
    if not asserted:
        return 0.0
    return len([c for c in asserted if c.status == UNSUPPORTED]) / len(asserted)


def supported_rate(claims: list[Claim]) -> float:
    if not claims:
        return 0.0
    return len([c for c in claims if c.status == SUPPORTED]) / len(claims)
