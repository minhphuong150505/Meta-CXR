"""Rule-based reconciliation of a draft report against independent evidence.

The reconciler is the only component permitted to change report text. Every
change it makes is recorded as a ``ClaimEdit`` carrying the rule that fired and
the premises that were actually evaluated, so a reviewer can tell the difference
between "the classifier contradicted this" and "the classifier was not consulted".

The rules implement the required policy:

1. draft-positive + classifier-negative (+ grounding unsupported, when grounding
   is available) -> the positive assertion is not kept as-is.
2. draft-negative + classifier-positive (+ grounding supported) -> flagged as a
   possible omission; the draft text is not silently rewritten into a positive.
3. a measurement the measurement checker cannot confirm -> the number is removed
   and the claim is sent for review.
4. uncertainty above threshold -> cautious language, or abstention.
5. insufficient evidence -> an explicit "insufficient evidence" statement rather
   than a confident one.
6. every edit carries an audit trail.

stdlib only.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from safety.claims import (
    MEASUREMENT_PATTERN,
    NEGATIVE,
    POSITIVE,
    STATUS_UNCERTAIN,
    SUPPORTED,
    UNCERTAIN,
    UNSUPPORTED,
    Claim,
)
from safety.verifiers import (
    CONTRADICTED,
    UNAVAILABLE,
    VerificationResult,
)

RULE_POSITIVE_CONTRADICTED = "positive_contradicted_by_classifier"
RULE_POSSIBLE_OMISSION = "possible_omission"
RULE_UNCONFIRMED_MEASUREMENT = "unconfirmed_measurement"
RULE_HIGH_UNCERTAINTY = "high_uncertainty"
RULE_INSUFFICIENT_EVIDENCE = "insufficient_evidence"

INSUFFICIENT_EVIDENCE_TEXT = "Insufficient evidence to confirm this finding."


@dataclass
class ClaimEdit:
    """One recorded change, with the premises that justified it."""

    rule: str
    finding: str
    before: str
    after: str
    premises: dict[str, Any] = field(default_factory=dict)
    requires_review: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule": self.rule,
            "finding": self.finding,
            "before": self.before,
            "after": self.after,
            "premises": self.premises,
            "requires_review": self.requires_review,
        }


@dataclass
class ReconciliationOutcome:
    claims: list[Claim]
    edits: list[ClaimEdit]
    abstained: bool
    requires_review: bool
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "claims": [claim.to_dict() for claim in self.claims],
            "edits": [edit.to_dict() for edit in self.edits],
            "abstained": self.abstained,
            "requires_review": self.requires_review,
            "reasons": self.reasons,
        }


def hedge(sentence: str) -> str:
    """Downgrade an assertion to cautious language without deleting content.

    The original wording is preserved inside the hedge so a reviewer can still
    see what the model actually claimed.
    """
    text = sentence.strip().rstrip(".")
    if not text:
        return sentence
    # Appended rather than prefixed: prefixing "Possible" onto an arbitrary
    # sentence ("Possible there is a large effusion") is ungrammatical, and a
    # malformed report is its own safety problem.
    return f"{text} (not confirmed by independent verification)."


def strip_measurements(sentence: str) -> str:
    """Remove numeric measurements, leaving the qualitative claim intact."""
    stripped = MEASUREMENT_PATTERN.sub("unmeasured", sentence)
    return re.sub(r"\s{2,}", " ", stripped).strip()


class RuleBasedClaimReconciler:
    """Applies the reconciliation policy to verified claims.

    ``require_grounding`` controls whether rules 1 and 2 need grounding evidence
    in addition to the classifier verdict. It defaults to **False** because this
    repository integrates no phrase-grounding model: requiring grounding would
    make both rules unreachable and the safety layer a no-op. When a grounding
    model is wired in, set it to True to enforce the full three-premise policy.
    Either way the audit trail records which premises were evaluated.
    """

    def __init__(
        self,
        *,
        uncertainty_threshold: float = 0.6,
        abstain_unsupported_ratio: float = 0.5,
        require_grounding: bool = False,
    ):
        self.uncertainty_threshold = float(uncertainty_threshold)
        self.abstain_unsupported_ratio = float(abstain_unsupported_ratio)
        self.require_grounding = bool(require_grounding)

    def reconcile(
        self,
        claims: list[Claim],
        classifier: Mapping[int, VerificationResult],
        grounding: Mapping[int, VerificationResult],
        measurement: Mapping[int, VerificationResult],
    ) -> ReconciliationOutcome:
        edits: list[ClaimEdit] = []
        reasons: list[str] = []
        requires_review = False

        for index, claim in enumerate(claims):
            cls = classifier.get(index)
            gnd = grounding.get(index)
            msr = measurement.get(index)

            claim.classifier_score = cls.score if cls else None
            claim.grounding_score = gnd.score if gnd else None

            # --- Rule 1: the draft asserts something the classifier denies ----
            if (
                claim.polarity == POSITIVE
                and cls is not None
                and cls.verdict == CONTRADICTED
                and self._grounding_permits_rule1(gnd)
            ):
                before = claim.text
                claim.text = hedge(claim.text)
                claim.polarity = UNCERTAIN
                claim.status = UNSUPPORTED
                requires_review = True
                edits.append(
                    ClaimEdit(
                        rule=RULE_POSITIVE_CONTRADICTED,
                        finding=claim.finding,
                        before=before,
                        after=claim.text,
                        premises=self._premises(cls, gnd),
                        requires_review=True,
                    )
                )
                reasons.append(
                    f"{claim.finding}: draft asserted positive, classifier contradicted"
                )

            # --- Rule 2: the draft denies something the evidence supports -----
            elif (
                claim.polarity == NEGATIVE
                and cls is not None
                and cls.verdict == CONTRADICTED
                and self._grounding_permits_rule2(gnd)
            ):
                claim.status = UNSUPPORTED
                requires_review = True
                # The draft text is NOT rewritten into a positive assertion:
                # the evidence disagrees, it does not establish the finding.
                edits.append(
                    ClaimEdit(
                        rule=RULE_POSSIBLE_OMISSION,
                        finding=claim.finding,
                        before=claim.text,
                        after=claim.text,
                        premises=self._premises(cls, gnd),
                        requires_review=True,
                    )
                )
                reasons.append(
                    f"{claim.finding}: possible omission -- draft denied it, image evidence disagrees"
                )

            # --- Rule 3: unconfirmed measurement ------------------------------
            if msr is not None and claim.has_measurement and msr.verdict != SUPPORTED:
                before = claim.text
                claim.text = strip_measurements(claim.text)
                requires_review = True
                edits.append(
                    ClaimEdit(
                        rule=RULE_UNCONFIRMED_MEASUREMENT,
                        finding=claim.finding,
                        before=before,
                        after=claim.text,
                        premises={"measurement": msr.verdict, "detail": msr.detail},
                        requires_review=True,
                    )
                )
                reasons.append(f"{claim.finding}: measurement could not be confirmed")

            # --- Rule 4: uncertainty above threshold --------------------------
            if (
                claim.uncertainty is not None
                and claim.uncertainty > self.uncertainty_threshold
                and claim.polarity == POSITIVE
            ):
                before = claim.text
                claim.text = hedge(claim.text)
                claim.polarity = UNCERTAIN
                claim.status = STATUS_UNCERTAIN
                edits.append(
                    ClaimEdit(
                        rule=RULE_HIGH_UNCERTAINTY,
                        finding=claim.finding,
                        before=before,
                        after=claim.text,
                        premises={
                            "uncertainty": claim.uncertainty,
                            "threshold": self.uncertainty_threshold,
                        },
                    )
                )
                reasons.append(f"{claim.finding}: uncertainty above threshold")

            # --- Rule 5: nothing actually verified this claim -----------------
            # Only the classifier and grounding speak to whether the finding
            # exists. The measurement checker validates numbers, so its having
            # run says nothing about the claim's substance and must not suppress
            # this rule.
            if self._nothing_ran(cls, gnd) and claim.polarity == POSITIVE:
                before = claim.text
                claim.text = f"{claim.text.strip().rstrip('.')}. {INSUFFICIENT_EVIDENCE_TEXT}"
                claim.status = STATUS_UNCERTAIN
                requires_review = True
                edits.append(
                    ClaimEdit(
                        rule=RULE_INSUFFICIENT_EVIDENCE,
                        finding=claim.finding,
                        before=before,
                        after=claim.text,
                        premises=self._premises(cls, gnd),
                        requires_review=True,
                    )
                )
                reasons.append(f"{claim.finding}: no verifier could assess this claim")

            if claim.status == STATUS_UNCERTAIN and cls is not None and cls.verdict == SUPPORTED:
                claim.status = SUPPORTED

        abstained, abstain_reason = self._should_abstain(claims)
        if abstained:
            reasons.append(abstain_reason)
            requires_review = True

        return ReconciliationOutcome(
            claims=claims,
            edits=edits,
            abstained=abstained,
            requires_review=requires_review,
            reasons=reasons,
        )

    # -- premise helpers ---------------------------------------------------

    def _grounding_permits_rule1(self, gnd: VerificationResult | None) -> bool:
        """Rule 1 needs grounding to be unsupportive -- when grounding is required.

        Unavailable grounding never counts as unsupportive.
        """
        if not self.require_grounding:
            return True
        return gnd is not None and gnd.verdict == CONTRADICTED

    def _grounding_permits_rule2(self, gnd: VerificationResult | None) -> bool:
        if not self.require_grounding:
            return True
        return gnd is not None and gnd.verdict == SUPPORTED

    @staticmethod
    def _premises(cls: VerificationResult | None, gnd: VerificationResult | None) -> dict:
        return {
            "classifier": cls.verdict if cls else "not_run",
            "classifier_detail": cls.detail if cls else "",
            "grounding": gnd.verdict if gnd else "not_run",
            "grounding_evaluated": bool(gnd and gnd.ran),
        }

    @staticmethod
    def _nothing_ran(*results: VerificationResult | None) -> bool:
        return all(result is None or result.verdict == UNAVAILABLE for result in results)

    def _should_abstain(self, claims: list[Claim]) -> tuple[bool, str]:
        positives = [c for c in claims if c.polarity in (POSITIVE, UNCERTAIN)]
        if not positives:
            return False, ""
        unsupported = [c for c in positives if c.status == UNSUPPORTED]
        ratio = len(unsupported) / len(positives)
        if ratio >= self.abstain_unsupported_ratio:
            return True, (
                f"abstained: {len(unsupported)}/{len(positives)} asserted findings "
                f"are unsupported (ratio {ratio:.2f} >= {self.abstain_unsupported_ratio})"
            )
        return False, ""
