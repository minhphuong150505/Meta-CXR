"""Safety/XAI invariants. Pure stdlib -- no torch, no model, no network."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safety.claims import (  # noqa: E402
    NEGATIVE,
    POSITIVE,
    UNCERTAIN,
    UNSUPPORTED,
    Claim,
    LexiconClaimParser,
)
from safety.pipeline import ABSTENTION_TEXT, SafetyPipeline  # noqa: E402
from safety.reconciler import (  # noqa: E402
    RULE_INSUFFICIENT_EVIDENCE,
    RULE_POSITIVE_CONTRADICTED,
    RULE_POSSIBLE_OMISSION,
    RULE_UNCONFIRMED_MEASUREMENT,
    RuleBasedClaimReconciler,
)
from safety.verifiers import (  # noqa: E402
    CONTRADICTED,
    INCONCLUSIVE,
    SUPPORTED,
    UNAVAILABLE,
    ClassifierAbnormalityVerifier,
    EntropyUncertaintyEstimator,
    RegexMeasurementChecker,
    UnavailablePhraseGrounding,
    VerificationResult,
)

CONFIDENT_NEGATIVE = {"negative": 0.92, "positive": 0.04, "uncertain": 0.04}
CONFIDENT_POSITIVE = {"negative": 0.04, "positive": 0.92, "uncertain": 0.04}
UNIFORM = {"negative": 1 / 3, "positive": 1 / 3, "uncertain": 1 / 3}


class TestParserPolarity(unittest.TestCase):
    def setUp(self):
        self.parser = LexiconClaimParser()

    def _polarity(self, sentence, finding):
        for claim in self.parser.parse(sentence):
            if claim.finding == finding:
                return claim.polarity
        self.fail(f"no claim for {finding} in {sentence!r}")

    def test_negation_is_detected_not_deleted(self):
        """'No pneumothorax' is a negative claim, never a positive one."""
        claims = self.parser.parse("No pneumothorax.")
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].polarity, NEGATIVE)
        # The negation cue survives in the claim text.
        self.assertIn("No", claims[0].text)

    def test_hedge_is_uncertain_not_negative(self):
        """'cannot exclude' contains 'not' and must not read as a denial."""
        self.assertEqual(
            self._polarity("Cannot exclude pneumonia.", "Pneumonia"), UNCERTAIN
        )

    def test_trailing_hedge_is_uncertain(self):
        self.assertEqual(
            self._polarity("Pneumonia cannot be excluded.", "Pneumonia"), UNCERTAIN
        )

    def test_plain_assertion_is_positive(self):
        self.assertEqual(
            self._polarity("There is a large pleural effusion.", "Pleural Effusion"),
            POSITIVE,
        )

    def test_negation_does_not_leak_across_clauses(self):
        """A cue in an earlier clause must not negate a later finding."""
        sentence = "There is a pleural effusion, no pneumothorax."
        self.assertEqual(self._polarity(sentence, "Pleural Effusion"), POSITIVE)
        self.assertEqual(self._polarity(sentence, "Pneumothorax"), NEGATIVE)

    def test_location_and_severity_extracted(self):
        claim = self.parser.parse("Large right pleural effusion.")[0]
        self.assertEqual(claim.severity, "large")
        self.assertEqual(claim.location, "right")

    def test_unmatched_sentence_yields_no_claim(self):
        """Never invent a claim for text the lexicon does not cover."""
        self.assertEqual(self.parser.parse("The patient is stable."), [])


class TestUnavailableIsNotContradicted(unittest.TestCase):
    """The failure this layer exists to prevent."""

    def test_grounding_reports_unavailable_with_no_score(self):
        result, evidence = UnavailablePhraseGrounding().verify(
            Claim(text="x", finding="Edema", polarity=POSITIVE)
        )
        self.assertEqual(result.verdict, UNAVAILABLE)
        self.assertFalse(result.ran)
        self.assertIsNone(result.score)
        self.assertEqual(evidence.type, "none")

    def test_classifier_without_that_finding_is_unavailable(self):
        verifier = ClassifierAbnormalityVerifier({})
        result = verifier.verify(Claim(text="x", finding="Edema", polarity=POSITIVE))
        self.assertEqual(result.verdict, UNAVAILABLE)

    def test_requiring_grounding_blocks_rule1_when_unavailable(self):
        """With require_grounding, missing grounding must not strip a claim."""
        claim = Claim(text="There is edema.", finding="Edema", polarity=POSITIVE)
        outcome = RuleBasedClaimReconciler(require_grounding=True).reconcile(
            [claim],
            {0: VerificationResult(CONTRADICTED, 0.05, "")},
            {0: VerificationResult(UNAVAILABLE, None, "")},
            {0: VerificationResult(SUPPORTED, None, "")},
        )
        fired = [e.rule for e in outcome.edits]
        self.assertNotIn(RULE_POSITIVE_CONTRADICTED, fired)


class TestReconciliationRules(unittest.TestCase):
    def setUp(self):
        self.reconciler = RuleBasedClaimReconciler()

    def _run(self, claim, cls, gnd=None, msr=None):
        return self.reconciler.reconcile(
            [claim],
            {0: cls} if cls else {},
            {0: gnd or VerificationResult(UNAVAILABLE, None, "")},
            {0: msr or VerificationResult(SUPPORTED, None, "")},
        )

    def test_rule1_positive_contradicted_is_not_kept_as_is(self):
        claim = Claim(text="There is edema.", finding="Edema", polarity=POSITIVE)
        outcome = self._run(claim, VerificationResult(CONTRADICTED, 0.05, "negative"))
        self.assertIn(RULE_POSITIVE_CONTRADICTED, [e.rule for e in outcome.edits])
        self.assertNotEqual(claim.text, "There is edema.")
        self.assertEqual(claim.polarity, UNCERTAIN)
        self.assertEqual(claim.status, UNSUPPORTED)

    def test_rule2_flags_omission_without_asserting_the_finding(self):
        """A denial the evidence disputes is flagged, not rewritten to positive."""
        claim = Claim(text="No edema.", finding="Edema", polarity=NEGATIVE)
        outcome = self._run(claim, VerificationResult(CONTRADICTED, 0.9, "positive"))
        self.assertIn(RULE_POSSIBLE_OMISSION, [e.rule for e in outcome.edits])
        self.assertEqual(claim.text, "No edema.")
        self.assertNotEqual(claim.polarity, POSITIVE)
        self.assertTrue(outcome.requires_review)

    def test_rule3_strips_unconfirmed_measurement(self):
        claim = Claim(text="A 3.2 cm nodule.", finding="Lung Lesion", polarity=POSITIVE)
        outcome = self._run(
            claim,
            VerificationResult(SUPPORTED, 0.9, ""),
            msr=VerificationResult(INCONCLUSIVE, None, "no source"),
        )
        self.assertIn(RULE_UNCONFIRMED_MEASUREMENT, [e.rule for e in outcome.edits])
        self.assertNotIn("3.2", claim.text)
        self.assertTrue(outcome.requires_review)

    def test_rule4_hedges_high_uncertainty(self):
        claim = Claim(text="There is edema.", finding="Edema", polarity=POSITIVE)
        claim.uncertainty = 0.95
        outcome = self._run(claim, VerificationResult(SUPPORTED, 0.6, ""))
        self.assertIn("high_uncertainty", [e.rule for e in outcome.edits])
        self.assertEqual(claim.polarity, UNCERTAIN)

    def test_rule5_states_insufficient_evidence_when_nothing_assessed(self):
        claim = Claim(text="There is edema.", finding="Edema", polarity=POSITIVE)
        outcome = self._run(claim, VerificationResult(UNAVAILABLE, None, ""))
        self.assertIn(RULE_INSUFFICIENT_EVIDENCE, [e.rule for e in outcome.edits])
        self.assertIn("Insufficient evidence", claim.text)

    def test_rule5_ignores_measurement_checker_availability(self):
        """A ran measurement check says nothing about whether the finding exists."""
        claim = Claim(text="There is edema.", finding="Edema", polarity=POSITIVE)
        outcome = self._run(
            claim,
            VerificationResult(UNAVAILABLE, None, ""),
            msr=VerificationResult(SUPPORTED, None, "no measurement stated"),
        )
        self.assertIn(RULE_INSUFFICIENT_EVIDENCE, [e.rule for e in outcome.edits])

    def test_rule6_every_edit_carries_an_audit_trail(self):
        claim = Claim(text="There is edema.", finding="Edema", polarity=POSITIVE)
        outcome = self._run(claim, VerificationResult(CONTRADICTED, 0.05, "negative"))
        for edit in outcome.edits:
            self.assertTrue(edit.rule)
            self.assertTrue(edit.before)
            self.assertIn("classifier", edit.premises)
            self.assertIn("grounding_evaluated", edit.premises)
            self.assertFalse(edit.premises["grounding_evaluated"])


class TestUncertainty(unittest.TestCase):
    def test_unknown_finding_returns_none_not_zero(self):
        """'Unknown' must not be readable as 'confident'."""
        estimator = EntropyUncertaintyEstimator({})
        value = estimator.estimate(Claim(text="x", finding="Edema", polarity=POSITIVE))
        self.assertIsNone(value)

    def test_uniform_distribution_is_maximally_uncertain(self):
        estimator = EntropyUncertaintyEstimator({"Edema": UNIFORM})
        value = estimator.estimate(Claim(text="x", finding="Edema", polarity=POSITIVE))
        self.assertAlmostEqual(value, 1.0, places=6)

    def test_confident_distribution_is_low(self):
        estimator = EntropyUncertaintyEstimator({"Edema": CONFIDENT_POSITIVE})
        value = estimator.estimate(Claim(text="x", finding="Edema", polarity=POSITIVE))
        self.assertLess(value, 0.5)


class TestMeasurementChecker(unittest.TestCase):
    def test_no_measurement_is_supported(self):
        result = RegexMeasurementChecker().check(
            Claim(text="A nodule.", finding="Lung Lesion", polarity=POSITIVE)
        )
        self.assertEqual(result.verdict, SUPPORTED)

    def test_measurement_without_source_is_inconclusive_not_supported(self):
        result = RegexMeasurementChecker().check(
            Claim(text="A 3.2 cm nodule.", finding="Lung Lesion", polarity=POSITIVE)
        )
        self.assertEqual(result.verdict, INCONCLUSIVE)

    def test_matching_measurement_is_confirmed(self):
        result = RegexMeasurementChecker([32.0]).check(
            Claim(text="A 3.2 cm nodule.", finding="Lung Lesion", polarity=POSITIVE)
        )
        self.assertEqual(result.verdict, SUPPORTED)

    def test_mismatched_measurement_is_contradicted(self):
        result = RegexMeasurementChecker([10.0]).check(
            Claim(text="A 3.2 cm nodule.", finding="Lung Lesion", polarity=POSITIVE)
        )
        self.assertEqual(result.verdict, CONTRADICTED)

    def test_cm_and_mm_are_normalised(self):
        self.assertEqual(RegexMeasurementChecker.to_millimetres("3.2 cm"), 32.0)
        self.assertEqual(RegexMeasurementChecker.to_millimetres("8 mm"), 8.0)


class TestEndToEnd(unittest.TestCase):
    def test_contradicted_draft_is_downgraded_and_flagged(self):
        report = SafetyPipeline().run(
            "There is a large right pleural effusion.",
            classifier_probabilities={"Pleural Effusion": CONFIDENT_NEGATIVE},
        )
        self.assertTrue(report.outcome.requires_review)
        self.assertNotIn("There is a large right pleural effusion.", report.final_report)

    def test_supported_draft_is_preserved(self):
        report = SafetyPipeline().run(
            "There is a large right pleural effusion.",
            classifier_probabilities={"Pleural Effusion": CONFIDENT_POSITIVE},
        )
        self.assertIn("large right pleural effusion", report.final_report)
        self.assertFalse(report.outcome.abstained)

    def test_abstains_when_most_assertions_are_unsupported(self):
        report = SafetyPipeline().run(
            "There is edema. There is consolidation.",
            classifier_probabilities={
                "Edema": CONFIDENT_NEGATIVE,
                "Consolidation": CONFIDENT_NEGATIVE,
            },
        )
        self.assertTrue(report.outcome.abstained)
        self.assertEqual(report.final_report, ABSTENTION_TEXT)

    def test_availability_reports_grounding_as_absent(self):
        report = SafetyPipeline().run(
            "There is edema.", classifier_probabilities={"Edema": CONFIDENT_POSITIVE}
        )
        self.assertFalse(report.verifier_availability["phrase_grounding"])
        self.assertTrue(report.verifier_availability["abnormality_verifier"])

    def test_parse_coverage_exposes_unchecked_sentences(self):
        report = SafetyPipeline().run(
            "There is edema. The patient is stable. Comparison is made to a prior study."
        )
        self.assertEqual(len(report.unparsed), 2)
        self.assertLess(report.parse_coverage, 0.5)

    def test_output_record_carries_no_identifiers(self):
        record = SafetyPipeline().run(
            "There is edema.", classifier_probabilities={"Edema": CONFIDENT_POSITIVE}
        ).to_dict()
        flat = repr(record)
        for forbidden in ("subject_id", "study_id", "dicom_id", "image_path"):
            self.assertNotIn(forbidden, flat)


if __name__ == "__main__":
    unittest.main()
