"""Per-section scoring must make an omitted IMPRESSION visible.

Corpus BLEU over the joined report barely penalises a missing IMPRESSION,
because the reference impression is short. These tests pin the behaviour that
makes the omission observable. Pure string logic -- no torch, no NLTK.
"""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "training"))

from training.dataio.manifest import (  # noqa: E402
    FINDINGS_AND_IMPRESSION,
    format_report,
    split_generated_report,
)
from training.stage2_utils import (  # noqa: E402
    prefix_metric_keys,
    section_omission_rate,
)


class TestSectionOmissionRate(unittest.TestCase):
    def test_counts_only_rows_whose_reference_has_the_section(self):
        preds = ["", "something", ""]
        refs = ["a reference", "another", ""]
        # Row 3 has no reference section, so there is nothing to omit there.
        self.assertAlmostEqual(section_omission_rate(preds, refs), 0.5)

    def test_all_present_is_zero(self):
        self.assertEqual(section_omission_rate(["a", "b"], ["x", "y"]), 0.0)

    def test_all_omitted_is_one(self):
        self.assertEqual(section_omission_rate(["", "  "], ["x", "y"]), 1.0)

    def test_empty_reference_cohort_is_zero_not_one(self):
        """Nothing to omit must not read as total omission."""
        self.assertEqual(section_omission_rate(["", ""], ["", ""]), 0.0)

    def test_whitespace_only_prediction_counts_as_omitted(self):
        self.assertEqual(section_omission_rate(["\n \t"], ["real impression"]), 1.0)

    def test_length_mismatch_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "differ in length"):
            section_omission_rate(["a"], ["a", "b"])


class TestSplitRoundTrip(unittest.TestCase):
    def test_format_then_split_round_trips(self):
        text = format_report("Lungs are clear.", "No acute process.", FINDINGS_AND_IMPRESSION)
        findings, impression = split_generated_report(text)
        self.assertEqual(findings, "Lungs are clear.")
        self.assertEqual(impression, "No acute process.")

    def test_unheadered_generation_is_not_duplicated_into_both(self):
        """The failure this guards: crediting findings text as the impression."""
        findings, impression = split_generated_report("Lungs are clear.")
        self.assertEqual(findings, "Lungs are clear.")
        self.assertEqual(impression, "")

    def test_omitted_impression_is_visible_after_split(self):
        reference = format_report("Lungs are clear.", "No acute process.", FINDINGS_AND_IMPRESSION)
        prediction = format_report("Lungs are clear.", "", FINDINGS_AND_IMPRESSION)

        _, ref_impression = split_generated_report(reference)
        _, pred_impression = split_generated_report(prediction)

        self.assertTrue(ref_impression)
        self.assertFalse(pred_impression)
        self.assertEqual(
            section_omission_rate([pred_impression], [ref_impression]), 1.0
        )


class TestMetricNamespacing(unittest.TestCase):
    def test_section_blocks_cannot_collide(self):
        findings = prefix_metric_keys({"BLEU-4": 0.3, "BERTScore": 0.8}, "Findings")
        impression = prefix_metric_keys({"BLEU-4": 0.1, "BERTScore": 0.5}, "Impression")
        merged = {**findings, **impression}
        self.assertEqual(len(merged), 4)
        self.assertEqual(merged["Findings/BLEU-4"], 0.3)
        self.assertEqual(merged["Impression/BLEU-4"], 0.1)

    def test_prefixed_keys_do_not_overwrite_full_report_metrics(self):
        full = {"BLEU-4": 0.25}
        full.update(prefix_metric_keys({"BLEU-4": 0.9}, "Findings"))
        self.assertEqual(full["BLEU-4"], 0.25)
        self.assertEqual(full["Findings/BLEU-4"], 0.9)


if __name__ == "__main__":
    unittest.main()
