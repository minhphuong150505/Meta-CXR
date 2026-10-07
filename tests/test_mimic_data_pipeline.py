"""Synthetic CPU smoke tests for MIMIC report parsing and study sampling.

Run with ``python tests/test_mimic_data_pipeline.py``.  The test intentionally
loads dependency-light helper files directly, so it needs neither MIMIC data nor
the training environment's Torch/Pandas stack.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


parser = load_module("mimic_report_parser", "preporcessing/mimic_report_parser.py")
sampling = load_module("mimic_cxr_utils", "model/lavis/data/mimic_cxr_utils.py")


class FindingsAndImpressionTargetTest(unittest.TestCase):
    """The parser output must compose into a findings_and_impression target."""

    def setUp(self):
        import sys

        sys.path.insert(0, str(REPO_ROOT / "training"))
        from dataio import manifest

        self.manifest = manifest

    def _row(self, report_text: str) -> dict:
        findings, impression, _method = parser.get_target_text(report_text)
        findings = parser.clean_report_text(findings)
        impression = parser.clean_report_text(impression)
        return {
            "findings_clean": findings,
            "impression_clean": impression,
            "target_valid": bool(findings),
            "impression_valid": bool(impression),
        }

    def test_both_sections_round_trip_through_the_target_format(self):
        row = self._row(
            "FINDINGS: Heart size is normal. No pneumothorax.\n"
            "IMPRESSION: No acute cardiopulmonary process."
        )
        target = self.manifest.row_target(row, self.manifest.FINDINGS_AND_IMPRESSION)
        self.assertTrue(target)
        findings, impression = self.manifest.split_generated_report(target)
        self.assertEqual(findings, "Heart size is normal. No pneumothorax.")
        self.assertEqual(impression, "No acute cardiopulmonary process.")

    def test_impression_only_report_yields_no_combined_target(self):
        row = self._row("FINAL REPORT\nIMPRESSION: Mild pulmonary edema.")
        self.assertEqual(
            self.manifest.row_target(row, self.manifest.FINDINGS_AND_IMPRESSION), ""
        )
        self.assertEqual(
            self.manifest.row_target(row, self.manifest.IMPRESSION_ONLY),
            "Mild pulmonary edema.",
        )

    def test_findings_without_impression_yields_no_combined_target(self):
        row = self._row("FINDINGS: Lungs are clear.")
        self.assertEqual(
            self.manifest.row_target(row, self.manifest.FINDINGS_AND_IMPRESSION), ""
        )
        self.assertEqual(
            self.manifest.row_target(row, self.manifest.FINDINGS_ONLY), "Lungs are clear."
        )

    def test_deidentification_tokens_are_stripped_from_both_sections(self):
        row = self._row(
            "FINDINGS: Compared to [**2154-1-1**] there is new opacity.\n"
            "IMPRESSION: Findings discussed with Dr. [**Last Name**]."
        )
        target = self.manifest.row_target(row, self.manifest.FINDINGS_AND_IMPRESSION)
        self.assertNotIn("[**", target)


class ReportParserTest(unittest.TestCase):
    def test_explicit_findings_does_not_include_impression(self):
        findings, impression, method = parser.get_target_text(
            "FINDINGS: Heart size is normal.\nIMPRESSION: No acute disease."
        )
        self.assertEqual(findings, "Heart size is normal.")
        self.assertEqual(impression, "No acute disease.")
        self.assertEqual(method, "FINDINGS_TAG")
        self.assertNotIn("acute disease", findings)

    def test_impression_only_is_not_a_generation_target(self):
        findings, impression, method = parser.get_target_text(
            "FINAL REPORT\nIMPRESSION: Mild pulmonary edema."
        )
        self.assertEqual(findings, "")
        self.assertEqual(impression, "Mild pulmonary edema.")
        self.assertEqual(method, "IMPRESSION_ONLY")

    def test_combined_section_is_not_treated_as_pure_findings(self):
        findings, combined, method = parser.get_target_text(
            "FINDINGS/IMPRESSION: No acute cardiopulmonary process."
        )
        self.assertEqual(findings, "")
        self.assertEqual(combined, "No acute cardiopulmonary process.")
        self.assertEqual(method, "FINDINGS_IMPRESSION_COMBINED")

    def test_unlabelled_narrative_after_preamble_is_recovered(self):
        findings, _, method = parser.get_target_text(
            "INDICATION: cough\n\nCOMPARISON: None.\n\n"
            "The lungs are clear. Heart size is normal."
        )
        self.assertEqual(findings, "The lungs are clear. Heart size is normal.")
        self.assertEqual(method, "NARRATIVE_BODY")
        self.assertNotIn("cough", findings)

    def test_comparison_fallback_stops_before_impression(self):
        findings, impression, method = parser.get_target_text(
            "COMPARISON: Reviewed in comparison to prior. The lungs are clear.\n"
            "IMPRESSION: No acute disease."
        )
        self.assertEqual(findings, "The lungs are clear.")
        self.assertEqual(impression, "No acute disease.")
        self.assertEqual(method, "NARRATIVE_AFTER_COMPARISON")

    def test_inline_headers_and_token_count(self):
        findings, impression, _ = parser.get_target_text(
            "Findings: No focal opacity. Impression: Normal chest."
        )
        self.assertEqual(findings, "No focal opacity.")
        self.assertEqual(impression, "Normal chest.")
        self.assertEqual(parser.count_lexical_tokens(findings), 4)

    def test_header_only_narrative_is_not_a_generation_target(self):
        # The exam line was the whole "narrative"; the content is in IMPRESSION.
        findings, impression, method = parser.get_target_text(
            "FINAL REPORT\nAP CHEST, 10:11 A.M., ___\n\nHISTORY: Line placement.\n\n"
            "IMPRESSION: Tip in the SVC. No pneumothorax."
        )
        self.assertEqual(findings, "")
        self.assertEqual(impression, "Tip in the SVC. No pneumothorax.")
        self.assertEqual(method, "IMPRESSION_ONLY")

    def test_exam_header_and_reason_are_stripped_from_narrative(self):
        findings, _, method = parser.get_target_text(
            "FINAL REPORT\nSINGLE FRONTAL VIEW OF THE CHEST\n\nREASON FOR EXAM: fever\n\n"
            "The lungs are clear. No effusion."
        )
        self.assertEqual(findings, "The lungs are clear. No effusion.")
        self.assertEqual(method, "NARRATIVE_BODY")

    def test_header_prefix_keeps_the_content_after_it(self):
        findings, _, _ = parser.get_target_text(
            "CLINICAL INDICATION: cough\n\nFRONTAL AND LATERAL VIEWS OF THE CHEST: "
            "The lungs are clear."
        )
        self.assertEqual(findings, "The lungs are clear.")

    def test_same_line_header_prefix_keeps_the_sentence(self):
        for line, kept in (
            ("CHEST, SINGLE AP PORTABLE VIEW. The heart is normal.", "The heart is normal."),
            ("AP CHEST 10:11 A.M. ___. The tube is unchanged.", "The tube is unchanged."),
            ("STUDY: CHEST RADIOGRAPH. REPORT: Lungs clear.", "Lungs clear."),
            ("ONE VIEW OF THE CHEST: No effusion.", "No effusion."),
        ):
            findings, _, _ = parser.get_target_text("INDICATION: x\n\n" + line)
            self.assertEqual(findings, kept, line)
        findings, _, _ = parser.get_target_text(
            "INDICATION: x\n\nCHEST TUBE IN PLACE. No pneumothorax."
        )
        self.assertEqual(findings, "CHEST TUBE IN PLACE. No pneumothorax.")

    def test_exam_header_recognition_is_narrow(self):
        for header in ("AP CHEST, 10:11 A.M., ___", "PA AND LATERAL CHEST RADIOGRAPHS:",
                       "CHEST, SINGLE AP PORTABLE VIEW.",
                       "PORTABLE SUPINE CHEST FILM ___ AT 1:23 A.M."):
            self.assertTrue(parser.is_exam_header(header), header)
        # All-caps findings and sentence-case text must never be dropped.
        for text in ("CHEST TUBE IN PLACE.", "THE LUNGS ARE CLEAR.",
                     "Chest radiograph shows no change.", "NO PNEUMOTHORAX."):
            self.assertFalse(parser.is_exam_header(text), text)


class StudySamplingTest(unittest.TestCase):
    def test_anchor_priority_and_complementary_auxiliary(self):
        rows = [
            {"subject_id": 1, "study_id": 10, "ViewPosition": "AP"},
            {"subject_id": 1, "study_id": 10, "ViewPosition": "PA"},
            {"subject_id": 1, "study_id": 10, "ViewPosition": "PA"},
            {"subject_id": 1, "study_id": 10, "ViewPosition": "LATERAL"},
        ]
        [study] = sampling.build_study_index(rows, max_aux_views=1)
        self.assertEqual(study["anchor"], 1)
        # The repeated PA is skipped; AP has higher auxiliary priority than lateral.
        self.assertEqual(study["aux"], [0])
        self.assertEqual(study["anchor_view_id"], sampling.VIEW_ID_MAP["PA"])
        self.assertEqual(study["aux_view_ids"], [sampling.VIEW_ID_MAP["AP"]])

    def test_subject_is_part_of_study_identity(self):
        rows = [
            {"subject_id": 1, "study_id": 10, "ViewPosition": "PA"},
            {"subject_id": 2, "study_id": 10, "ViewPosition": "AP"},
        ]
        studies = sampling.build_study_index(rows)
        self.assertEqual(len(studies), 2)

    def test_multiple_auxiliaries_follow_priority(self):
        rows = [
            {"subject_id": 1, "study_id": 10, "ViewPosition": "LATERAL"},
            {"subject_id": 1, "study_id": 10, "ViewPosition": "PA"},
            {"subject_id": 1, "study_id": 10, "ViewPosition": "PA"},
            {"subject_id": 1, "study_id": 10, "ViewPosition": "AP"},
            {"subject_id": 1, "study_id": 10, "ViewPosition": "LL"},
        ]
        [two] = sampling.build_study_index(rows, max_aux_views=2)
        self.assertEqual(two["anchor"], 1)
        # The repeated PA never becomes an auxiliary; AP outranks lateral.
        self.assertEqual(two["aux"], [3, 0])
        [three] = sampling.build_study_index(rows, max_aux_views=3)
        # A second lateral-type image is kept: only the anchor projection is skipped.
        self.assertEqual(three["aux"], [3, 0, 4])
        self.assertEqual(
            three["aux_view_ids"],
            [sampling.VIEW_ID_MAP["AP"], sampling.VIEW_ID_MAP["LATERAL"],
             sampling.VIEW_ID_MAP["LL"]],
        )
        # max=1 is unchanged by the relaxed cap.
        [one] = sampling.build_study_index(rows, max_aux_views=1)
        self.assertEqual(one["aux"], [3])

    def test_aux_view_cap(self):
        with self.assertRaises(ValueError):
            sampling.build_study_index([], max_aux_views=sampling.MAX_SUPPORTED_AUX_VIEWS + 1)
        with self.assertRaises(ValueError):
            sampling.build_study_index([], max_aux_views=-1)


if __name__ == "__main__":
    unittest.main()
