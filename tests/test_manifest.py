"""Manifest invariants: section targets, anchor selection, and split leakage."""

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))

from dataio import manifest as mf


def make_frame(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "subject_id": 1,
        "study_id": 10,
        "dicom_id": "d0",
        "image_path": "files/p10/p1/s10/d0.jpg",
        "ViewPosition": "PA",
        "findings_clean": "Heart size is normal.",
        "impression_clean": "No acute disease.",
        "target_valid": True,
        "impression_valid": True,
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


class SectionTargets(unittest.TestCase):
    def test_structured_report_keeps_sections_separable(self):
        text = mf.format_report("Lungs are clear.", "No pneumonia.", mf.FINDINGS_AND_IMPRESSION)
        self.assertIn(mf.FINDINGS_HEADER, text)
        self.assertIn(mf.IMPRESSION_HEADER, text)
        findings, impression = mf.split_generated_report(text)
        self.assertEqual(findings, "Lungs are clear.")
        self.assertEqual(impression, "No pneumonia.")

    def test_single_section_modes_emit_bare_text(self):
        self.assertEqual(mf.format_report("A.", "B.", mf.FINDINGS_ONLY), "A.")
        self.assertEqual(mf.format_report("A.", "B.", mf.IMPRESSION_ONLY), "B.")

    def test_missing_impression_is_not_duplicated_from_findings(self):
        findings, impression = mf.split_generated_report("Lungs are clear.")
        self.assertEqual(findings, "Lungs are clear.")
        self.assertEqual(impression, "")

    def test_negations_survive_section_formatting(self):
        source = "No pneumothorax. Cannot exclude edema. Unchanged without effusion."
        text = mf.format_report(source, "No acute process.", mf.FINDINGS_AND_IMPRESSION)
        findings, _ = mf.split_generated_report(text)
        for phrase in ("No ", "Cannot exclude", "unchanged without", "without"):
            self.assertIn(phrase.lower(), findings.lower())

    def test_findings_and_impression_requires_both_sections(self):
        row = {"target_valid": True, "impression_valid": False,
               "findings_clean": "F.", "impression_clean": ""}
        self.assertEqual(mf.row_target(row, mf.FINDINGS_AND_IMPRESSION), "")
        self.assertEqual(mf.row_target(row, mf.FINDINGS_ONLY), "F.")

    def test_invalid_flag_blocks_target_even_with_text_present(self):
        row = {"target_valid": False, "impression_valid": True,
               "findings_clean": "leftover text", "impression_clean": "I."}
        self.assertEqual(mf.row_target(row, mf.FINDINGS_ONLY), "")
        self.assertEqual(mf.row_target(row, mf.IMPRESSION_ONLY), "I.")

    def test_unknown_section_mode_raises(self):
        with self.assertRaises(ValueError):
            mf.row_target({"target_valid": True, "findings_clean": "F."}, "full_report")


class ManifestSchema(unittest.TestCase):
    def test_impression_mode_rejects_pre_impression_manifest(self):
        frame = make_frame([{}]).drop(columns=["impression_clean", "impression_valid"])
        mf.assert_columns(frame, mf.FINDINGS_ONLY, "train")
        with self.assertRaises(mf.ManifestError) as ctx:
            mf.assert_columns(frame, mf.FINDINGS_AND_IMPRESSION, "train")
        self.assertIn("impression_clean", str(ctx.exception))


class AnchorSelection(unittest.TestCase):
    def test_one_row_per_study_with_priority_view(self):
        frame = make_frame([
            {"dicom_id": "lat", "ViewPosition": "LATERAL"},
            {"dicom_id": "pa", "ViewPosition": "PA"},
            {"subject_id": 2, "study_id": 20, "dicom_id": "ap", "ViewPosition": "AP"},
        ])
        anchors = mf.select_anchor_rows(frame)
        self.assertEqual(len(anchors), 2)
        first = anchors[anchors["study_id"] == 10].iloc[0]
        self.assertEqual(first["dicom_id"], "pa")

    def test_unknown_view_does_not_win_over_named_view(self):
        frame = make_frame([
            {"dicom_id": "weird", "ViewPosition": "XX"},
            {"dicom_id": "ap", "ViewPosition": "AP"},
        ])
        anchors = mf.select_anchor_rows(frame)
        self.assertEqual(len(anchors), 1)
        self.assertEqual(anchors.iloc[0]["dicom_id"], "ap")


class LeakageDetection(unittest.TestCase):
    def test_disjoint_splits_pass(self):
        frames = {
            "train": make_frame([{"subject_id": 1, "study_id": 10, "dicom_id": "a"}]),
            "val": make_frame([{"subject_id": 2, "study_id": 20, "dicom_id": "b"}]),
            "test": make_frame([{"subject_id": 3, "study_id": 30, "dicom_id": "c"}]),
        }
        mf.assert_no_leakage(frames)

    def test_subject_in_two_splits_is_rejected(self):
        frames = {
            "train": make_frame([{"subject_id": 1, "study_id": 10, "dicom_id": "a"}]),
            "test": make_frame([{"subject_id": 1, "study_id": 99, "dicom_id": "c"}]),
        }
        with self.assertRaises(mf.ManifestError) as ctx:
            mf.assert_no_leakage(frames)
        self.assertIn("subject_id", str(ctx.exception))

    def test_repeated_image_across_splits_is_rejected(self):
        frames = {
            "train": make_frame([{"subject_id": 1, "study_id": 10, "dicom_id": "same"}]),
            "test": make_frame([{"subject_id": 2, "study_id": 20, "dicom_id": "same"}]),
        }
        with self.assertRaises(mf.ManifestError):
            mf.assert_no_leakage(frames)


class RecordBuilding(unittest.TestCase):
    def test_records_carry_no_raw_identifiers(self):
        frame = make_frame([{"subject_id": 42, "study_id": 4242, "dicom_id": "secret"}])
        records = mf.build_records(
            frame, split="train", section_mode=mf.FINDINGS_AND_IMPRESSION,
            vis_root="/vis", cohort_id="c0",
        )
        self.assertEqual(len(records), 1)
        serialized = repr(records[0])
        for leaked in ("42", "4242", "secret"):
            self.assertNotIn(leaked, records[0]["sample_key"])
        self.assertEqual(
            set(records[0]),
            {"index", "sample_key", "ref", "image_path", "anchor_view", "auxiliary_views"},
        )
        # View metadata is populated from ViewPosition (no patient identifier).
        self.assertEqual(records[0]["anchor_view"], "PA")
        self.assertEqual(records[0]["auxiliary_views"], [])
        self.assertIn(mf.IMPRESSION_HEADER, serialized)

    def test_rows_failing_the_section_mode_are_dropped(self):
        frame = make_frame([
            {"dicom_id": "ok"},
            {"subject_id": 2, "study_id": 20, "dicom_id": "no_imp",
             "impression_valid": False, "impression_clean": ""},
        ])
        records = mf.build_records(
            frame, split="train", section_mode=mf.FINDINGS_AND_IMPRESSION,
            vis_root="/vis", cohort_id="c0",
        )
        self.assertEqual(len(records), 1)

    def test_empty_result_fails_loudly(self):
        frame = make_frame([{"target_valid": False, "impression_valid": False,
                             "findings_clean": "", "impression_clean": ""}])
        with self.assertRaises(mf.ManifestError):
            mf.build_records(
                frame, split="train", section_mode=mf.FINDINGS_AND_IMPRESSION,
                vis_root="/vis", cohort_id="c0",
            )

    def test_subset_limit_is_deterministic(self):
        frame = make_frame([
            {"subject_id": i, "study_id": i * 10, "dicom_id": f"d{i}"} for i in range(20)
        ])
        kwargs = dict(
            split="train", section_mode=mf.FINDINGS_ONLY, vis_root="/vis",
            cohort_id="c0", limit=5, seed=16,
        )
        first = mf.build_records(frame, **kwargs)
        second = mf.build_records(frame, **kwargs)
        self.assertEqual(len(first), 5)
        self.assertEqual([r["sample_key"] for r in first], [r["sample_key"] for r in second])


if __name__ == "__main__":
    unittest.main()
