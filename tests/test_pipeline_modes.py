"""Pipeline-mode resolution: the default must be native MedGemma."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))

import pipeline_modes as pm


class PipelineModeResolution(unittest.TestCase):
    def test_default_is_native_medgemma_without_stage1(self):
        self.assertEqual(pm.DEFAULT_PIPELINE_MODE, "medgemma_direct")
        modes = pm.resolve_pipeline_modes(pm.DEFAULT_PIPELINE_MODE)
        self.assertEqual([mode.name for mode in modes], ["medgemma_direct"])
        self.assertFalse(pm.requires_stage1(modes))

    def test_native_mode_declares_no_stage1_components(self):
        mode = pm.PIPELINE_MODES["medgemma_direct"]
        self.assertEqual(mode.image_mode, "native")
        self.assertFalse(mode.requires_stage1)
        self.assertFalse(mode.uses_mhcac_prompt)

    def test_qformer_modes_require_stage1(self):
        for name in ("meta_cxr_qformer", "meta_cxr_qformer_with_mhcac_prompt"):
            mode = pm.PIPELINE_MODES[name]
            self.assertEqual(mode.image_mode, "qformer")
            self.assertTrue(mode.requires_stage1)
            self.assertTrue(pm.requires_stage1([mode]))

    def test_mhcac_prompt_variant_is_distinct_from_plain_qformer(self):
        plain = pm.PIPELINE_MODES["meta_cxr_qformer"]
        with_prompt = pm.PIPELINE_MODES["meta_cxr_qformer_with_mhcac_prompt"]
        self.assertFalse(plain.uses_mhcac_prompt)
        self.assertTrue(with_prompt.uses_mhcac_prompt)
        self.assertNotEqual(plain.name, with_prompt.name)

    def test_ablation_runs_primary_pipeline_first(self):
        modes = pm.resolve_pipeline_modes(pm.ABLATION_MODE)
        self.assertEqual(
            [mode.name for mode in modes], ["medgemma_direct", "meta_cxr_qformer"]
        )
        self.assertTrue(pm.requires_stage1(modes))

    def test_legacy_image_mode_aliases_still_resolve(self):
        cases = {
            "native": ["medgemma_direct"],
            "qformer": ["meta_cxr_qformer"],
            "both": ["medgemma_direct", "meta_cxr_qformer"],
        }
        for legacy, expected in cases.items():
            modes = pm.resolve_pipeline_modes(legacy)
            self.assertEqual([mode.name for mode in modes], expected, legacy)

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            pm.resolve_pipeline_modes("native_medgemma_probably")

    def test_no_mode_name_is_ambiguous(self):
        # "qformer"/"native" alone described an implementation detail, not an
        # architecture; every exposed name must state the visual pathway.
        for name in pm.PIPELINE_MODES:
            self.assertNotIn(name, {"native", "qformer", "both"})


if __name__ == "__main__":
    unittest.main()
