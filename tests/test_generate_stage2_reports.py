"""CLI and cohort invariants for scripts/generate_stage2_reports.py.

Nothing here loads a model. What is pinned is the part that fails SILENTLY:
which studies get generated, and whether two runs can be compared at all.
"""

from __future__ import annotations

import sys
from hashlib import blake2b
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

gen = pytest.importorskip("scripts.generate_stage2_reports")


class TestSampleKey:
    def test_matches_the_historical_blake2b_of_dicom_id(self):
        """Existing JSONL must keep joining to newly generated rows.

        The native path has always written blake2b(dicom_id, 12). The key is
        now derived from the image filename stem so the Stage-1 path -- which
        carries no dicom_id -- can compute the same value. Those must agree,
        or --restrict-to silently matches nothing.
        """
        # Synthetic. MIMIC-CXR identifiers -- subject_id, study_id, dicom_id --
        # must never appear in this repository, and a realistic-looking one is
        # not worth the ambiguity even when invented.
        dicom = "SYNTHETIC-dicom-0000-0000-0000"
        historical = blake2b(dicom.encode(), digest_size=12).hexdigest()
        derived = gen.sample_key_for(f"/data/files/pXX/pSUBJECT/sSTUDY/{dicom}.jpg")
        assert derived == historical

    def test_is_independent_of_the_directory(self):
        a = gen.sample_key_for("/one/root/files/p10/x/y/abc.jpg")
        b = gen.sample_key_for("/another/root/abc.jpg")
        assert a == b


class TestSubsample:
    def test_is_deterministic_for_a_seed(self):
        records = [{"i": i} for i in range(100)]
        assert gen.subsample(records, 10, 16) == gen.subsample(records, 10, 16)

    def test_a_different_seed_selects_differently(self):
        records = [{"i": i} for i in range(100)]
        assert gen.subsample(records, 10, 16) != gen.subsample(records, 10, 17)

    def test_preserves_input_order(self):
        records = [{"i": i} for i in range(100)]
        chosen = [r["i"] for r in gen.subsample(records, 25, 16)]
        assert chosen == sorted(chosen)

    def test_limit_zero_or_oversized_keeps_everything(self):
        records = [{"i": i} for i in range(10)]
        assert gen.subsample(records, 0, 16) == records
        assert gen.subsample(records, 999, 16) == records


class TestPipelineModeWiring:
    """The mode decides the record source; it must not be guessable."""

    @pytest.mark.parametrize(
        "name, image_mode, needs_stage1",
        [
            ("medgemma_direct", "native", False),
            ("meta_cxr_qformer", "qformer", True),
            ("meta_cxr_native_qformer_guided", "native_qformer", True),
        ],
    )
    def test_modes_route_as_declared(self, name, image_mode, needs_stage1):
        from training.pipeline_modes import resolve_pipeline_modes

        (mode,) = resolve_pipeline_modes(name)
        assert mode.image_mode == image_mode
        assert mode.requires_stage1 is needs_stage1

    def test_ablation_mode_expands_to_two_and_must_be_refused(self):
        """Both arms would write one generated_test.jsonl over the other.

        Exactly the class of collision that made two concurrent runs share an
        output path on 2026-08-19 and left the wrong report on disk.
        """
        from training.pipeline_modes import resolve_pipeline_modes

        assert len(resolve_pipeline_modes("both_for_ablation")) == 2


class TestArgumentParsing:
    def test_frontal_only_can_actually_be_switched_off(self):
        """It could not before: store_true with default=True is always True."""
        on = gen.parse_args(["--output-dir", "/tmp/x"])
        off = gen.parse_args(["--output-dir", "/tmp/x", "--no-frontal-only"])
        assert on.frontal_only is True
        assert off.frontal_only is False

    def test_default_mode_is_the_documented_one(self):
        assert gen.parse_args(["--output-dir", "/tmp/x"]).pipeline_mode == "medgemma_direct"

    def test_stage1_cache_dir_defaults_to_none_so_output_dir_is_used(self):
        assert gen.parse_args(["--output-dir", "/tmp/x"]).stage1_cache_dir is None


class TestMainRefusesAmbiguousRuns:
    """These exit before torch is imported, so they run on a CPU box."""

    def test_ablation_mode_is_refused_rather_than_writing_one_file_twice(self, tmp_path):
        with pytest.raises(SystemExit) as excinfo:
            gen.main(["--output-dir", str(tmp_path), "--pipeline-mode", "both_for_ablation"])
        assert "run each mode separately" in str(excinfo.value)

    def test_unknown_mode_names_the_valid_ones(self, tmp_path):
        with pytest.raises(ValueError, match="unknown pipeline mode"):
            gen.main(["--output-dir", str(tmp_path), "--pipeline-mode", "qformer_guided"])

    def test_native_mode_without_a_manifest_says_so(self, tmp_path):
        with pytest.raises(SystemExit, match="--manifest and --image-root"):
            gen.main(["--output-dir", str(tmp_path)])


class TestEarlyValidation:
    """Refusals must land before the model stack is imported, not after.

    A forgotten flag otherwise costs a full torch/transformers/nltk import
    before it is reported, and on the training host that is a minute of GPU
    time per typo.
    """

    def _mode(self, name):
        from training.pipeline_modes import resolve_pipeline_modes

        (mode,) = resolve_pipeline_modes(name)
        return mode

    def test_native_without_manifest(self, tmp_path):
        args = gen.parse_args(["--output-dir", str(tmp_path)])
        with pytest.raises(SystemExit, match="--manifest and --image-root"):
            gen.validate_invocation(args, self._mode("medgemma_direct"))

    def test_soft_token_mode_refuses_zero_shot(self, tmp_path):
        args = gen.parse_args([
            "--output-dir", str(tmp_path), "--checkpoint-root", str(tmp_path),
            "--pipeline-mode", "meta_cxr_qformer",
        ])
        with pytest.raises(SystemExit, match="no zero-shot form"):
            gen.validate_invocation(args, self._mode("meta_cxr_qformer"))

    def test_soft_token_mode_refuses_an_adapter_without_img_proj(self, tmp_path):
        adapter = tmp_path / "adapter"
        adapter.mkdir()
        (adapter / "adapter_model.safetensors").write_bytes(b"")
        args = gen.parse_args([
            "--output-dir", str(tmp_path), "--checkpoint-root", str(tmp_path),
            "--pipeline-mode", "meta_cxr_qformer", "--adapter", str(adapter),
        ])
        with pytest.raises(SystemExit, match="img_proj.pt is missing"):
            gen.validate_invocation(args, self._mode("meta_cxr_qformer"))

    def test_native_qformer_refuses_without_a_prompt_config(self, tmp_path):
        adapter = tmp_path / "adapter"
        adapter.mkdir()
        (adapter / "img_proj.pt").write_bytes(b"")
        args = gen.parse_args([
            "--output-dir", str(tmp_path), "--checkpoint-root", str(tmp_path),
            "--pipeline-mode", "meta_cxr_native_qformer_guided",
            "--adapter", str(adapter),
        ])
        with pytest.raises(SystemExit, match="requires --prompt-config"):
            gen.validate_invocation(args, self._mode("meta_cxr_native_qformer_guided"))

    def test_a_fully_specified_soft_token_run_is_accepted(self, tmp_path):
        adapter = tmp_path / "adapter"
        adapter.mkdir()
        (adapter / "img_proj.pt").write_bytes(b"")
        args = gen.parse_args([
            "--output-dir", str(tmp_path), "--checkpoint-root", str(tmp_path),
            "--pipeline-mode", "meta_cxr_native_qformer_guided",
            "--adapter", str(adapter), "--prompt-config", "configs/x.yaml",
        ])
        gen.validate_invocation(args, self._mode("meta_cxr_native_qformer_guided"))

    def test_missing_checkpoint_root_is_caught_early(self, tmp_path):
        args = gen.parse_args([
            "--output-dir", str(tmp_path), "--pipeline-mode", "meta_cxr_qformer",
            "--checkpoint-root", str(tmp_path / "nope"),
        ])
        with pytest.raises(SystemExit, match="does not exist"):
            gen.validate_invocation(args, self._mode("meta_cxr_qformer"))


def test_the_local_soft_token_set_matches_the_engine():
    """Skipped on a CPU box; the drift it guards is caught on the host."""
    fig9 = pytest.importorskip("training.train_eval_figure9_llm_variants_200")
    assert fig9.SOFT_TOKEN_MODES == gen.SOFT_TOKEN_IMAGE_MODES
