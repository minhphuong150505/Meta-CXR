"""--clean-train-records: train Stage-1 records encoded without the training augmentation."""

import pytest


def test_cache_identity_changes_only_for_clean_train_records(tmp_path):
    """Existing caches must still hit when the flag is off, and for val/test either way."""
    fig9 = pytest.importorskip("training.train_eval_figure9_llm_variants_200")
    from training.run_context import Stage1Context

    context = Stage1Context(run_name="mimic_cxr_full_blip2", thresholds={})

    def fp(split, clean):
        return fig9.stage1_cohort_fingerprint(
            context, tmp_path, split, None, "argmax", fig9.FINDING_TOKENS_OFF, clean
        )

    default_id, default_payload = fp("train", False)
    clean_id, clean_payload = fp("train", True)
    assert default_id != clean_id
    assert "train_images" not in default_payload
    assert clean_payload["train_images"] == "clean"
    # the flag is absent from the positional default, so old call sites are unchanged
    assert default_id == fig9.stage1_cohort_fingerprint(
        context, tmp_path, "train", None, "argmax", fig9.FINDING_TOKENS_OFF
    )[0]
    for split in ("val", "test"):
        assert fp(split, False)[0] == fp(split, True)[0]


def test_stage1_options_enter_the_cache_identity_only_when_given(tmp_path):
    """--stage1-options (2026-10-15): a refit checkpoint must not hit an older cache."""
    fig9 = pytest.importorskip("training.train_eval_figure9_llm_variants_200")
    from training.run_context import Stage1Context

    plain = Stage1Context(run_name="mimic_cxr_full_blip2", thresholds={})
    refit = Stage1Context(run_name="mimic_cxr_full_blip2", thresholds={},
                          config_options=("model.qformer_input_norm=per_stream",))
    a, pa = fig9.stage1_cohort_fingerprint(plain, tmp_path, "val", None, "argmax")
    b, pb = fig9.stage1_cohort_fingerprint(refit, tmp_path, "val", None, "argmax")
    assert a != b
    assert "stage1_options" not in pa
    assert pb["stage1_options"] == ["model.qformer_input_norm=per_stream"]
