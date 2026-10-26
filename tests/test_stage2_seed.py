"""run_medgemma_qlora.py --seed (2026-10-26): training randomness only, default bit-identical."""

import json
import sys
import types

import pytest


def test_default_seed_is_the_historical_one(monkeypatch, tmp_path):
    train = pytest.importorskip("training.run_medgemma_qlora")
    monkeypatch.setattr(sys, "argv", ["train", "--output-dir", str(tmp_path)])
    assert train.parse_args().seed == train.fig9.SEED == 16


def test_train_seed_enters_the_cache_identity_only_for_augmented_train_records(tmp_path):
    """Existing caches must still hit at the default seed, and for val/test at any seed."""
    fig9 = pytest.importorskip("training.train_eval_figure9_llm_variants_200")
    from training.run_context import Stage1Context

    context = Stage1Context(run_name="mimic_cxr_full_blip2", thresholds={})

    def fp(split, seed, clean=False):
        return fig9.stage1_cohort_fingerprint(
            context, tmp_path, split, None, "argmax", fig9.FINDING_TOKENS_OFF, clean,
            train_seed=seed,
        )

    default_id, default_payload = fig9.stage1_cohort_fingerprint(
        context, tmp_path, "train", None, "argmax"
    )
    assert "train_seed" not in default_payload
    assert fp("train", fig9.SEED)[0] == default_id
    other_id, other_payload = fp("train", 17)
    assert other_id != default_id
    assert other_payload["train_seed"] == 17
    # clean train records carry no augmentation draw, so the seed cannot change them
    assert "train_seed" not in fp("train", 17, clean=True)[1]
    for split in ("val", "test"):
        assert fp(split, 17)[0] == fp(split, fig9.SEED)[0]


@pytest.mark.parametrize("flags, expected", [([], 16), (["--seed", "17"], 17)])
def test_seed_reaches_the_rng_the_train_records_and_the_manifests(monkeypatch, tmp_path, flags, expected):
    train = pytest.importorskip("training.run_medgemma_qlora")
    checkpoint = tmp_path / "synthetic.pth"
    checkpoint.touch()
    output = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", [
        "run_medgemma_qlora.py", "--pipeline-mode", "meta_cxr_qformer_with_mhcac_prompt",
        "--section-mode", "findings_only", "--stage1-checkpoint", str(checkpoint),
        "--output-dir", str(output), "--no-upload", *flags,
    ])
    seeds, record_calls = [], []

    def records(*args, **kwargs):
        record_calls.append((args[3], kwargs.get("train_seed")))
        return [train.fig9.with_cue_state({"pred_groups": {}}, kwargs["cue_rule"])]

    monkeypatch.setattr(train.fig9, "set_seed", seeds.append)
    monkeypatch.setattr(train.fig9, "build_stage1_records", records)
    monkeypatch.setattr(train, "train_mode", lambda context, args, mode, *rest: ({}, output / "adapter"))
    train.main()
    assert seeds == [expected]
    assert record_calls == [("train", expected), ("val", None), ("test", None)]
    assert json.loads((output / "run_manifest.json").read_text())["seed"] == expected
    assert json.loads((output / "summary.json").read_text())["seed"] == expected


def test_seed_reaches_the_oracle_mix_draw(monkeypatch):
    train = pytest.importorskip("training.run_medgemma_qlora")
    from dataio import oracle_cues

    draws = []
    monkeypatch.setitem(sys.modules, "local_config", types.SimpleNamespace(CHEXPERT_CSV="unused"))
    monkeypatch.setattr(oracle_cues, "true_label_codes", lambda *a, **kw: {})
    monkeypatch.setattr(oracle_cues, "mix_oracle_cues", lambda *a, **kw: draws.append(kw["seed"]) or {})
    monkeypatch.setattr(train.fig9, "_split_csv_for", lambda split: "unused")
    args = types.SimpleNamespace(prompt_style="fine", train_cues="oracle_mix",
                                 train_cue_model_share=0.2, seed=17)
    train.attach_prompt_inputs(args, [{}], [], [])
    assert draws == [17]
