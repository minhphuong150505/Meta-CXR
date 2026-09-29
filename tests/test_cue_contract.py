"""Cue semantics and train/generation wiring using synthetic inputs only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from stage2.prompts import PromptBuilder, load_prompt_config
from stage2.prompts.records import context_from_record
from stage2.prompts.schemas import CueState, PartKind
from stage2.prompts.templates import COMPACT_NORMAL_STATEMENT, STRUCTURED_HEADER

ROOT = Path(__file__).resolve().parents[1]
PROMPT = ROOT / "configs/experiment_native_qformer_guided.yaml"


def render(record):
    config = load_prompt_config(PROMPT)
    context = context_from_record(
        record, visual_mode=config.visual_mode, qformer_token_count=32
    )
    return context, PromptBuilder(config).build(context)


@pytest.mark.parametrize("state", ["not_provided", "abstained"])
def test_empty_cues_never_assert_normal_and_keep_both_visual_channels(state):
    context, prompt = render({"cue_state": state, "pred_groups": {}})
    assert context.cue_state == state
    assert not context.is_structurally_normal
    assert STRUCTURED_HEADER not in prompt.user_text()
    assert COMPACT_NORMAL_STATEMENT not in prompt.user_text()
    assert sum(p.kind is PartKind.IMAGE for p in prompt.parts) == 1
    assert prompt.user_text().count("<qformer_soft_token>") == 32


def test_legacy_empty_groups_are_abstention_not_negative_evidence():
    context, prompt = render({"pred_groups": {}})
    assert context.cue_state is CueState.ABSTAINED
    assert STRUCTURED_HEADER not in prompt.user_text()


def test_explicit_withholding_is_allowed_but_missing_predictions_still_fail():
    render({"cue_state": "not_provided"})
    with pytest.raises(ValueError, match="carries none"):
        render({})


@pytest.mark.parametrize("state", ["not_provided", "abstained", "invalid"])
def test_contradictory_or_unknown_states_fail(state):
    with pytest.raises(ValueError):
        render({"cue_state": state, "pred_groups": {"positive": ["Edema"]}})


def test_negative_subset_is_explicit_and_does_not_claim_ontology_wide_normality():
    context, prompt = render({"pred_groups": {"negative": ["Pneumothorax"]}})
    assert context.cue_state is CueState.PREDICTED
    assert "Clinically relevant absent: Pneumothorax" in prompt.user_text()
    assert COMPACT_NORMAL_STATEMENT not in prompt.user_text()
    assert "Edema" not in prompt.user_text()


def _all_class_thresholds(value):
    fig9 = pytest.importorskip("training.train_eval_figure9_llm_variants_200")
    return {
        name: {"negative": value, "positive": value, "uncertain": value}
        for name in fig9.ABNORMALITIES_14
    }


@pytest.mark.parametrize("rule", ["none", "paper_thresholds"])
def test_classifier_to_prompt_abstention_and_withholding(rule):
    torch = pytest.importorskip("torch")
    fig9 = pytest.importorskip("training.train_eval_figure9_llm_variants_200")
    from training.run_context import Stage1Context

    logits = torch.zeros(14, 3)  # 1/3 each: no class clears a 0.5 threshold
    context_in = Stage1Context(run_name="synthetic", thresholds=_all_class_thresholds(0.5))
    groups = fig9.classify_with_thresholds(context_in, logits, cue_rule=rule)
    record = fig9.with_cue_state({"pred_groups": groups}, rule)
    context, prompt = render(record)
    expected = CueState.NOT_PROVIDED if rule == "none" else CueState.ABSTAINED
    assert context.cue_state is expected
    assert STRUCTURED_HEADER not in prompt.user_text()


def test_argmax_rule_lists_every_reportable_finding_under_its_class():
    torch = pytest.importorskip("torch")
    fig9 = pytest.importorskip("training.train_eval_figure9_llm_variants_200")
    from training.run_context import Stage1Context

    logits = torch.full((14, 3), -5.0)
    logits[:, 0] = 5.0
    logits[fig9.ABNORMALITIES_14.index("Edema"), :] = torch.tensor([-5.0, -5.0, 5.0])
    groups = fig9.classify_with_thresholds(Stage1Context(run_name="s"), logits, cue_rule="argmax")
    assert groups["uncertain"] == ["Edema"]
    assert len(groups["negative"]) == 12 and "No Finding" not in groups["negative"]


def test_paper_thresholds_take_the_class_furthest_above_its_threshold():
    torch = pytest.importorskip("torch")
    fig9 = pytest.importorskip("training.train_eval_figure9_llm_variants_200")
    from training.run_context import Stage1Context

    thresholds = _all_class_thresholds(0.99)
    thresholds["Edema"] = {"negative": 0.6, "positive": 0.2, "uncertain": 0.9}
    logits = torch.zeros(14, 3)
    # Edema probs [0.62, 0.30, 0.08]: negative clears by .02, positive by .10.
    logits[fig9.ABNORMALITIES_14.index("Edema")] = torch.log(torch.tensor([0.62, 0.30, 0.08]))
    groups = fig9.classify_with_thresholds(
        Stage1Context(run_name="s", thresholds=thresholds), logits, cue_rule="paper_thresholds"
    )
    assert groups == {"positive": ["Edema"], "negative": [], "uncertain": []}


def test_paper_thresholds_refuse_to_run_without_a_threshold_file():
    torch = pytest.importorskip("torch")
    fig9 = pytest.importorskip("training.train_eval_figure9_llm_variants_200")
    from training.run_context import Stage1Context

    with pytest.raises(ValueError, match="threshold"):
        fig9.classify_with_thresholds(
            Stage1Context(run_name="s"), torch.zeros(14, 3), cue_rule="paper_thresholds"
        )


def test_retired_binary_cue_rules_are_gone():
    fig9 = pytest.importorskip("training.train_eval_figure9_llm_variants_200")
    assert set(fig9.CUE_RULES) == {"argmax", "paper_thresholds", "none"}


def test_legacy_cache_hit_gets_cue_state_without_loading_a_model(monkeypatch, tmp_path):
    fig9 = pytest.importorskip("training.train_eval_figure9_llm_variants_200")
    from training.run_context import Stage1Context

    context = Stage1Context(run_name="synthetic")
    monkeypatch.setattr(fig9, "stage1_cohort_fingerprint", lambda *a: ("fixture", {}))
    cache = tmp_path / ".sensitive_stage1_cache"
    cache.mkdir()
    (cache / "synthetic_val_qformer_all_fixture.pt").touch()
    old_record = {"pred_groups": {}, "qformer_embs": object()}
    monkeypatch.setattr(fig9, "load_torch_checkpoint", lambda _: {
        "cohort_id": "fixture", "records": [old_record]
    })

    def no_model(*args, **kwargs):
        pytest.fail("cache hit must not load a Stage-1 model")

    monkeypatch.setattr(fig9, "build_stage1_model", no_model)
    records = fig9.build_stage1_records(
        context, tmp_path, tmp_path, "val", None, 0, cue_rule="none"
    )
    assert records[0]["cue_state"] == "not_provided"
    assert "cue_state" not in old_record
    assert records[0]["qformer_embs"] is old_record["qformer_embs"]
    assert STRUCTURED_HEADER not in render(records[0])[1].user_text()


@pytest.mark.parametrize("rule", [None, "argmax", "none"])
def test_training_and_generation_pass_same_rule_and_render_same_prompt(monkeypatch, tmp_path, rule):
    train = pytest.importorskip("training.run_medgemma_qlora")
    fig9 = pytest.importorskip("training.train_eval_figure9_llm_variants_200")
    from scripts import generate_stage2_reports as gen

    checkpoint = tmp_path / "synthetic.pth"
    checkpoint.touch()
    output = tmp_path / "run"
    rule_flags = [] if rule is None else ["--cue-rule", rule]
    rule = rule or "argmax"
    monkeypatch.setattr(sys, "argv", [
        "run_medgemma_qlora.py", "--pipeline-mode", "meta_cxr_native_qformer_guided",
        "--section-mode", "findings_only", "--prompt-config", str(PROMPT),
        "--stage1-checkpoint", str(checkpoint), "--output-dir", str(output),
        *rule_flags, "--no-upload",
    ])
    train_calls = []
    generation_calls = []
    groups = {"positive": ["Edema"]} if rule != "none" else {}

    def records(calls, *args, **kwargs):
        calls.append((args[3], kwargs["cue_rule"]))
        return [fig9.with_cue_state({"pred_groups": groups}, kwargs["cue_rule"])]

    monkeypatch.setattr(train.fig9, "build_stage1_records", lambda *a, **kw: records(train_calls, *a, **kw))
    monkeypatch.setattr(train.fig9, "set_seed", lambda _: None)
    rendered_train = []

    def train_mode(context, args, mode, train_records, val_records, test_records, root, prompt_config):
        for split_records in (train_records, val_records, test_records):
            rendered_train.append(render(split_records[0])[1].user_text())
        return {}, root / "synthetic_adapter"

    monkeypatch.setattr(train, "train_mode", train_mode)
    train.main()
    assert train_calls == [(split, rule) for split in ("train", "val", "test")]
    assert json.loads((output / "run_manifest.json").read_text())["cue_rule"] == rule
    monkeypatch.setattr(fig9, "build_stage1_records", lambda *a, **kw: records(generation_calls, *a, **kw))
    monkeypatch.setattr(fig9, "set_seed", lambda _: None)
    args = gen.parse_args([
        "--output-dir", str(tmp_path / "gen"), *rule_flags, "--split", "val",
        "--pipeline-mode", "meta_cxr_native_qformer_guided",
    ])
    generated_records = gen.stage1_records(args)
    assert generation_calls == [("val", rule)]
    assert rendered_train == [render(generated_records[0])[1].user_text()] * 3


@pytest.mark.parametrize("rule", ["paper_thresholds", "none"])
def test_generation_rejects_abstaining_cues_on_legacy_prompt(tmp_path, rule):
    from scripts import generate_stage2_reports as gen
    from training.pipeline_modes import resolve_pipeline_modes

    thresholds = tmp_path / "t.json"
    args = gen.parse_args([
        "--output-dir", str(tmp_path), "--cue-rule", rule, "--threshold-path", str(thresholds),
        "--pipeline-mode", "meta_cxr_qformer_with_mhcac_prompt",
    ])
    (mode,) = resolve_pipeline_modes(args.pipeline_mode)
    with pytest.raises(SystemExit, match="guided --prompt-config"):
        gen.validate_invocation(args, mode)


@pytest.mark.parametrize("mode", [
    "meta_cxr_native_qformer_guided",
    "meta_cxr_qformer_with_mhcac_prompt",
    "medgemma_direct",
    "meta_cxr_qformer",
    "both_for_ablation",
])
def test_both_cli_defaults_are_the_papers_argmax(monkeypatch, tmp_path, mode):
    from scripts import generate_stage2_reports as gen
    from training import run_medgemma_qlora as train

    flags = ["--pipeline-mode", mode, "--output-dir", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", ["train", *flags])
    assert train.parse_args().cue_rule == "argmax"
    assert gen.parse_args(flags).cue_rule == "argmax"


@pytest.mark.parametrize("retired", ["conditional_positive", "marginal_positive", "mention_gated"])
def test_both_clis_refuse_retired_binary_cue_rules(monkeypatch, tmp_path, retired):
    from scripts import generate_stage2_reports as gen
    from training import run_medgemma_qlora as train

    flags = ["--output-dir", str(tmp_path), "--cue-rule", retired]
    with pytest.raises(SystemExit):
        gen.parse_args(flags)
    monkeypatch.setattr(sys, "argv", ["train", *flags])
    with pytest.raises(SystemExit):
        train.parse_args()
