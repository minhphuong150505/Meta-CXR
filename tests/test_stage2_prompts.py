"""Unit tests for the Stage-2 prompt builder (stage2/prompts).

Torch-free except where a helper genuinely concerns token ids. Covers the 30
required cases: normal-study compaction, P/N/U handling, uncertainty preservation,
negative policies, metadata honesty, view/prior handling, temporal guards, train/
inference parity, label masking, soft-token counting and suppression, budget
truncation order, config validation and hash stability.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage2.prompts import (  # noqa: E402
    MODELED_FINDINGS,
    PromptBuilder,
    PromptConfig,
    PromptConfigError,
    PromptContext,
    TemporalTargetPolicy,
    VisualMode,
    apply_temporal_target_policy,
    config_from_mapping,
    fit_to_budget,
)
from stage2.prompts.policies import (  # noqa: E402
    NegativePolicy,
    NormalPolicy,
    UncertaintyPolicy,
)
from stage2.prompts.schemas import PartKind  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "training"))
from medgemma.soft_tokens import soft_token_bad_words_ids  # noqa: E402
from stage2_utils import masked_label_ids  # noqa: E402

QF = VisualMode.QFORMER_GUIDED


def build(config: PromptConfig, context: PromptContext):
    return PromptBuilder(config).build(context)


def ctx(mode=QF, **kw):
    if mode.uses_soft_tokens:
        kw.setdefault("qformer_token_count", 32)
    return PromptContext(study_id="s", visual_mode=mode, **kw)


# 1 -------------------------------------------------------------------------
def test_normal_study_does_not_dump_all_negatives():
    negatives = tuple(MODELED_FINDINGS)  # a fully-normal prediction
    rendered = build(
        PromptConfig(visual_mode=QF, normal_policy=NormalPolicy.COMPACT_SUMMARY),
        ctx(negative_findings=negatives),
    )
    text = rendered.user_text()
    assert "No high-confidence positive or uncertain abnormality" in text
    present = [name for name in MODELED_FINDINGS if name in text]
    assert len(present) < len(MODELED_FINDINGS)  # not a 13-item enumeration


# 2 -------------------------------------------------------------------------
def test_positive_findings_are_rendered():
    rendered = build(
        PromptConfig(visual_mode=QF),
        ctx(positive_findings=("Cardiomegaly", "Pleural Effusion")),
    )
    text = rendered.user_text()
    assert "Present: Cardiomegaly, Pleural Effusion" in text


# 3 -------------------------------------------------------------------------
def test_uncertain_finding_is_in_possible_group():
    rendered = build(PromptConfig(visual_mode=QF), ctx(uncertain_findings=("Atelectasis",)))
    assert "Possible or uncertain: Atelectasis" in rendered.user_text()


# 4 -------------------------------------------------------------------------
def test_uncertain_is_not_promoted_to_positive():
    rendered = build(PromptConfig(visual_mode=QF), ctx(uncertain_findings=("Atelectasis",)))
    text = rendered.user_text()
    assert "Present: none" in text
    assert "Present: Atelectasis" not in text


def test_probability_bins_without_uncertains_does_not_raise():
    """probability_bins must not demand probabilities when nothing is uncertain."""
    rendered = build(
        PromptConfig(visual_mode=QF, uncertainty_policy=UncertaintyPolicy.PROBABILITY_BINS),
        ctx(positive_findings=("Edema",)),  # no uncertain findings at all
    )
    assert "Possible or uncertain: none" in rendered.user_text()


def test_probability_bins_with_uncertains_requires_probabilities():
    with pytest.raises(ValueError, match="probability_bins requires"):
        build(
            PromptConfig(visual_mode=QF, uncertainty_policy=UncertaintyPolicy.PROBABILITY_BINS),
            ctx(uncertain_findings=("Atelectasis",)),  # uncertain but no probabilities
        )


# 5 -------------------------------------------------------------------------
def test_critical_negative_policy_filters_to_critical():
    rendered = build(
        PromptConfig(visual_mode=QF, negative_policy=NegativePolicy.CRITICAL_ONLY),
        ctx(
            positive_findings=("Cardiomegaly",),
            negative_findings=("Pneumothorax", "Lung Opacity", "Fracture", "Edema"),
        ),
    )
    text = rendered.user_text()
    assert "Pneumothorax" in text and "Edema" in text
    assert "Lung Opacity" not in text and "Fracture" not in text


# 6 -------------------------------------------------------------------------
def test_top_k_negative_respects_k():
    negatives = ("Pneumothorax", "Edema", "Cardiomegaly", "Fracture", "Atelectasis")
    rendered = build(
        PromptConfig(
            visual_mode=QF,
            negative_policy=NegativePolicy.TOP_K_CONFIDENT,
            max_negative_findings=2,
        ),
        ctx(positive_findings=("Consolidation",), negative_findings=negatives),
    )
    absent_line = [
        p.text for p in rendered.parts if p.text and p.text.startswith("- Clinically relevant absent")
    ][0]
    shown = absent_line.split(":", 1)[1].split(",")
    assert len(shown) == 2


# 7 -------------------------------------------------------------------------
def test_empty_groups_do_not_error():
    rendered = build(PromptConfig(visual_mode=QF), ctx())  # everything empty
    assert rendered.user_text()  # normal-study path, no exception


# 8 -------------------------------------------------------------------------
def test_missing_metadata_does_not_fabricate():
    rendered = build(
        PromptConfig(visual_mode=QF),
        ctx(positive_findings=("Edema",), indication=None, technique=None),
    )
    text = rendered.user_text()
    assert "Indication:" not in text
    assert "Technique:" not in text
    # No fabricated placeholder value for the missing fields.
    assert "unavailable" not in text.lower().replace("comparison is unavailable", "")


# 9 -------------------------------------------------------------------------
def test_pa_only_view():
    rendered = build(PromptConfig(visual_mode=QF), ctx(positive_findings=("Edema",), anchor_view="PA"))
    assert "Views: PA" in rendered.user_text()


# 10 ------------------------------------------------------------------------
def test_ap_plus_lateral_views():
    rendered = build(
        PromptConfig(visual_mode=QF),
        ctx(positive_findings=("Edema",), anchor_view="AP", auxiliary_views=("lateral",)),
    )
    assert "AP (anchor), lateral (auxiliary)" in rendered.user_text()


# 11 ------------------------------------------------------------------------
def test_no_prior_forbids_temporal_comparison():
    rendered = build(PromptConfig(visual_mode=QF), ctx(positive_findings=("Edema",), prior_available=False))
    text = rendered.user_text()
    assert "do not state or imply that a finding is new, improved, worsened, stable or unchanged" in text
    assert "Prior comparison available: no" in text


# 12 ------------------------------------------------------------------------
def test_prior_available_is_not_hardcoded_unavailable():
    rendered = build(
        PromptConfig(visual_mode=QF),
        ctx(positive_findings=("Edema",), prior_available=True, comparison_available=True),
    )
    text = rendered.user_text()
    assert "Prior comparison available: yes" in text
    assert "If prior comparison is unavailable" not in text


# 13 ------------------------------------------------------------------------
def test_train_and_inference_prompt_are_identical():
    builder = PromptBuilder(PromptConfig(visual_mode=QF))
    context = ctx(positive_findings=("Edema",))
    train_parts = builder.build_user_messages(context)
    inference_parts = builder.build_user_messages(context)
    assert train_parts == inference_parts


# 14 ------------------------------------------------------------------------
def test_assistant_target_never_appears_in_user_prompt():
    target = "There is a large right pleural effusion with adjacent atelectasis."
    rendered = build(PromptConfig(visual_mode=QF), ctx(positive_findings=("Pleural Effusion",)))
    # The builder has no channel to receive the target; assert it is absent.
    assert target not in rendered.user_text()


# 15 ------------------------------------------------------------------------
def test_label_masking_masks_prompt_prefix_only():
    prompt_ids = [5, 6, 7]
    full_ids = [5, 6, 7, 10, 11, 2]
    labels = masked_label_ids(full_ids, prompt_ids)
    assert labels[:3] == [-100, -100, -100]
    assert labels[3:] == [10, 11, 2]


# 16 ------------------------------------------------------------------------
def test_qformer_token_count_is_exact():
    rendered = build(PromptConfig(visual_mode=QF), ctx(qformer_token_count=32))
    soft = [p for p in rendered.parts if p.kind is PartKind.SOFT_TOKENS]
    assert len(soft) == 1 and soft[0].count == 32
    assert rendered.user_text().count("<qformer_soft_token>") == 32


# 17 ------------------------------------------------------------------------
def test_soft_token_string_is_a_single_repeated_token():
    rendered = build(PromptConfig(visual_mode=QF), ctx(qformer_token_count=4))
    text = rendered.user_text(soft_token="<qformer_soft_token>")
    tokens = [t for t in text.split() if t == "<qformer_soft_token>"]
    assert len(tokens) == 4  # exactly count, none merged/partial


def test_qformer_mode_requires_positive_token_count():
    with pytest.raises(PromptConfigError, match="qformer_token_count"):
        build(PromptConfig(visual_mode=QF), PromptContext(study_id="s", visual_mode=QF))


# 18 ------------------------------------------------------------------------
def test_soft_token_positions_are_masked_to_minus_100():
    # Simulate a chat prefix ending in 4 soft-token ids; the target follows.
    soft_id = 99
    prompt_ids = [1, 2, soft_id, soft_id, soft_id, soft_id]
    full_ids = prompt_ids + [40, 41, 2]
    labels = masked_label_ids(full_ids, prompt_ids)
    assert all(label == -100 for label in labels[:6])
    assert labels[6:] == [40, 41, 2]


# 19 ------------------------------------------------------------------------
def test_report_target_is_not_fully_masked():
    labels = masked_label_ids([1, 2, 3, 40], [1, 2, 3])
    assert any(label != -100 for label in labels)


# 20 ------------------------------------------------------------------------
def test_invalid_config_value_raises():
    with pytest.raises(PromptConfigError, match="negative_policy"):
        config_from_mapping({"visual_mode": "qformer_guided", "negative_policy": "bogus"})


def test_unknown_config_key_raises():
    with pytest.raises(PromptConfigError, match="unknown prompt config key"):
        config_from_mapping({"visual_mode": "qformer_guided", "made_up": 1})


# 21 ------------------------------------------------------------------------
def test_prompt_hash_is_stable_and_config_sensitive():
    a = build(PromptConfig(visual_mode=QF, max_negative_findings=4), ctx(positive_findings=("Edema",)))
    b = build(PromptConfig(visual_mode=QF, max_negative_findings=4), ctx(positive_findings=("Edema",)))
    c = build(PromptConfig(visual_mode=QF, max_negative_findings=2), ctx(positive_findings=("Edema",)))
    assert a.prompt_hash == b.prompt_hash
    assert a.config_hash != c.config_hash and a.prompt_hash != c.prompt_hash


# 22 ------------------------------------------------------------------------
def test_special_token_is_suppressed_at_generation():
    assert soft_token_bad_words_ids(99) == [[99]]
    assert soft_token_bad_words_ids(None) is None  # native path adds no suppression


# 23 ------------------------------------------------------------------------
def test_negatives_are_truncated_before_metadata_and_positives():
    rendered = build(
        PromptConfig(visual_mode=QF, negative_policy=NegativePolicy.ALL),
        ctx(
            positive_findings=("Cardiomegaly",),
            negative_findings=("Pneumothorax", "Edema", "Fracture"),
            anchor_view="PA",
        ),
    )
    words = lambda s: len(s.split())
    # Count exactly as fit_to_budget does (soft-token parts contribute their count).
    part_cost = lambda p: words(p.text) if p.text else (p.count or 0)
    total = sum(part_cost(p) for p in rendered.parts)
    # Budget that forces dropping the absent line but keeps context + present.
    absent_words = part_cost(
        next(p for p in rendered.parts if p.text and p.text.startswith("- Clinically relevant absent"))
    )
    fitted = fit_to_budget(rendered, words, total - absent_words)
    texts = " ".join(p.text for p in fitted.parts if p.text)
    assert "Clinically relevant absent" not in texts
    assert "Present: Cardiomegaly" in texts
    assert "Views: PA" in texts  # metadata kept; negatives cut first


# 24 ------------------------------------------------------------------------
def test_normal_summary_still_declares_visual_primary():
    rendered = build(
        PromptConfig(visual_mode=QF, normal_policy=NormalPolicy.COMPACT_SUMMARY, visual_primary=True),
        ctx(negative_findings=("Pneumothorax",)),
    )
    assert "primary evidence" in rendered.user_text()


# 25 ------------------------------------------------------------------------
def test_native_anchor_only_has_no_stage1_labels():
    rendered = build(
        PromptConfig(visual_mode=VisualMode.NATIVE_ANCHOR_ONLY),
        ctx(mode=VisualMode.NATIVE_ANCHOR_ONLY, positive_findings=("Edema",)),
    )
    text = rendered.user_text()
    assert "Auxiliary Stage-1 predictions" not in text
    assert "Edema" not in text
    assert any(p.kind is PartKind.IMAGE for p in rendered.parts)


# 26 ------------------------------------------------------------------------
def test_qformer_visual_only_has_no_stage1_labels():
    rendered = build(
        PromptConfig(visual_mode=VisualMode.QFORMER_VISUAL_ONLY),
        ctx(mode=VisualMode.QFORMER_VISUAL_ONLY, positive_findings=("Edema",)),
    )
    text = rendered.user_text()
    assert "Auxiliary Stage-1 predictions" not in text
    assert "Edema" not in text
    assert text.count("<qformer_soft_token>") == 32


# 27 ------------------------------------------------------------------------
def test_qformer_guided_has_structured_predictions():
    rendered = build(
        PromptConfig(visual_mode=QF),
        ctx(positive_findings=("Edema",)),
    )
    assert "Auxiliary Stage-1 predictions, which may be imperfect" in rendered.user_text()


# 28 ------------------------------------------------------------------------
def test_temporal_target_policy_variants():
    temporal = "The lungs are clear. Cardiac size is unchanged from the prior study."
    kept, action = apply_temporal_target_policy(temporal, False, TemporalTargetPolicy.KEEP)
    assert kept == temporal and action == "kept"

    stripped, action = apply_temporal_target_policy(
        temporal, False, TemporalTargetPolicy.REMOVE_TEMPORAL_CLAUSES
    )
    assert "unchanged" not in stripped and "The lungs are clear." in stripped
    assert action == "temporal_clauses_removed"

    excluded, action = apply_temporal_target_policy(
        temporal, False, TemporalTargetPolicy.EXCLUDE_SAMPLE
    )
    assert excluded is None and action == "excluded"

    # With a prior available, temporal language is legitimate and kept untouched.
    with_prior, action = apply_temporal_target_policy(
        temporal, True, TemporalTargetPolicy.EXCLUDE_SAMPLE
    )
    assert with_prior == temporal and action == "kept"


# 29 ------------------------------------------------------------------------
def test_unicode_and_newlines_do_not_break_the_prompt():
    rendered = build(
        PromptConfig(visual_mode=QF, include_indication=True),
        ctx(positive_findings=("Edema",), indication="Dyspnée — r/o edema\nfollow-up"),
    )
    text = rendered.user_text()
    assert isinstance(text, str)
    assert "Dyspnée" in text


# 30 ------------------------------------------------------------------------
def test_empty_target_is_handled_explicitly():
    # A fully-truncated target raises in encode_train_example; here the pure guard
    # is that masking an all-prompt sequence leaves nothing to train, which
    # masked_label_ids surfaces as an all -100 vector the caller must reject.
    labels = masked_label_ids([1, 2, 3], [1, 2, 3])
    assert all(label == -100 for label in labels)
    # And the temporal policy tolerates an empty string without raising.
    out, action = apply_temporal_target_policy("", False, TemporalTargetPolicy.REMOVE_TEMPORAL_CLAUSES)
    assert out == "" and action == "kept"


def test_config_round_trips_from_yaml(tmp_path):
    cfg_text = (REPO_ROOT / "configs" / "stage2_prompt_v2.yaml").read_text()
    (tmp_path / "c.yaml").write_text(cfg_text)
    from stage2.prompts import load_prompt_config

    config = load_prompt_config(tmp_path / "c.yaml")
    assert config.visual_mode is VisualMode.QFORMER_GUIDED
    assert config.negative_policy is NegativePolicy.CRITICAL_ONLY
    assert config.uncertainty_policy is UncertaintyPolicy.EXPLICIT_POSSIBLE


# --------------------------------------------------------------------------
# Experiment 2: the two arms must isolate exactly one variable
# --------------------------------------------------------------------------


def _arm_payload(name):
    """The config with comments stripped, so only the settings are compared."""
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "configs" / f"experiment_{name}.yaml"
    return [
        line for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_the_two_experiment_arms_differ_in_one_line():
    """native_anchor_only vs native_anchor_guided is the whole experiment.

    Both arms use MedGemma's own vision tower; the ONLY difference is whether
    structured P/N/U cues from MHCAC classification appear in the prompt. If a
    second setting ever drifts between these files the comparison stops
    isolating that variable, and the result stops meaning what it claims --
    silently, because both runs would still complete and still produce metrics.
    """
    only = _arm_payload("native_anchor_only")
    guided = _arm_payload("native_anchor_guided")
    assert len(only) == len(guided), "the two arms have different numbers of settings"
    differing = [
        (a, b) for a, b in zip(only, guided, strict=True) if a != b
    ]
    assert len(differing) == 1, f"expected exactly one differing line, got {differing}"
    assert "visual_mode" in differing[0][0]
    assert differing[0][0].strip().endswith("native_anchor_only")
    assert differing[0][1].strip().endswith("native_anchor_guided")


def test_neither_arm_uses_a_qformer_visual_mode():
    """Stage 1 ships lambda_itc/itm/lm = 0.0, so soft tokens are untrained."""
    for name in ("native_anchor_only", "native_anchor_guided"):
        assert not any("qformer" in line for line in _arm_payload(name)), name


def test_both_arms_load_through_the_real_config_loader():
    import yaml

    from stage2.prompts import VisualMode

    for name, expected in (
        ("native_anchor_only", VisualMode.NATIVE_ANCHOR_ONLY),
        ("native_anchor_guided", VisualMode.NATIVE_ANCHOR_GUIDED),
    ):
        raw = yaml.safe_load("\n".join(_arm_payload(name)))
        assert VisualMode(raw["prompt"]["visual_mode"]) is expected
        assert expected.image_mode == "native"
