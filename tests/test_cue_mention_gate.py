"""Two-stage cue emission: open the mention gate, then read the class off q.

`classify_with_thresholds` is pure arithmetic over two tensors, so it is
testable on a CPU box even though the model that produces those tensors is not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

torch = pytest.importorskip("torch")
fig9 = pytest.importorskip("training.train_eval_figure9_llm_variants_200")
from training.run_context import Stage1Context  # noqa: E402

N = len(fig9.ABNORMALITIES_14)


def ctx(thresholds=None):
    return Stage1Context(run_name="t", thresholds=thresholds or {})


def logits_all_positive():
    """q strongly Positive for every finding. CLASS_MAP order is the truth."""
    out = torch.full((N, 3), -5.0)
    out[:, list(fig9.CLASS_MAP).index("positive")] = 5.0
    return out


class TestDefaultIsUnchanged:
    """Every recorded run sorted all 13 findings on q. That must still happen."""

    def test_without_the_gate_all_reportable_findings_are_emitted(self):
        groups = fig9.classify_with_thresholds(ctx(), logits_all_positive())
        emitted = sum(len(v) for v in groups.values())
        assert emitted == N - 1  # No Finding is excluded
        assert len(groups["positive"]) == N - 1

    def test_mention_logits_are_ignored_when_the_gate_is_off(self):
        shut = torch.full((N,), -20.0)  # sigmoid ~ 0, would gate everything out
        with_m = fig9.classify_with_thresholds(ctx(), logits_all_positive(), shut)
        without = fig9.classify_with_thresholds(ctx(), logits_all_positive())
        assert with_m == without


class TestGatedEmission:
    def test_a_shut_gate_emits_the_finding_in_no_group(self):
        shut = torch.full((N,), -20.0)
        groups = fig9.classify_with_thresholds(
            ctx(), logits_all_positive(), shut, cue_rule=fig9.CUE_RULE_MENTION_GATED
        )
        assert sum(len(v) for v in groups.values()) == 0

    def test_an_open_gate_reproduces_the_ungated_answer(self):
        wide = torch.full((N,), 20.0)
        gated = fig9.classify_with_thresholds(
            ctx(), logits_all_positive(), wide, cue_rule=fig9.CUE_RULE_MENTION_GATED
        )
        assert gated == fig9.classify_with_thresholds(ctx(), logits_all_positive())

    def test_the_gate_is_per_label(self):
        mention = torch.full((N,), -20.0)
        idx = next(i for i, a in enumerate(fig9.ABNORMALITIES_14) if a != "No Finding")
        mention[idx] = 20.0
        groups = fig9.classify_with_thresholds(
            ctx(), logits_all_positive(), mention, cue_rule=fig9.CUE_RULE_MENTION_GATED
        )
        assert groups["positive"] == [fig9.ABNORMALITIES_14[idx]]

    def test_no_finding_is_never_emitted_even_with_its_gate_open(self):
        if "No Finding" not in fig9.ABNORMALITIES_14:
            pytest.skip("No Finding not in the label set")
        mention = torch.full((N,), 20.0)
        groups = fig9.classify_with_thresholds(
            ctx(), logits_all_positive(), mention, cue_rule=fig9.CUE_RULE_MENTION_GATED
        )
        assert "No Finding" not in groups["positive"]

    def test_a_per_label_threshold_from_the_json_is_honoured(self):
        """m is uncalibrated, so the threshold must come from validation."""
        name = next(a for a in fig9.ABNORMALITIES_14 if a != "No Finding")
        mention = torch.full((N,), -20.0)
        mention[fig9.ABNORMALITIES_14.index(name)] = 0.0  # sigmoid = 0.5 exactly
        loose = fig9.classify_with_thresholds(
            ctx({name: {fig9.MENTION_THRESHOLD_KEY: 0.4}}),
            logits_all_positive(), mention, cue_rule=fig9.CUE_RULE_MENTION_GATED)
        strict = fig9.classify_with_thresholds(
            ctx({name: {fig9.MENTION_THRESHOLD_KEY: 0.6}}),
            logits_all_positive(), mention, cue_rule=fig9.CUE_RULE_MENTION_GATED)
        assert name in loose["positive"]
        assert sum(len(v) for v in strict.values()) == 0


class TestItRefusesToGuess:
    def test_gate_without_mention_logits_raises(self):
        with pytest.raises(ValueError, match="needs mention_logits"):
            fig9.classify_with_thresholds(
                ctx(), logits_all_positive(), None, cue_rule=fig9.CUE_RULE_MENTION_GATED
            )

    def test_wrong_length_mention_logits_raises(self):
        with pytest.raises(ValueError, match="expected"):
            fig9.classify_with_thresholds(
                ctx(), logits_all_positive(), torch.zeros(3), cue_rule=fig9.CUE_RULE_MENTION_GATED
            )


def test_the_cue_rule_is_part_of_the_cache_identity():
    """Otherwise a gated run silently reuses q-only pred_groups from cache."""
    import inspect

    params = inspect.signature(fig9.stage1_cohort_fingerprint).parameters
    assert "cue_rule" in params
    params = inspect.signature(fig9.build_stage1_records).parameters
    assert "cue_rule" in params


class TestMarginalPositiveRule:
    """The measured-best rule: threshold sigmoid(m)*q_pos per label."""

    def test_it_emits_only_positives(self):
        mention = torch.full((N,), 20.0)
        groups = fig9.classify_with_thresholds(
            ctx(), logits_all_positive(), mention, cue_rule=fig9.CUE_RULE_MARGINAL
        )
        assert groups["negative"] == [] and groups["uncertain"] == []
        assert len(groups["positive"]) == N - 1

    def test_a_low_marginal_emits_nothing(self):
        """q says Positive, but the gate says the finding is never mentioned."""
        mention = torch.full((N,), -20.0)
        groups = fig9.classify_with_thresholds(
            ctx(), logits_all_positive(), mention, cue_rule=fig9.CUE_RULE_MARGINAL
        )
        assert sum(len(v) for v in groups.values()) == 0

    def test_the_per_label_threshold_is_read_from_the_json(self):
        name = next(a for a in fig9.ABNORMALITIES_14 if a != "No Finding")
        mention = torch.full((N,), -20.0)
        mention[fig9.ABNORMALITIES_14.index(name)] = 0.0  # sigmoid = 0.5
        # q_pos is ~1.0, so the marginal is ~0.5.
        loose = fig9.classify_with_thresholds(
            ctx({name: {fig9.MARGINAL_THRESHOLD_KEY: 0.3}}),
            logits_all_positive(), mention, cue_rule=fig9.CUE_RULE_MARGINAL)
        strict = fig9.classify_with_thresholds(
            ctx({name: {fig9.MARGINAL_THRESHOLD_KEY: 0.7}}),
            logits_all_positive(), mention, cue_rule=fig9.CUE_RULE_MARGINAL)
        assert loose["positive"] == [name]
        assert strict["positive"] == []

    def test_an_unknown_rule_is_refused(self):
        with pytest.raises(ValueError, match="cue_rule must be one of"):
            fig9.classify_with_thresholds(
                ctx(), logits_all_positive(), None, cue_rule="marginal"
            )


def test_load_thresholds_accepts_the_cue_keys():
    """It used to reject any key that was not a P/N/U class name."""
    import json
    import tempfile

    payload = {"Cardiomegaly": {"marginal_positive": 0.62, "mention": 0.6,
                                "positive": 0.5}}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(payload, handle)
        path = handle.name
    loaded = fig9.load_thresholds(path)
    assert loaded["Cardiomegaly"]["marginal_positive"] == pytest.approx(0.62)
    assert loaded["Cardiomegaly"]["mention"] == pytest.approx(0.6)


class TestNoCueRule:
    def test_it_emits_nothing_but_keeps_the_three_keys(self):
        """Present-but-empty is a real prediction; a missing key is refused."""
        groups = fig9.classify_with_thresholds(
            ctx(), logits_all_positive(), None, cue_rule=fig9.CUE_RULE_NONE
        )
        assert set(groups) == {"positive", "negative", "uncertain"}
        assert sum(len(v) for v in groups.values()) == 0

    def test_it_needs_no_mention_logits(self):
        fig9.classify_with_thresholds(
            ctx(), logits_all_positive(), None, cue_rule=fig9.CUE_RULE_NONE
        )


def test_the_shipped_threshold_file_loads_and_covers_every_reportable_label():
    """Fitted on val from run_20260820_ft; see CLAUDE.md for provenance."""
    path = _REPO_ROOT / "configs/stage2_cue_thresholds_marginal_pfit.json"
    loaded = fig9.load_thresholds(path)
    reportable = {a for a in fig9.ABNORMALITIES_14 if a != "No Finding"}
    assert set(loaded) == reportable
    for name, values in loaded.items():
        assert fig9.MARGINAL_THRESHOLD_KEY in values, name
        assert 0.0 < values[fig9.MARGINAL_THRESHOLD_KEY] < 1.0, name
