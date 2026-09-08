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
            ctx(), logits_all_positive(), shut, use_mention_gate=True
        )
        assert sum(len(v) for v in groups.values()) == 0

    def test_an_open_gate_reproduces_the_ungated_answer(self):
        wide = torch.full((N,), 20.0)
        gated = fig9.classify_with_thresholds(
            ctx(), logits_all_positive(), wide, use_mention_gate=True
        )
        assert gated == fig9.classify_with_thresholds(ctx(), logits_all_positive())

    def test_the_gate_is_per_label(self):
        mention = torch.full((N,), -20.0)
        idx = next(i for i, a in enumerate(fig9.ABNORMALITIES_14) if a != "No Finding")
        mention[idx] = 20.0
        groups = fig9.classify_with_thresholds(
            ctx(), logits_all_positive(), mention, use_mention_gate=True
        )
        assert groups["positive"] == [fig9.ABNORMALITIES_14[idx]]

    def test_no_finding_is_never_emitted_even_with_its_gate_open(self):
        if "No Finding" not in fig9.ABNORMALITIES_14:
            pytest.skip("No Finding not in the label set")
        mention = torch.full((N,), 20.0)
        groups = fig9.classify_with_thresholds(
            ctx(), logits_all_positive(), mention, use_mention_gate=True
        )
        assert "No Finding" not in groups["positive"]

    def test_a_per_label_threshold_from_the_json_is_honoured(self):
        """m is uncalibrated, so the threshold must come from validation."""
        name = next(a for a in fig9.ABNORMALITIES_14 if a != "No Finding")
        mention = torch.full((N,), -20.0)
        mention[fig9.ABNORMALITIES_14.index(name)] = 0.0  # sigmoid = 0.5 exactly
        loose = fig9.classify_with_thresholds(
            ctx({name: {fig9.MENTION_THRESHOLD_KEY: 0.4}}),
            logits_all_positive(), mention, use_mention_gate=True)
        strict = fig9.classify_with_thresholds(
            ctx({name: {fig9.MENTION_THRESHOLD_KEY: 0.6}}),
            logits_all_positive(), mention, use_mention_gate=True)
        assert name in loose["positive"]
        assert sum(len(v) for v in strict.values()) == 0


class TestItRefusesToGuess:
    def test_gate_without_mention_logits_raises(self):
        with pytest.raises(ValueError, match="needs mention_logits"):
            fig9.classify_with_thresholds(
                ctx(), logits_all_positive(), None, use_mention_gate=True
            )

    def test_wrong_length_mention_logits_raises(self):
        with pytest.raises(ValueError, match="expected"):
            fig9.classify_with_thresholds(
                ctx(), logits_all_positive(), torch.zeros(3), use_mention_gate=True
            )


def test_the_cue_rule_is_part_of_the_cache_identity():
    """Otherwise a gated run silently reuses q-only pred_groups from cache."""
    import inspect

    params = inspect.signature(fig9.stage1_cohort_fingerprint).parameters
    assert "use_mention_gate" in params
    params = inspect.signature(fig9.build_stage1_records).parameters
    assert "use_mention_gate" in params
