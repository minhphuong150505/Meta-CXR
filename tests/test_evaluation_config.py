"""Evaluation config validation tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.evaluation.config import (  # noqa: E402
    DEFAULT_COMPOSITE_WEIGHTS,
    EvaluationConfig,
    EvaluationConfigError,
    composite_score,
)


def test_defaults_validate():
    config = EvaluationConfig().validate()
    assert config.uncertain_policy == "three_class"
    assert config.selection_metric == "f1_positive_macro"


def test_from_dict_parses_the_documented_block():
    config = EvaluationConfig.from_dict(
        {
            "uncertain_policy": "ignore_uncertain",
            "selection_metric": "positive_macro_f1",
            "threshold_mode": "calibrated",
            "threshold_objective": "f1",
            "bootstrap": {"enabled": True, "samples": 1000, "confidence": 0.95, "seed": 42},
            "classification_metrics": ["accuracy", "positive_macro_f1", "macro_auprc"],
            "generation_metrics": ["bleu", "rouge"],
            "clinical_metrics": ["chexbert", "radgraph"],
            "save_predictions": True,
        }
    ).validate()
    assert config.threshold_mode == "calibrated"
    assert config.bootstrap.samples == 1000
    assert config.clinical_metrics == ("chexbert", "radgraph")


def test_unknown_policy_is_rejected():
    with pytest.raises(EvaluationConfigError, match="uncertain_policy"):
        EvaluationConfig(uncertain_policy="uncertain_as_maybe").validate()


def test_unknown_selection_metric_is_rejected_before_training():
    with pytest.raises(EvaluationConfigError, match="nothing to select on"):
        EvaluationConfig(selection_metric="radgraph_f1").validate(stage="stage1")


def test_stage2_accepts_its_own_selection_metrics():
    EvaluationConfig(selection_metric="rouge_l").validate(stage="stage2")
    with pytest.raises(EvaluationConfigError):
        EvaluationConfig(selection_metric="rouge_l").validate(stage="stage1")


def test_unknown_metric_names_are_rejected():
    with pytest.raises(EvaluationConfigError, match="classification_metrics"):
        EvaluationConfig(classification_metrics=("f1_but_wrong",)).validate()
    with pytest.raises(EvaluationConfigError, match="generation_metrics"):
        EvaluationConfig(generation_metrics=("blue",)).validate()
    with pytest.raises(EvaluationConfigError, match="clinical_metrics"):
        EvaluationConfig(clinical_metrics=("radcliq",)).validate()


def test_unknown_config_key_is_rejected():
    with pytest.raises(EvaluationConfigError, match="unknown key"):
        EvaluationConfig.from_dict({"uncertian_policy": "three_class"})


def test_invalid_bootstrap_settings_are_rejected():
    with pytest.raises(EvaluationConfigError, match="samples"):
        EvaluationConfig.from_dict({"bootstrap": {"samples": -1}}).validate()
    with pytest.raises(EvaluationConfigError, match="confidence"):
        EvaluationConfig.from_dict({"bootstrap": {"confidence": 1.5}}).validate()


def test_composite_weights_must_sum_to_one():
    with pytest.raises(EvaluationConfigError, match="sum to 1"):
        EvaluationConfig(
            selection_metric="composite",
            composite_weights={"rouge_l": 0.5, "bertscore_f1": 0.2},
        ).validate(stage="stage2")


def test_composite_score_is_the_documented_weighted_sum():
    metrics = {
        "radgraph_f1": 0.5,
        "chexbert_macro_f1": 0.4,
        "rouge_l": 0.3,
        "bertscore_f1": 0.2,
    }
    # 0.4*0.5 + 0.2*0.4 + 0.2*0.3 + 0.2*0.2 = 0.2 + 0.08 + 0.06 + 0.04
    assert composite_score(metrics) == pytest.approx(0.38)
    assert sum(DEFAULT_COMPOSITE_WEIGHTS.values()) == pytest.approx(1.0)


def test_composite_score_refuses_to_treat_a_missing_term_as_zero():
    with pytest.raises(EvaluationConfigError, match="not treated as zero"):
        composite_score({"rouge_l": 0.3, "bertscore_f1": 0.2})
