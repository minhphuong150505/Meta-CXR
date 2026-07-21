"""Clinical metric adapters must be honest about what is not installed.

No clinical metric is implemented in this repo. These tests pin the behaviour
that matters: a missing dependency produces a named error, never a number, and
never a lexical score wearing a clinical name.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from training.evaluation.clinical import (  # noqa: E402
    ERROR,
    REGISTRY,
    SKIP,
    WARN,
    CheXbertMetric,
    ClinicalMetric,
    MissingOptionalDependency,
    RadGraphMetric,
    build_metric,
    resolve_metrics,
    validate_selection_metric,
)

PREDS = ["clear lungs", "small effusion"]
REFS = ["lungs are clear", "there is a small pleural effusion"]


def test_no_clinical_dependency_is_imported_at_module_import():
    """Importing this module must not require radgraph/chexbert to exist."""
    for package in ("radgraph", "chexbert", "chexpert_labeler"):
        assert package not in sys.modules


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_every_registered_metric_satisfies_the_protocol(name):
    metric = build_metric(name)
    assert isinstance(metric, ClinicalMetric)
    assert metric.name == name


def test_unknown_metric_is_rejected():
    with pytest.raises(ValueError, match="unknown clinical metric"):
        build_metric("radcliq")


def test_radcliq_and_radgraph_are_not_silently_aliased():
    """CLAUDE.md: this repo has no RadGraph or RadCliq implementation."""
    assert "radcliq" not in REGISTRY


def test_missing_dependency_raises_and_names_the_install_command():
    metric = RadGraphMetric()
    assert metric.available() is False
    with pytest.raises(MissingOptionalDependency) as excinfo:
        metric.compute(PREDS, REFS)

    message = str(excinfo.value)
    assert "radgraph" in message
    assert "pip install radgraph" in message
    # The critical promise: no lexical stand-in.
    assert "no lexical fallback" in message


def test_missing_dependency_returns_no_score_at_all():
    """A caller must not be able to read a number out of the failure."""
    with pytest.raises(MissingOptionalDependency):
        CheXbertMetric().compute(PREDS, REFS)


def test_length_mismatch_is_caught_before_the_dependency_check():
    with pytest.raises(ValueError, match="1 predictions vs 2 references"):
        RadGraphMetric().compute(["one"], REFS)


def test_error_policy_raises_on_a_missing_dependency():
    with pytest.raises(MissingOptionalDependency):
        resolve_metrics(["radgraph"], policy=ERROR)


def test_warn_policy_drops_the_metric_and_explains_why():
    usable, warnings = resolve_metrics(["radgraph", "chexbert"], policy=WARN)
    assert usable == []
    assert len(warnings) == 2
    assert all("not installed" in warning for warning in warnings)


def test_skip_policy_drops_the_metric_silently():
    usable, warnings = resolve_metrics(["radgraph"], policy=SKIP)
    assert usable == []
    assert warnings == []


def test_unknown_policy_is_rejected():
    with pytest.raises(ValueError, match="missing_dependency_policy"):
        resolve_metrics(["radgraph"], policy="ignore")


def test_selection_metric_must_be_available():
    """A run must not start if it cannot compute its own selection criterion."""
    with pytest.raises(ValueError, match="refusing to start"):
        validate_selection_metric("chexbert", available=[])


def test_selection_metric_passes_when_available():
    validate_selection_metric("chexbert", available=["chexbert"])


def test_non_clinical_selection_metric_is_left_alone():
    """BLEU-4 is not this module's business and must not be rejected."""
    validate_selection_metric("BLEU-4", available=[])


def test_installed_but_unwired_adapter_says_so(monkeypatch):
    """If the package appears, the adapter admits it is unwired rather than guessing.

    Fabricating a score from an API that has never been run against reference
    outputs would be worse than the gap.
    """
    metric = RadGraphMetric()
    monkeypatch.setattr(metric, "available", lambda: True)
    monkeypatch.setitem(sys.modules, "radgraph", type(sys)("radgraph"))

    with pytest.raises(NotImplementedError, match="has not been wired"):
        metric.compute(PREDS, REFS)
