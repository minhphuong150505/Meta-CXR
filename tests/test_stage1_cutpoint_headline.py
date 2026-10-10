"""Validation-fitted per-finding cutpoints: the project's headline Stage-1 rule.

Since 2026-10-10 (user decision) the reported Stage-1 numbers decide each
finding by two cutpoints on ``p_pos / (p_pos + p_neg)`` fitted on validation,
with argmax reported beside them. No model, GPU or dataset is needed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import calibrate_thresholds as calibrate_cli  # noqa: E402
from scripts import evaluate_stage1 as stage1_cli  # noqa: E402
from tests.test_evaluation_integration import synthetic_predictions  # noqa: E402
from training.evaluation.classification_metrics import evaluate_classification  # noqa: E402
from training.evaluation.threshold_calibration import (  # noqa: E402
    CUTPOINT_FORMAT,
    CalibrationError,
    CutpointFile,
    ThresholdFile,
    apply_cutpoints,
    load_cutpoint_file,
    load_cutpoints,
)

SHIPPED = _REPO_ROOT / "configs" / "stage1_cutpoints" / "run_20261005_paper.json"


def _fit(tmp_path: Path, *extra: str):
    val = synthetic_predictions(400, seed=1, split="validation")
    test = synthetic_predictions(300, seed=2, split="test")
    val_path = val.save(tmp_path / "val.npz")
    test_path = test.save(tmp_path / "test.npz")
    cut_path = tmp_path / "cutpoints.json"
    assert calibrate_cli.main([
        "--rule", "cutpoints", "--predictions", str(val_path),
        "--output", str(cut_path), *extra,
    ]) == 0
    return val_path, test_path, cut_path, test


def test_cutpoint_file_round_trips(tmp_path):
    cut = {"Edema": (0.3, 0.6), "Fracture": (0.5, 0.5)}
    path = CutpointFile(cutpoints=cut, metadata={"split": "val"}).save(tmp_path / "c.json")
    loaded, meta = load_cutpoint_file(path)
    assert loaded == cut and meta["split"] == "val"
    assert json.loads(path.read_text())["format"] == CUTPOINT_FORMAT


def test_an_eq22_threshold_file_is_not_read_as_cutpoints(tmp_path):
    path = ThresholdFile(thresholds={"Edema": {"negative": 0.5, "positive": 0.5,
                                               "uncertain": 0.5}},
                         metadata={"split": "val"}).save(tmp_path / "t.json")
    with pytest.raises(CalibrationError):
        load_cutpoints(path)


def test_inverted_cutpoints_are_refused(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"format": CUTPOINT_FORMAT,
                                "cutpoints": {"Edema": [0.7, 0.4]}}))
    with pytest.raises(CalibrationError):
        load_cutpoints(path)


def test_evaluate_classification_uses_the_cutpoints(tmp_path):
    _, _, cut_path, test = _fit(tmp_path)
    cut = load_cutpoints(cut_path)
    report = evaluate_classification(test, cutpoints=cut)
    decisions = apply_cutpoints(test.probabilities, cut, tuple(test.pathology_names))
    one_hot = np.eye(3)[decisions]
    expected = evaluate_classification(type(test)(
        labels=test.labels, probabilities=one_hot,
        pathology_names=test.pathology_names, sample_keys=test.sample_keys,
    ))
    assert report.aggregates["weighted_f1"] == pytest.approx(expected.aggregates["weighted_f1"])
    # AUROC never depends on the decision rule.
    assert report.aggregates["auroc_mean"] == pytest.approx(
        evaluate_classification(test).aggregates["auroc_mean"])
    assert "cutpoints" in report.settings["decision"]


def test_thresholds_and_cutpoints_are_exclusive(tmp_path):
    _, _, cut_path, test = _fit(tmp_path)
    with pytest.raises(ValueError):
        evaluate_classification(test, cutpoints=load_cutpoints(cut_path),
                                thresholds={"x": {}})


def test_one_cutpoint_never_calls_uncertain(tmp_path):
    _, _, cut_path, _ = _fit(tmp_path, "--one-cutpoint")
    assert all(t1 == t2 for t1, t2 in load_cutpoints(cut_path).values())


def test_cli_headline_carries_the_argmax_reference(tmp_path):
    _, test_path, cut_path, _ = _fit(tmp_path)
    out = tmp_path / "out"
    assert stage1_cli.main([
        "--predictions", str(test_path), "--cutpoints", str(cut_path),
        "--no-bootstrap", "--no-plots", "--no-baselines", "--output-dir", str(out),
    ]) == 0
    payload = json.loads((out / "metrics.json").read_text())
    assert "cutpoints" in payload["metadata"]["threshold_source"]
    assert "argmax_reference" in payload["classification"]
    assert "weighted_f1" in payload["classification"]["argmax_reference"]["aggregates"]


def test_cli_refuses_cutpoints_fitted_on_the_scored_split(tmp_path):
    val_path, _, cut_path, _ = _fit(tmp_path)
    assert stage1_cli.main([
        "--predictions", str(val_path), "--cutpoints", str(cut_path),
        "--no-bootstrap", "--no-plots", "--output-dir", str(tmp_path / "x"),
    ]) == 2


def test_cli_refuses_thresholds_and_cutpoints_together(tmp_path):
    _, test_path, cut_path, _ = _fit(tmp_path)
    assert stage1_cli.main([
        "--predictions", str(test_path), "--cutpoints", str(cut_path),
        "--thresholds", str(cut_path), "--output-dir", str(tmp_path / "x"),
    ]) == 2


@pytest.mark.skipif(not SHIPPED.is_file(), reason="shipped cutpoint file not written yet")
def test_shipped_cutpoints_cover_14_findings_and_come_from_val():
    cut, meta = load_cutpoint_file(SHIPPED)
    assert len(cut) == 14
    assert meta["split"] in ("val", "validation")
    assert all(0.0 <= t1 <= t2 <= 1.0 for t1, t2 in cut.values())
