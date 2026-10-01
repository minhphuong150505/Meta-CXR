"""Per-class thresholds of the META-CXR paper (Sec. V-C, Eq. 22, Fig. 11)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.evaluation.schemas import ClassificationPredictions  # noqa: E402
from training.evaluation.threshold_calibration import (  # noqa: E402
    ABSTAIN,
    CalibrationError,
    ThresholdFile,
    apply_thresholds,
    decide_with_thresholds,
    fit_class_thresholds,
    load_thresholds,
    roc_distance_threshold,
)


def test_roc_distance_picks_the_point_nearest_the_top_left_corner():
    # threshold 0.6 separates perfectly: TPR 1, FPR 0, distance 0.
    scores = np.array([0.9, 0.6, 0.4, 0.1])
    y = np.array([1, 1, 0, 0])
    assert roc_distance_threshold(scores, y) == pytest.approx(0.6)


def test_roc_distance_hand_computed_with_overlap():
    # candidates (TPR, FPR): 0.9->(1/2,0) d=.5 ; 0.7->(1/2,1/2) d=.707 ;
    # 0.5->(1,1/2) d=.5 ; 0.2->(1,1) d=1. Tie between 0.9 and 0.5 -> first (0.9).
    scores = np.array([0.9, 0.7, 0.5, 0.2])
    y = np.array([1, 0, 1, 0])
    assert roc_distance_threshold(scores, y) == pytest.approx(0.9)


def test_undefined_when_a_class_never_occurs():
    assert math.isnan(roc_distance_threshold(np.array([0.2, 0.8]), np.array([0, 0])))


def _preds():
    labels = np.array([[0], [0], [1], [1], [2], [2]])
    probs = np.array(
        [[[0.8, 0.1, 0.1]], [[0.7, 0.2, 0.1]], [[0.1, 0.8, 0.1]],
         [[0.2, 0.7, 0.1]], [[0.1, 0.1, 0.8]], [[0.1, 0.2, 0.7]]]
    )
    return ClassificationPredictions(
        labels=labels, probabilities=probs, pathology_names=("Edema",),
        sample_keys=np.arange(6).astype(str),
    )


def test_fit_gives_one_threshold_per_class_for_every_finding():
    thresholds = fit_class_thresholds(_preds())
    assert set(thresholds["Edema"]) == {"negative", "positive", "uncertain"}
    assert thresholds["Edema"]["positive"] == pytest.approx(0.7)


def test_apply_thresholds_takes_largest_margin_and_abstains_below_all():
    thresholds = {"Edema": {"negative": 0.6, "positive": 0.3, "uncertain": 0.9}}
    probs = np.array([
        [[0.65, 0.30, 0.05]],   # neg margin .05, pos margin 0 -> negative
        [[0.20, 0.50, 0.30]],   # only positive clears -> positive
        [[0.50, 0.20, 0.30]],   # nothing clears -> abstain
    ])
    out = apply_thresholds(probs, thresholds, ("Edema",))
    assert out[:, 0].tolist() == [0, 1, ABSTAIN]


def test_decide_with_thresholds_falls_back_to_argmax_instead_of_abstaining():
    thresholds = {"Edema": {"negative": 0.6, "positive": 0.3, "uncertain": 0.9}}
    probs = np.array([
        [[0.65, 0.30, 0.05]],   # as apply_thresholds -> negative
        [[0.20, 0.50, 0.30]],   # as apply_thresholds -> positive
        [[0.50, 0.20, 0.30]],   # nothing clears -> argmax -> negative
    ])
    assert decide_with_thresholds(probs, thresholds, ("Edema",))[:, 0].tolist() == [0, 1, 0]


def test_thresholds_change_the_decision_but_not_the_auroc():
    from training.evaluation.classification_metrics import evaluate_classification

    # Uncertain never wins argmax, but clears a low uncertain threshold.
    probs = np.array([[[0.50, 0.30, 0.20]], [[0.50, 0.30, 0.20]], [[0.6, 0.35, 0.05]]])
    preds = ClassificationPredictions(
        labels=np.array([[2], [2], [0]]),
        probabilities=probs,
        pathology_names=("Edema",),
        sample_keys=np.array(["a", "b", "c"]),
    )
    thresholds = {"Edema": {"negative": 0.55, "positive": 0.9, "uncertain": 0.1}}
    argmax = evaluate_classification(preds)
    tuned = evaluate_classification(preds, thresholds=thresholds)
    assert argmax.pathology("Edema").per_class["uncertain"]["recall"] == 0.0
    assert tuned.pathology("Edema").per_class["uncertain"]["recall"] == 1.0
    assert tuned.aggregates["auroc_mean"] == argmax.aggregates["auroc_mean"]
    assert "NOT the paper" in tuned.settings["decision"]
    assert argmax.settings["decision"] == "argmax"


def test_round_trip_and_format_guard(tmp_path):
    path = ThresholdFile(fit_class_thresholds(_preds()), {"split": "val"}).save(tmp_path / "t.json")
    loaded = load_thresholds(path)
    assert loaded["Edema"]["uncertain"] == pytest.approx(0.7)

    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({"thresholds": {"Edema": 0.4}}))
    with pytest.raises(CalibrationError, match="per-class"):
        load_thresholds(legacy)


def test_cutpoints_decide_negative_uncertain_positive_by_severity():
    from training.evaluation.threshold_calibration import decide_with_cutpoints, severity_scores

    probs = np.array([[0.8, 0.2, 0.0], [0.5, 0.5, 0.0], [0.1, 0.9, 0.0]])
    s = severity_scores(probs)
    assert s.tolist() == pytest.approx([0.2, 0.5, 0.9])
    assert decide_with_cutpoints(s, 0.4, 0.7).tolist() == [0, 2, 1]
    assert decide_with_cutpoints(s, 0.4, 0.4).tolist() == [0, 1, 1]   # one cutpoint: no U
    with pytest.raises(CalibrationError):
        decide_with_cutpoints(s, 0.7, 0.4)


def _ordinal_preds():
    # Severity rises N < U < P; a middle band recovers Uncertain.
    sev = np.r_[np.linspace(0.0, 0.3, 40), np.linspace(0.4, 0.6, 20), np.linspace(0.7, 1.0, 40)]
    labels = np.r_[np.zeros(40), np.full(20, 2), np.ones(40)].astype(int)[:, None]
    probs = np.stack([1 - sev, sev, np.zeros_like(sev)], axis=-1)[:, None, :] * 0.9
    probs[..., 2] = 0.1
    return ClassificationPredictions(
        labels=labels, probabilities=probs, pathology_names=("Edema",),
        sample_keys=np.array([f"s{i}" for i in range(100)]), metadata={"split": "val"},
    )


def test_two_cutpoints_recover_a_middle_uncertain_band():
    from training.evaluation.threshold_calibration import apply_cutpoints, fit_severity_cutpoints

    preds = _ordinal_preds()
    cut = fit_severity_cutpoints(preds, objective="weighted_f1")
    t1, t2 = cut["Edema"]
    assert 0.3 < t1 <= 0.4 and 0.6 < t2 <= 0.7
    assert (apply_cutpoints(preds.probabilities, cut, ("Edema",))[:, 0] == preds.labels[:, 0]).all()
    one = fit_severity_cutpoints(preds, objective="weighted_f1", allow_uncertain=False)
    assert one["Edema"][0] == one["Edema"][1]


def test_cutpoint_script_refuses_to_fit_on_test(tmp_path):
    from scripts import evaluate_stage1_cutpoints as cli

    preds = _ordinal_preds()
    test_like = ClassificationPredictions(
        labels=preds.labels, probabilities=preds.probabilities, pathology_names=preds.pathology_names,
        sample_keys=preds.sample_keys, metadata={"split": "test"},
    )
    path = test_like.save(tmp_path / "t.npz")
    assert cli.main(["--val", str(path), "--test", str(path), "--output-dir", str(tmp_path / "o")]) == 2


def test_cutpoint_script_runs_end_to_end(tmp_path):
    import json as _json

    from scripts import evaluate_stage1_cutpoints as cli

    val = _ordinal_preds().save(tmp_path / "v.npz")
    test = _ordinal_preds().save(tmp_path / "te.npz")
    out = tmp_path / "o"
    assert cli.main(["--val", str(val), "--test", str(test), "--output-dir", str(out),
                     "--bootstrap-samples", "20"]) == 0
    report = _json.loads((out / "cutpoints_report.json").read_text())
    assert "not the paper" in report["note"]
    assert report["rules"]["two cutpoints"]["metrics"]["weighted_f1"] == pytest.approx(1.0)
    assert report["rules"]["one cutpoint"]["uncertain_predicted"] == 0
