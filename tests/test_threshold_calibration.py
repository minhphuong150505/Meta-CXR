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


def test_round_trip_and_format_guard(tmp_path):
    path = ThresholdFile(fit_class_thresholds(_preds()), {"split": "val"}).save(tmp_path / "t.json")
    loaded = load_thresholds(path)
    assert loaded["Edema"]["uncertain"] == pytest.approx(0.7)

    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({"thresholds": {"Edema": 0.4}}))
    with pytest.raises(CalibrationError, match="per-class"):
        load_thresholds(legacy)
