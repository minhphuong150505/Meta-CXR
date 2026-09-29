"""Stage-1 classification metrics: the META-CXR paper's three-class protocol.

Expected values are hand-computed in the test bodies. Where sklearn is
installed (the training host) the weighted P/R/F1 are also checked against
``sklearn.metrics`` with ``average='weighted', zero_division=1`` -- the exact
call in the reference code (META-CXR/mhcac/utils.py).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.evaluation.classification_metrics import (  # noqa: E402
    AGGREGATE_METRICS,
    PAPER_FIVE_FINDINGS,
    average_precision,
    confusion_matrix,
    decide,
    evaluate_classification,
    roc_auc,
    weighted_prf,
)
from training.evaluation.schemas import (  # noqa: E402
    ClassificationPredictions,
    SchemaError,
)


def _one_hot(classes, confidence=0.8):
    classes = np.asarray(classes)
    probs = np.full(classes.shape + (3,), (1 - confidence) / 2)
    np.put_along_axis(probs, classes[..., None], confidence, axis=-1)
    return probs


def make(labels, predicted, names=None):
    labels = np.asarray(labels)
    names = names or tuple(f"P{i}" for i in range(labels.shape[1]))
    return ClassificationPredictions(
        labels=labels,
        probabilities=_one_hot(predicted),
        pathology_names=names,
        sample_keys=np.asarray([f"s{i}" for i in range(labels.shape[0])]),
    )


# ---------------------------------------------------------------------------
# ROC / AP primitives
# ---------------------------------------------------------------------------


def test_roc_auc_perfect_and_inverted():
    assert roc_auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert roc_auc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0


def test_roc_auc_ties_count_half():
    assert roc_auc([0.5, 0.5], [0, 1]) == 0.5


def test_roc_auc_undefined_with_one_class():
    assert math.isnan(roc_auc([0.1, 0.9], [1, 1]))


def test_average_precision_hand_computed():
    # ranks: 0.9(+) 0.8(-) 0.7(+): AP = 1*0.5 + (2/3)*0.5
    assert average_precision([0.9, 0.8, 0.7], [1, 0, 1]) == pytest.approx(0.5 + 1 / 3)


# ---------------------------------------------------------------------------
# weighted P/R/F1 -- the paper's headline
# ---------------------------------------------------------------------------


def test_weighted_prf_hand_computed():
    y_true = np.array([0, 0, 0, 1, 1, 2])
    y_pred = np.array([0, 0, 1, 1, 0, 2])
    # class 0: tp2 pred3 sup3 -> P 2/3 R 2/3 F1 2/3
    # class 1: tp1 pred2 sup2 -> P 1/2 R 1/2 F1 1/2
    # class 2: tp1 pred1 sup1 -> P 1   R 1   F1 1
    out = weighted_prf(y_true, y_pred)
    assert out["precision"] == pytest.approx((3 * 2 / 3 + 2 * 0.5 + 1) / 6)
    assert out["recall"] == pytest.approx((3 * 2 / 3 + 2 * 0.5 + 1) / 6)
    assert out["f1"] == pytest.approx((3 * 2 / 3 + 2 * 0.5 + 1) / 6)
    assert out["accuracy"] == pytest.approx(4 / 6)


def test_never_predicted_class_gets_precision_one_like_sklearn_zero_division_1():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 0])
    out = weighted_prf(y_true, y_pred)
    # class 0: P 2/4, R 1, F1 2*2/(2+4)=2/3 ; class 1: P 1 (zero_division), R 0, F1 0
    assert out["precision"] == pytest.approx((2 * 0.5 + 2 * 1.0) / 4)
    assert out["recall"] == pytest.approx((2 * 1.0 + 2 * 0.0) / 4)
    assert out["f1"] == pytest.approx((2 * (2 / 3) + 0) / 4)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_weighted_prf_matches_sklearn(seed):
    metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(seed)
    y_true = rng.choice(3, size=200, p=[0.7, 0.2, 0.1])
    y_pred = rng.choice(3, size=200, p=[0.5, 0.45, 0.05])
    ours = weighted_prf(y_true, y_pred)
    for key, fn in (
        ("precision", metrics.precision_score),
        ("recall", metrics.recall_score),
        ("f1", metrics.f1_score),
    ):
        expected = fn(y_true, y_pred, average="weighted", zero_division=1)
        assert ours[key] == pytest.approx(expected), key


def test_confusion_matrix_rows_are_truth():
    m = confusion_matrix([0, 1, 2, 2], [0, 2, 2, 1])
    assert m.tolist() == [[1, 0, 0], [0, 0, 1], [0, 1, 1]]


def test_decision_is_argmax_over_three_classes():
    probs = np.array([[[0.2, 0.3, 0.5], [0.6, 0.3, 0.1]]])
    assert decide(probs).tolist() == [[2, 0]]


# ---------------------------------------------------------------------------
# evaluate_classification
# ---------------------------------------------------------------------------


def test_aggregates_are_equal_weight_means_over_findings():
    labels = [[0, 1], [0, 1], [1, 1], [2, 0]]
    predicted = [[0, 1], [0, 1], [1, 1], [2, 0]]
    report = evaluate_classification(make(labels, predicted))
    assert report.aggregates["weighted_f1"] == pytest.approx(1.0)
    assert report.aggregates["accuracy"] == pytest.approx(1.0)
    per = [m.weighted_f1 for m in report.per_pathology]
    assert per == [1.0, 1.0]


def test_uncertain_is_its_own_class_not_folded():
    """Predicting Positive for an Uncertain cell is an error, not a hit."""
    labels = [[2], [2], [0], [1]]
    as_positive = evaluate_classification(make(labels, [[1], [1], [0], [1]]))
    correct = evaluate_classification(make(labels, [[2], [2], [0], [1]]))
    assert correct.aggregates["weighted_f1"] == pytest.approx(1.0)
    assert as_positive.aggregates["weighted_f1"] < 0.7
    assert as_positive.pathology("P0").support == {"negative": 1, "positive": 1, "uncertain": 2}


def test_missing_cells_are_skipped_per_finding():
    labels = [[0, -1], [1, 1], [1, 0]]
    report = evaluate_classification(make(labels, [[0, 2], [1, 1], [1, 0]]))
    assert report.pathology("P1").support == {"negative": 1, "positive": 1, "uncertain": 0}
    assert report.pathology("P1").weighted_f1 == pytest.approx(1.0)


def test_per_class_one_vs_rest_auroc_like_paper_fig5():
    labels = np.array([[0], [0], [1], [1], [2], [2]])
    probs = np.array(
        [
            [[0.8, 0.1, 0.1]],
            [[0.7, 0.2, 0.1]],
            [[0.1, 0.8, 0.1]],
            [[0.2, 0.7, 0.1]],
            [[0.1, 0.1, 0.8]],
            [[0.1, 0.2, 0.7]],
        ]
    )
    preds = ClassificationPredictions(
        labels=labels, probabilities=probs, pathology_names=("P0",),
        sample_keys=np.arange(6).astype(str),
    )
    report = evaluate_classification(preds)
    assert report.pathology("P0").auroc == {"negative": 1.0, "positive": 1.0, "uncertain": 1.0}
    assert report.aggregates["auroc_positive_mean"] == 1.0


def test_five_finding_mean_uses_the_papers_five():
    names = tuple(["No Finding", *PAPER_FIVE_FINDINGS])
    labels = np.zeros((4, 6), dtype=int)
    labels[:2, :] = 1
    predicted = labels.copy()
    predicted[:, 0] = 0  # get No Finding wrong
    report = evaluate_classification(make(labels, predicted, names))
    assert report.aggregates["mean_weighted_f1_5"] == pytest.approx(1.0)
    assert report.aggregates["weighted_f1"] < 1.0


def test_every_advertised_aggregate_is_computed():
    report = evaluate_classification(make([[0], [1], [2]], [[0], [1], [2]]))
    assert set(AGGREGATE_METRICS) <= set(report.aggregates)


def test_there_is_no_binary_or_positive_only_aggregate():
    report = evaluate_classification(make([[0], [1], [2]], [[0], [1], [2]]))
    for key in report.aggregates:
        assert "positive_macro" not in key and "presence" not in key and "binary" not in key


def test_two_class_probabilities_are_refused():
    with pytest.raises(SchemaError):
        evaluate_classification(
            ClassificationPredictions(
                labels=np.array([[0], [1]]),
                probabilities=np.array([[[0.9, 0.1]], [[0.2, 0.8]]]),
                pathology_names=("P0",),
                sample_keys=np.array(["a", "b"]),
            )
        )


def test_save_and_load_round_trip(tmp_path):
    preds = make([[0, 1], [1, 2]], [[0, 1], [1, 2]])
    path = preds.save(tmp_path / "p.npz")
    loaded = ClassificationPredictions.load(path)
    got = evaluate_classification(loaded).aggregates
    want = evaluate_classification(preds).aggregates
    assert got.keys() == want.keys()
    for key in want:
        assert (math.isnan(got[key]) and math.isnan(want[key])) or got[key] == want[key], key
