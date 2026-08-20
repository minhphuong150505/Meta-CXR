"""The label framing decides what the classification metrics mean.

These pin the property that made ``positive_macro_f1`` uninterpretable: under
``masked_polarity`` a blank cell is dropped, so the surviving cells are
positive-heavy and a constant predictor scores well; under ``study_presence``
every cell is answerable and the constant predictor collapses.
"""

from __future__ import annotations

import numpy as np
import pytest

from training.evaluation.label_framing import (
    MASKED_POLARITY,
    STUDY_PRESENCE,
    ScoreUnavailableError,
    UnknownFramingError,
    apply_framing,
    frame_labels,
    presence_scores,
)
from training.evaluation.schemas import (
    MISSING,
    NEGATIVE,
    POSITIVE,
    UNCERTAIN,
    ClassificationPredictions,
)


def _predictions(labels, q_pos, mention=None):
    labels = np.asarray(labels)
    q_pos = np.asarray(q_pos, dtype=float)
    probabilities = np.zeros(labels.shape + (3,), dtype=float)
    probabilities[..., POSITIVE] = q_pos
    probabilities[..., NEGATIVE] = 1.0 - q_pos
    return ClassificationPredictions(
        labels=labels,
        probabilities=probabilities,
        pathology_names=tuple(f"p{i}" for i in range(labels.shape[1])),
        sample_keys=np.array([f"s{i}" for i in range(labels.shape[0])]),
        mention_probabilities=mention,
    )


def test_masked_polarity_leaves_labels_untouched():
    labels = np.array([[MISSING, NEGATIVE], [POSITIVE, UNCERTAIN]])
    assert np.array_equal(frame_labels(labels, MASKED_POLARITY), labels)


def test_study_presence_folds_blank_and_uncertain_into_negative():
    labels = np.array([[MISSING, NEGATIVE], [POSITIVE, UNCERTAIN]])
    framed = frame_labels(labels, STUDY_PRESENCE)
    assert np.array_equal(framed, np.array([[NEGATIVE, NEGATIVE], [POSITIVE, NEGATIVE]]))
    assert MISSING not in framed
    assert UNCERTAIN not in framed


def test_study_presence_lowers_prevalence_and_breaks_the_constant_predictor():
    # 10 studies, one finding: 2 explicit positives, 1 explicit negative, 7 blank.
    labels = np.array([[POSITIVE], [POSITIVE], [NEGATIVE]] + [[MISSING]] * 7)
    preds = _predictions(labels, np.full((10, 1), 0.9))

    masked = apply_framing(preds, MASKED_POLARITY)
    valid = masked.labels[:, 0] >= 0
    assert valid.sum() == 3
    assert (masked.labels[valid, 0] == POSITIVE).mean() == pytest.approx(2 / 3)

    presence = apply_framing(preds, STUDY_PRESENCE)
    assert presence.labels.shape == labels.shape
    assert (presence.labels[:, 0] == POSITIVE).mean() == pytest.approx(0.2)


def test_marginal_presence_multiplies_the_gate_into_the_score():
    labels = np.array([[POSITIVE, NEGATIVE]])
    q_pos = np.array([[0.8, 0.5]])
    mention = np.array([[0.5, 0.2]])
    preds = _predictions(labels, q_pos, mention=mention)

    assert presence_scores(preds, "conditional_positive") == pytest.approx(q_pos)
    assert presence_scores(preds, "marginal_presence") == pytest.approx(
        np.array([[0.4, 0.1]])
    )

    framed = apply_framing(preds, STUDY_PRESENCE, "marginal_presence")
    assert framed.positive_probabilities == pytest.approx(np.array([[0.4, 0.1]]))
    # The rewritten distribution must still be a distribution, or AUROC and the
    # calibration search silently read a column that does not mean what it says.
    assert framed.probabilities.sum(axis=-1) == pytest.approx(np.ones((1, 2)))


def test_marginal_presence_refuses_a_file_without_the_gate():
    preds = _predictions(np.array([[POSITIVE]]), np.array([[0.6]]))
    with pytest.raises(ScoreUnavailableError):
        presence_scores(preds, "marginal_presence")


def test_framing_is_recorded_so_a_threshold_file_can_be_checked():
    preds = _predictions(np.array([[POSITIVE]]), np.array([[0.6]]))
    framed = apply_framing(preds, STUDY_PRESENCE)
    assert framed.metadata["label_framing"] == STUDY_PRESENCE
    assert framed.metadata["score"] == "conditional_positive"


def test_unknown_framing_is_rejected():
    with pytest.raises(UnknownFramingError):
        frame_labels(np.array([[POSITIVE]]), "whatever")


def test_mention_probabilities_survive_a_save_load_round_trip(tmp_path):
    mention = np.array([[0.25, 0.75]])
    preds = _predictions(np.array([[POSITIVE, NEGATIVE]]), np.array([[0.4, 0.6]]), mention)
    path = preds.save(tmp_path / "p.npz")
    again = ClassificationPredictions.load(path)
    assert again.mention_probabilities == pytest.approx(mention)


def test_mention_probabilities_shape_is_validated():
    from training.evaluation.schemas import SchemaError

    with pytest.raises(SchemaError):
        _predictions(
            np.array([[POSITIVE, NEGATIVE]]),
            np.array([[0.4, 0.6]]),
            mention=np.array([[0.5]]),
        )
