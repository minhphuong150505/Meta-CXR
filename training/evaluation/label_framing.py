"""Which question the classification metrics are actually answering.

Stage-1 scores every metric over the CheXpert label matrix, but that matrix has
two defensible readings and they are **not** interchangeable. Leaving the choice
implicit made ``positive_macro_f1`` uninterpretable, so it is an explicit
parameter recorded in every output file, exactly like
:mod:`training.evaluation.uncertain_policy`.

Framings
--------
``masked_polarity`` (historical default)
    A blank CheXpert cell is *masked*: the labeler found no mention of the
    finding, which is not the radiologist ruling it out. Metrics therefore score
    **polarity given that the finding was mentioned**.

    This is the correct denominator for the loss -- and a broken one for F1.
    Measured on the 2026-08-19 test split: 53-98% of cells per label are blank,
    so the surviving "negatives" are the rare population where a radiologist
    wrote an explicit denial. Positives end up the majority for 12 of 14
    findings, and ``all_positive`` alone scores ``positive_macro_f1`` 0.8397
    against the model's 0.8717. A metric a constant beats by 0.03 is not
    measuring skill.

``study_presence``
    The clinical question: *does this study have finding X?* Every study counts
    for every finding. A cell is positive iff the label is POSITIVE; blank,
    negative and uncertain all mean "not present in the report".

    Prevalence returns to 0.019-0.344 on the same split, so ``all_positive``
    collapses to a macro F1 of 0.2280 and F1 can no longer be gamed by
    over-calling. This is the framing under which F1 is worth quoting.

    ⚠ Uncertain cells are folded into "not present" here, so
    :mod:`~training.evaluation.uncertain_policy` has nothing left to decide --
    the framing has already made the call. Say so when reporting.

Scores
------
The framing fixes the *labels*; ``score`` fixes what the model is asked for.

``conditional_positive`` (default)
    ``probabilities[..., POSITIVE]`` -- the polarity head's ``q``, which is
    conditional on the finding being mentioned.

``marginal_presence``
    ``mention_probability x q_positive``, the joint ``P(mentioned and
    positive)``. This is the quantity ``study_presence`` actually asks about,
    and it needs ``mention_probabilities`` in the prediction file (written only
    by runs whose eval hook collected the gate).

Scoring ``study_presence`` with ``conditional_positive`` is legitimate but
handicapped: the model's mention gate is exactly the missing factor, so the
resulting numbers are a floor, not an estimate.
"""

from __future__ import annotations

import numpy as np

from training.evaluation.schemas import (
    NEGATIVE,
    POSITIVE,
    ClassificationPredictions,
)

MASKED_POLARITY = "masked_polarity"
STUDY_PRESENCE = "study_presence"

FRAMINGS = (MASKED_POLARITY, STUDY_PRESENCE)
DEFAULT_FRAMING = MASKED_POLARITY

CONDITIONAL_POSITIVE = "conditional_positive"
MARGINAL_PRESENCE = "marginal_presence"

SCORES = (CONDITIONAL_POSITIVE, MARGINAL_PRESENCE)
DEFAULT_SCORE = CONDITIONAL_POSITIVE


class UnknownFramingError(ValueError):
    """The requested label framing does not exist."""


class ScoreUnavailableError(ValueError):
    """The prediction file lacks the arrays the requested score needs."""


def validate_framing(framing: str) -> str:
    if framing not in FRAMINGS:
        raise UnknownFramingError(
            f"unknown label framing {framing!r}; expected one of {', '.join(FRAMINGS)}"
        )
    return framing


def validate_score(score: str) -> str:
    if score not in SCORES:
        raise UnknownFramingError(
            f"unknown score {score!r}; expected one of {', '.join(SCORES)}"
        )
    return score


def frame_labels(labels: np.ndarray, framing: str = DEFAULT_FRAMING) -> np.ndarray:
    """Rewrite class-index labels under ``framing``.

    ``masked_polarity`` returns a copy unchanged. ``study_presence`` returns an
    array in ``{NEGATIVE, POSITIVE}`` with no missing entries: every cell is
    answerable because "not mentioned" is a valid answer to "is it present".
    """
    validate_framing(framing)
    labels = np.asarray(labels)
    if framing == MASKED_POLARITY:
        return labels.copy()
    return np.where(labels == POSITIVE, POSITIVE, NEGATIVE).astype(labels.dtype)


def presence_scores(
    predictions: ClassificationPredictions, score: str = DEFAULT_SCORE
) -> np.ndarray:
    """``[N, P]`` probability the finding is present, under ``score``."""
    validate_score(score)
    q_pos = predictions.probabilities[..., POSITIVE]
    if score == CONDITIONAL_POSITIVE:
        return q_pos
    mention = predictions.mention_probabilities
    if mention is None:
        raise ScoreUnavailableError(
            "score 'marginal_presence' needs mention_probabilities, which this "
            "prediction file does not carry. It is written only by runs whose "
            "eval hook collected the mention gate; re-export with a build that "
            "does, or fall back to 'conditional_positive'."
        )
    return np.asarray(mention, dtype=np.float64) * q_pos


def apply_framing(
    predictions: ClassificationPredictions,
    framing: str = DEFAULT_FRAMING,
    score: str = DEFAULT_SCORE,
) -> ClassificationPredictions:
    """Return a new prediction set rewritten for ``framing`` / ``score``.

    The rewrite is deliberately expressed as another
    :class:`ClassificationPredictions` rather than as a special case inside each
    metric: everything downstream -- calibration, AUROC, bootstrap, subgroups --
    then keeps reading ``probabilities[..., POSITIVE]`` and needs no change.

    Under ``study_presence`` the returned probabilities are a two-point
    distribution ``[1 - p, p, 0]``, because "uncertain" is not one of the
    answers the framing admits.
    """
    validate_framing(framing)
    validate_score(score)

    labels = frame_labels(predictions.labels, framing)
    if framing == MASKED_POLARITY and score == CONDITIONAL_POSITIVE:
        probabilities = predictions.probabilities.copy()
    else:
        p = np.clip(presence_scores(predictions, score), 0.0, 1.0)
        probabilities = np.zeros_like(predictions.probabilities)
        probabilities[..., NEGATIVE] = 1.0 - p
        probabilities[..., POSITIVE] = p

    metadata = dict(predictions.metadata)
    metadata["label_framing"] = framing
    metadata["score"] = score

    return ClassificationPredictions(
        labels=labels,
        probabilities=probabilities,
        pathology_names=predictions.pathology_names,
        sample_keys=predictions.sample_keys,
        logits=None,
        view_positions=predictions.view_positions,
        num_views=predictions.num_views,
        mention_probabilities=predictions.mention_probabilities,
        metadata=metadata,
    )
