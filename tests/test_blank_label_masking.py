"""A blank CheXpert cell must never train as a negative.

79.4% of the CheXpert label matrix is blank, and blank means the labeler found
no mention of the finding -- not that the radiologist ruled it out. Folding
blanks into class 0 made roughly nine in ten "negatives" an absence of evidence,
so they are masked per cell instead. These tests pin that the mask actually
reaches the loss, because the failure is silent: training a blank as a negative
lowers the loss just as smoothly as training a real one.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from mhcac.explanation import logit_difference_squared
from mhcac.loss import ClassificationLoss

IGNORE_LABEL = -100


def _map_chexpert(frame, cols):
    """The exact expression ReportDataset.__init__ applies to the export."""
    return frame[cols].replace(-1, 2).fillna(IGNORE_LABEL).astype("int8")


def test_blank_cells_map_to_the_ignore_sentinel_not_negative():
    frame = pd.DataFrame({"Cardiomegaly": [1.0, 0.0, -1.0, np.nan]})

    mapped = _map_chexpert(frame, ["Cardiomegaly"])["Cardiomegaly"].tolist()

    assert mapped == [1, 0, 2, IGNORE_LABEL]
    assert mapped[3] != 0, "a blank must not be indistinguishable from a negative"


def test_ignore_sentinel_survives_int8_storage():
    """int8 spans -128..127, so the sentinel must not wrap into a real class."""
    stored = pd.Series([IGNORE_LABEL], dtype="int8").iloc[0]

    assert int(stored) == IGNORE_LABEL
    assert int(stored) < 0


def test_masked_cells_contribute_no_gradient():
    torch.manual_seed(0)
    logits = torch.randn(4, 2, 3, requires_grad=True)
    labels = torch.tensor([[1, IGNORE_LABEL], [0, IGNORE_LABEL],
                           [1, IGNORE_LABEL], [0, IGNORE_LABEL]])

    ClassificationLoss(num_abnormalities=2)(logits, labels).backward()

    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad[:, 0]) > 0, "label 0 should train"
    assert torch.equal(logits.grad[:, 1], torch.zeros_like(logits.grad[:, 1])), (
        "every cell of label 1 is blank, so it must receive no gradient"
    )


def test_blank_label_loss_matches_dropping_the_label_entirely():
    torch.manual_seed(0)
    both = torch.randn(4, 2, 3)
    labels_masked = torch.tensor([[1, IGNORE_LABEL], [0, IGNORE_LABEL],
                                  [1, IGNORE_LABEL], [0, IGNORE_LABEL]])
    only = both[:, :1].clone()
    labels_only = labels_masked[:, :1]

    loss_masked = ClassificationLoss(num_abnormalities=2)(both, labels_masked)
    loss_only = ClassificationLoss(num_abnormalities=1)(only, labels_only)

    assert loss_masked == pytest.approx(float(loss_only))


def test_blank_cells_are_not_read_as_positive_by_the_explanation_score():
    logits = torch.randn(1, 3, 3)
    labels = torch.tensor([[IGNORE_LABEL, IGNORE_LABEL, IGNORE_LABEL]])

    score, valid = logit_difference_squared(logits, labels)

    assert float(score) == 0.0
    assert not bool(valid[0]), "an all-blank study has no positive finding to explain"
