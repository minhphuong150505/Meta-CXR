"""Excluding a label must remove it everywhere, and must not leave empty rows valid.

`No Finding` is excluded by default because CheXpert never emits a negative for
it -- 74,305 positives and 0 negatives across the whole train split -- so the
only thing it can teach is a constant. The trap is that 22.3% of train studies
carry it as their *only* label: masking the column without recomputing the
per-row validity flag leaves those rows marked valid while holding no usable
cell at all, which is silent, since the loss simply skips them and still returns
a finite number.
"""

import numpy as np
import pytest
import torch

from mhcac.loss import ClassificationLoss

IGNORE_LABEL = -100


def test_a_fully_masked_column_trains_nothing():
    """The state an excluded label is left in."""
    torch.manual_seed(0)
    logits = torch.randn(6, 3, 3, requires_grad=True)
    labels = torch.tensor([[1, 0, IGNORE_LABEL]] * 3 + [[0, 1, IGNORE_LABEL]] * 3)

    ClassificationLoss(num_abnormalities=3)(logits, labels).backward()

    assert torch.equal(logits.grad[:, 2], torch.zeros_like(logits.grad[:, 2]))
    assert torch.count_nonzero(logits.grad[:, 0]) > 0


def test_a_row_left_with_no_usable_cell_contributes_nothing():
    """A study whose only label was excluded must not shift the loss."""
    torch.manual_seed(0)
    loss_fn = ClassificationLoss(num_abnormalities=2)
    kept = torch.randn(4, 2, 3)
    kept_labels = torch.tensor([[1, 0], [0, 1], [1, 1], [0, 0]])

    empty_row = torch.randn(1, 2, 3)
    empty_labels = torch.full((1, 2), IGNORE_LABEL)
    with_empty = loss_fn(
        torch.cat([kept, empty_row]), torch.cat([kept_labels, empty_labels])
    )

    assert with_empty == pytest.approx(float(loss_fn(kept, kept_labels)))


def test_an_all_masked_batch_stays_finite_and_differentiable():
    """The whole batch can end up empty once a dominant label is excluded."""
    logits = torch.randn(3, 2, 3, requires_grad=True)
    labels = torch.full((3, 2), IGNORE_LABEL)

    loss = ClassificationLoss(num_abnormalities=2)(logits, labels)
    loss.backward()

    assert torch.isfinite(loss) and float(loss.detach()) == 0.0
    assert logits.grad is not None and torch.count_nonzero(logits.grad) == 0


def test_validity_flag_is_recomputed_from_the_kept_columns_only():
    """Mirrors what ReportDataset does after masking excluded columns."""
    labels = np.array(
        [[1, IGNORE_LABEL, IGNORE_LABEL],   # only the excluded label
         [IGNORE_LABEL, 0, IGNORE_LABEL],   # a real label survives
         [IGNORE_LABEL, IGNORE_LABEL, IGNORE_LABEL]],
        dtype=np.int8,
    )
    excluded = [0]

    kept = [i for i in range(labels.shape[1]) if i not in excluded]
    masked = labels.copy()
    masked[:, excluded] = IGNORE_LABEL
    valid = (masked[:, kept] >= 0).any(axis=1)

    assert valid.tolist() == [False, True, False], (
        "a row whose only label was excluded must stop counting as labelled"
    )
