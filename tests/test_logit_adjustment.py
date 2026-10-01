"""Logit-adjusted classification loss (Menon et al., ICLR 2021, Eq. 10), D-025.

The loss sees ``logits + tau * log(prior)``; the model's logits, and therefore
argmax and every metric, stay unadjusted. Three classes throughout.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mhcac.loss import ClassificationLoss, logit_adjustment_offsets  # noqa: E402

COUNTS = [[60, 30, 10], [90, 10, 0]]  # finding 1 never uncertain


def test_offsets_are_tau_log_prior_hand_computed():
    off = logit_adjustment_offsets(COUNTS, num_abnormalities=2, tau=1.0)
    assert off[0].tolist() == pytest.approx([math.log(0.6), math.log(0.3), math.log(0.1)])
    assert off[1].tolist() == pytest.approx([math.log(0.9), math.log(0.1), 0.0])
    half = logit_adjustment_offsets(COUNTS, num_abnormalities=2, tau=0.5)
    assert half[0, 2].item() == pytest.approx(0.5 * math.log(0.1))


def test_zero_count_class_gets_offset_zero_not_minus_infinity():
    off = logit_adjustment_offsets(COUNTS, num_abnormalities=2)
    assert torch.isfinite(off).all()
    assert off[1, 2].item() == 0.0


def test_loss_equals_plain_cross_entropy_on_shifted_logits():
    torch.manual_seed(0)
    logits = torch.randn(5, 2, 3)
    labels = torch.tensor([[0, 1], [1, 0], [2, 0], [0, 1], [2, 0]])
    loss = ClassificationLoss(num_abnormalities=2, logit_adjust_counts=COUNTS)(logits, labels)
    off = logit_adjustment_offsets(COUNTS, num_abnormalities=2)
    expected = torch.stack([
        F.cross_entropy(logits[:, i] + off[i], labels[:, i]) for i in range(2)
    ]).mean()
    assert loss.item() == pytest.approx(expected.item(), rel=1e-6)


def test_rare_class_costs_more_to_miss_than_with_plain_cross_entropy():
    # Same logits, true class Uncertain (prior 0.1): the adjusted loss is larger,
    # so the gradient pushes harder toward the rare class.
    logits = torch.zeros(1, 1, 3)
    labels = torch.tensor([[2]])
    plain = ClassificationLoss(num_abnormalities=1)(logits, labels)
    adjusted = ClassificationLoss(num_abnormalities=1, logit_adjust_counts=[COUNTS[0]])(
        logits, labels
    )
    assert adjusted.item() > plain.item()


def test_class_weights_and_logit_adjustment_are_exclusive():
    with pytest.raises(ValueError, match="replaces class_weights"):
        ClassificationLoss(
            num_abnormalities=2,
            class_weights=[[1.0, 1.0, 1.0]] * 2,
            logit_adjust_counts=COUNTS,
        )


def test_ignored_cells_are_still_skipped():
    logits = torch.randn(2, 2, 3)
    labels = torch.tensor([[-100, 1], [-100, 0]])
    loss = ClassificationLoss(num_abnormalities=2, logit_adjust_counts=COUNTS)(logits, labels)
    off = logit_adjustment_offsets(COUNTS, num_abnormalities=2)
    assert loss.item() == pytest.approx(
        F.cross_entropy(logits[:, 1] + off[1], labels[:, 1]).item(), rel=1e-6
    )


def test_offsets_are_not_saved_in_checkpoints():
    module = ClassificationLoss(num_abnormalities=2, logit_adjust_counts=COUNTS)
    assert "logit_offsets" not in module.state_dict()


@pytest.mark.parametrize(
    "counts, match",
    [([[1, 2, 3]], r"\[2, 3\]"), ([[1, 2, 3], [0, 0, 0]], "positive total"),
     ([[1, 2, 3], [-1, 2, 3]], "non-negative")],
)
def test_bad_counts_are_refused(counts, match):
    with pytest.raises(ValueError, match=match):
        logit_adjustment_offsets(counts, num_abnormalities=2)


def test_class_weights_are_used_but_not_saved_in_checkpoints():
    weights = [[1.0, 3.0, 10.0], [1.0, 2.0, 5.0]]
    module = ClassificationLoss(num_abnormalities=2, class_weights=weights)
    assert not any("weight" in k for k in module.state_dict())
    logits = torch.randn(4, 2, 3)
    labels = torch.tensor([[0, 1], [1, 2], [2, 0], [1, 1]])
    expected = torch.stack([
        F.cross_entropy(logits[:, i], labels[:, i], weight=torch.tensor(weights[i]))
        for i in range(2)
    ]).mean()
    assert module(logits, labels).item() == pytest.approx(expected.item(), rel=1e-6)


def test_old_checkpoint_loss_buffers_are_dropped_on_load():
    from collections import OrderedDict

    from mhcac.loss import drop_loss_config_state

    state = OrderedDict(
        [
            ("mhcac.head.weight", torch.zeros(1)),
            ("cls_loss_fn.cross_entropy_loss_list.0.weight", torch.ones(3)),
            ("cls_loss_fn.cross_entropy_loss_list.13.weight", torch.ones(3)),
        ]
    )
    state._metadata = {"": {"version": 1}}
    kept = drop_loss_config_state(state)
    assert list(kept) == ["mhcac.head.weight"]
    assert kept._metadata == state._metadata
