import torch

from model.lavis.models.blip2_models.blip2_qformer import (
    _hard_negative_sampling_weights,
)


def test_hard_negative_weights_exclude_dominant_positive_before_softmax():
    similarities = torch.tensor(
        [
            [1000.0, 0.0, -1.0],
            [0.0, 1000.0, -1.0],
            [0.0, -1.0, 1000.0],
        ],
        dtype=torch.bfloat16,
    )

    weights = _hard_negative_sampling_weights(
        similarities,
        candidate_valid=torch.ones(3, dtype=torch.bool),
        positive_indices=torch.arange(3),
    )

    assert torch.isfinite(weights).all()
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(3))
    assert torch.equal(weights.diag(), torch.zeros(3))
    for row in weights:
        torch.multinomial(row, 1)


def test_hard_negative_weights_fallback_for_non_finite_similarities():
    similarities = torch.tensor(
        [
            [0.0, float("nan"), float("nan")],
            [float("inf"), 0.0, float("-inf")],
        ]
    )
    valid = torch.tensor([True, True, True])

    weights = _hard_negative_sampling_weights(
        similarities,
        candidate_valid=valid,
        positive_indices=torch.tensor([0, 1]),
    )

    assert torch.isfinite(weights).all()
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(2))
    assert weights[0, 0] == 0
    assert weights[1, 1] == 0


def test_hard_negative_weights_never_select_invalid_candidates():
    similarities = torch.tensor([[0.0, 10.0, 5.0]])

    weights = _hard_negative_sampling_weights(
        similarities,
        candidate_valid=torch.tensor([True, False, True]),
        positive_indices=torch.tensor([0]),
    )

    torch.testing.assert_close(weights, torch.tensor([[0.0, 0.0, 1.0]]))
