"""Hand-computed checks for the pure-tensor rollout.

Every expected value in this file is derived on paper in the comment above it.
That is the whole point of keeping ``rollout.py`` free of model imports: if
these numbers are right, the attribution maths is right, independently of
whether a GPU, MedGemma or the dataset is reachable.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from training.explainability.rollout import (  # noqa: E402
    METHOD_ABNAR,
    METHOD_CHEFER,
    REDUCE_MAX,
    REDUCE_SUM,
    fuse_heads,
    rollout,
    span_attribution,
    stack_layers,
)


def _attn(rows) -> torch.Tensor:
    """One layer, one head: [1, 1, S, S]."""
    return torch.tensor(rows, dtype=torch.float32)[None, None]


# --------------------------------------------------------------------------
# fuse_heads
# --------------------------------------------------------------------------


def test_fuse_heads_without_gradient_is_the_head_average():
    # Two heads: [[1,0],[0,1]] and [[0,1],[1,0]]. Mean = [[.5,.5],[.5,.5]].
    attention = torch.tensor(
        [[[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]]], dtype=torch.float32
    )
    fused = fuse_heads(attention)
    assert fused.shape == (1, 2, 2)
    assert torch.allclose(fused[0], torch.full((2, 2), 0.5))


def test_fuse_heads_multiplies_elementwise_by_the_gradient():
    # A = [[1,0],[.5,.5]], G = [[2,3],[4,0]]
    # A*G = [[2,0],[2,0]] -> single head, so the mean is itself.
    attention = _attn([[1.0, 0.0], [0.5, 0.5]])
    gradient = _attn([[2.0, 3.0], [4.0, 0.0]])
    fused = fuse_heads(attention, gradient)
    assert torch.allclose(fused[0], torch.tensor([[2.0, 0.0], [2.0, 0.0]]))


def test_fuse_heads_clamps_before_averaging_not_after():
    # This is the assertion that pins the reference implementation's order.
    # Head 0: A*G = [[+4]]   Head 1: A*G = [[-4]]
    #   clamp-then-mean = mean(4, 0) = 2.0   <- what Chefer does
    #   mean-then-clamp = clamp(0)   = 0.0   <- the plausible-looking mistake
    attention = torch.tensor([[[[1.0]], [[1.0]]]], dtype=torch.float32)
    gradient = torch.tensor([[[[4.0]], [[-4.0]]]], dtype=torch.float32)
    fused = fuse_heads(attention, gradient)
    assert float(fused[0, 0, 0]) == pytest.approx(2.0)


def test_fuse_heads_row_normalize_makes_rows_sum_to_one():
    attention = _attn([[2.0, 6.0], [0.0, 0.0]])
    fused = fuse_heads(attention, row_normalize=True)
    # Row 0: 2/8, 6/8. Row 1 is all zero and must stay zero, not become NaN.
    assert torch.allclose(fused[0, 0], torch.tensor([0.25, 0.75]))
    assert torch.allclose(fused[0, 1], torch.zeros(2))


def test_fuse_heads_rejects_a_shape_mismatch():
    with pytest.raises(ValueError, match="identical shapes"):
        fuse_heads(_attn([[1.0, 0.0], [0.0, 1.0]]), torch.ones(1, 1, 3, 3))


def test_fuse_heads_rejects_non_finite_input():
    with pytest.raises(ValueError, match="non-finite"):
        fuse_heads(_attn([[float("nan"), 0.0], [0.0, 1.0]]))


def test_fuse_heads_rejects_a_non_square_matrix():
    with pytest.raises(ValueError, match="square"):
        fuse_heads(torch.ones(1, 1, 2, 3))


# --------------------------------------------------------------------------
# rollout -- Chefer
# --------------------------------------------------------------------------


def test_chefer_single_layer_is_identity_plus_a():
    # R = I + A@I = I + A
    #   A = [[1,0],[.5,.5]]  ->  R = [[2,0],[.5,1.5]]
    fused = fuse_heads(_attn([[1.0, 0.0], [0.5, 0.5]]))
    result, trace = rollout(fused, method=METHOD_CHEFER)
    assert torch.allclose(result, torch.tensor([[2.0, 0.0], [0.5, 1.5]]))
    assert trace.num_layers == 1
    assert trace.sequence_length == 2


def test_chefer_two_layers_is_i_plus_2a_plus_a_squared():
    # R1 = I + A
    # R2 = R1 + A@R1 = (I + A) + A(I + A) = I + 2A + A^2
    #   A   = [[1,0],[.5,.5]]
    #   A^2 = [[1,0],[.75,.25]]
    #   R2  = [[1+2+1, 0], [0+1+0.75, 1+1+0.25]] = [[4,0],[1.75,2.25]]
    single = _attn([[1.0, 0.0], [0.5, 0.5]])
    fused = fuse_heads(torch.cat([single, single], dim=0))
    result, trace = rollout(fused, method=METHOD_CHEFER)
    assert torch.allclose(result, torch.tensor([[4.0, 0.0], [1.75, 2.25]]))
    assert trace.num_layers == 2


def test_chefer_layers_compose_in_order_and_are_not_commuted():
    # Two DIFFERENT layers: R = I + A1 + A2 + A2@A1.
    # Wiring them the other way round gives A1@A2, which differs here, so this
    # test fails if the recurrence is ever flipped.
    a1 = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    a2 = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
    fused = torch.stack([a1, a2], dim=0)
    result, _ = rollout(fused, method=METHOD_CHEFER)
    expected = torch.eye(2) + a1 + a2 + a2 @ a1
    assert torch.allclose(result, expected)
    assert not torch.allclose(a2 @ a1, a1 @ a2)


def test_chefer_with_an_all_zero_layer_leaves_the_map_unchanged():
    # A layer that attributes nothing must be a no-op: R = R + 0@R = R.
    live = _attn([[1.0, 0.0], [0.5, 0.5]])
    dead = torch.zeros_like(live)
    with_dead = fuse_heads(torch.cat([live, dead], dim=0))
    alone = fuse_heads(live)
    assert torch.allclose(rollout(with_dead)[0], rollout(alone)[0])


def test_subtract_identity_removes_exactly_the_diagonal_one():
    fused = fuse_heads(_attn([[1.0, 0.0], [0.5, 0.5]]))
    plain, _ = rollout(fused)
    stripped, _ = rollout(fused, subtract_identity=True)
    assert torch.allclose(plain - stripped, torch.eye(2))


def test_rollout_is_deterministic():
    fused = fuse_heads(torch.rand(4, 3, 6, 6))
    first, _ = rollout(fused)
    second, _ = rollout(fused)
    assert torch.equal(first, second)


def test_rollout_output_is_non_negative():
    # Every entry is a sum of products of clamped, non-negative terms.
    fused = fuse_heads(torch.rand(5, 4, 7, 7), torch.randn(5, 4, 7, 7))
    result, _ = rollout(fused)
    assert bool((result >= 0).all())


# --------------------------------------------------------------------------
# rollout -- Abnar fallback
# --------------------------------------------------------------------------


def test_abnar_single_layer_adds_residual_then_row_normalizes():
    # A = [[1,0],[.5,.5]]  ->  0.5A + 0.5I = [[1,0],[.25,.75]]
    # Row sums are 1 and 1, so normalisation is already a no-op here.
    fused = fuse_heads(_attn([[1.0, 0.0], [0.5, 0.5]]))
    result, trace = rollout(fused, method=METHOD_ABNAR)
    assert torch.allclose(result, torch.tensor([[1.0, 0.0], [0.25, 0.75]]))
    assert trace.method == METHOD_ABNAR


def test_abnar_rows_always_sum_to_one():
    # Unnormalised attention, deliberately: the method must renormalise it.
    fused = fuse_heads(_attn([[2.0, 6.0], [1.0, 1.0]]))
    result, _ = rollout(fused, method=METHOD_ABNAR)
    assert torch.allclose(result.sum(dim=-1), torch.ones(2))


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="method must be one of"):
        rollout(fuse_heads(_attn([[1.0]])), method="grad_cam")


# --------------------------------------------------------------------------
# The trace -- a fallback must never be silent
# --------------------------------------------------------------------------


def test_trace_reports_the_gradient_free_fallback_as_unweighted():
    fused = fuse_heads(_attn([[1.0, 0.0], [0.0, 1.0]]))
    _, trace = rollout(fused, gradient_weighted=False)
    assert trace.gradient_weighted is False
    assert trace.to_dict()["gradient_weighted"] is False


def test_trace_defaults_to_unweighted_when_the_caller_says_nothing():
    # Guessing generously here would let a fallback be reported as a full
    # gradient-weighted run.
    fused = fuse_heads(_attn([[1.0, 0.0], [0.0, 1.0]]))
    _, trace = rollout(fused)
    assert trace.gradient_weighted is False


# --------------------------------------------------------------------------
# span_attribution
# --------------------------------------------------------------------------


def test_span_attribution_selects_the_requested_block_in_order():
    matrix = torch.tensor(
        [
            [0.0, 1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0, 7.0],
            [8.0, 9.0, 10.0, 11.0],
            [12.0, 13.0, 14.0, 15.0],
        ]
    )
    # Queries {2,3}, keys {0,1}: column means are (8+12)/2=10 and (9+13)/2=11.
    values = span_attribution(matrix, [2, 3], [0, 1], normalize=False)
    assert torch.allclose(values, torch.tensor([10.0, 11.0]))
    # Key order is honoured, not sorted.
    reversed_values = span_attribution(matrix, [2, 3], [1, 0], normalize=False)
    assert torch.allclose(reversed_values, torch.tensor([11.0, 10.0]))


def test_span_attribution_normalizes_to_a_distribution():
    matrix = torch.tensor([[0.0, 0.0, 0.0], [1.0, 3.0, 0.0], [0.0, 0.0, 0.0]])
    values = span_attribution(matrix, [1], [0, 1])
    assert torch.allclose(values, torch.tensor([0.25, 0.75]))
    assert float(values.sum()) == pytest.approx(1.0)


def test_a_span_with_no_mass_returns_zeros_not_a_uniform_distribution():
    # "This sentence did not use the image" is a real answer. Turning it into a
    # flat 1/N would invent evidence that is not there.
    matrix = torch.zeros(3, 3)
    values = span_attribution(matrix, [2], [0, 1])
    assert torch.allclose(values, torch.zeros(2))
    assert bool(torch.isfinite(values).all())


def test_span_attribution_reduction_modes():
    matrix = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [6.0, 0.0, 0.0]])
    assert float(span_attribution(matrix, [1, 2], [0], reduce=REDUCE_SUM, normalize=False)) == 8.0
    assert float(span_attribution(matrix, [1, 2], [0], reduce=REDUCE_MAX, normalize=False)) == 6.0


def test_span_attribution_rejects_out_of_range_and_duplicate_positions():
    matrix = torch.eye(3)
    with pytest.raises(IndexError, match="outside"):
        span_attribution(matrix, [0], [3])
    with pytest.raises(ValueError, match="duplicate"):
        span_attribution(matrix, [0], [1, 1])
    with pytest.raises(ValueError, match="must not be empty"):
        span_attribution(matrix, [0], [])


def test_span_attribution_rejects_a_non_square_matrix():
    with pytest.raises(ValueError, match="square"):
        span_attribution(torch.zeros(2, 3), [0], [1])


# --------------------------------------------------------------------------
# stack_layers -- the seam to a live model
# --------------------------------------------------------------------------


def test_stack_layers_picks_one_batch_row_from_each_layer():
    # Layer l, batch row b is filled with the constant (10*l + b).
    layers = [torch.full((2, 1, 3, 3), float(10 * layer + 0)) for layer in range(2)]
    for layer_index, layer in enumerate(layers):
        layer[1] = float(10 * layer_index + 1)
    stacked = stack_layers(layers, batch_index=1)
    assert stacked.shape == (2, 1, 3, 3)
    assert float(stacked[0, 0, 0, 0]) == 1.0
    assert float(stacked[1, 0, 0, 0]) == 11.0


def test_stack_layers_rejects_an_out_of_range_batch_index():
    with pytest.raises(IndexError, match="batch_index"):
        stack_layers([torch.zeros(1, 1, 2, 2)], batch_index=1)


def test_stack_layers_rejects_layers_of_different_shapes():
    with pytest.raises(ValueError, match="disagree on shape"):
        stack_layers([torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 3, 3)])


def test_stack_layers_rejects_an_empty_sequence():
    with pytest.raises(ValueError, match="at least one layer"):
        stack_layers([])


# --------------------------------------------------------------------------
# End-to-end on a hand-built two-layer example
# --------------------------------------------------------------------------


def test_soft_token_span_is_attributed_through_two_layers_by_hand():
    """S=4: positions 0,1 are 'soft tokens', 2,3 are 'generated tokens'.

    Layer 1 -- token 2 reads soft token 0, token 3 reads soft token 1:
        A1 = [[0,0,0,0],
              [0,0,0,0],
              [1,0,0,0],
              [0,1,0,0]]
    Layer 2 -- token 3 reads token 2 only:
        A2 = [[0,0,0,0],
              [0,0,0,0],
              [0,0,0,0],
              [0,0,1,0]]

    R = I + A1 + A2 + A2@A1, and A2@A1 row 3 = row 2 of A1 = [1,0,0,0].
    So R[3, 0:2] = A1[3,0:2] + (A2@A1)[3,0:2] = [0,1] + [1,0] = [1,1]
    and R[2, 0:2] = [1,0].

    Querying token 3 alone must therefore split its attribution evenly between
    the two soft tokens: it read soft token 1 directly and soft token 0 through
    token 2. Querying token 2 alone must put everything on soft token 0.
    """
    a1 = torch.zeros(4, 4)
    a1[2, 0] = 1.0
    a1[3, 1] = 1.0
    a2 = torch.zeros(4, 4)
    a2[3, 2] = 1.0
    fused = torch.stack([a1, a2], dim=0)
    matrix, _ = rollout(fused, method=METHOD_CHEFER)

    soft = [0, 1]
    assert torch.allclose(span_attribution(matrix, [3], soft), torch.tensor([0.5, 0.5]))
    assert torch.allclose(span_attribution(matrix, [2], soft), torch.tensor([1.0, 0.0]))
