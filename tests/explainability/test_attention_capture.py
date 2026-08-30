"""CPU tier for the one module that touches a live model.

Everything here runs against a hand-built stand-in transformer, so the wiring --
module selection, hook install and removal, span location, NLL shifting, the
ablation rule -- is pinned without a GPU, MedGemma, or transformers. The model
itself is exercised on the training host, behind the ``gpu`` marker.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from training.explainability.attention_capture import (  # noqa: E402
    ALLOWED_SPLITS,
    ATTN_IMPLEMENTATION,
    SOURCE_QFORMER_SOFT_TOKEN,
    QFormerCrossAttentionUnavailable,
    SoftTokenAblationFailed,
    TrainSplitRefused,
    assert_split_allowed,
    assert_visual_tokens_matter,
    attribute_visual_tokens,
    capture_attention,
    disable_gradient_checkpointing,
    language_attention_modules,
    locate_visual_tokens,
    per_token_nll,
    qformer_cross_attention,
    stack_captured,
    warn_if_soft_token_source,
)
from training.explainability.projection import GridSpec  # noqa: E402

# --------------------------------------------------------------------------
# A stand-in model shaped like the real one, including the trap
# --------------------------------------------------------------------------


class FakeAttention(torch.nn.Module):
    """Returns ``(output, weights)`` like eager attention.

    The weights genuinely produce the output -- ``output = weights @ hidden`` --
    rather than being computed alongside it. That matters: an earlier version
    returned weights that nothing downstream consumed, and
    ``torch.autograd.grad`` then reported them as "not used in the graph",
    which is a property of the stand-in and not of any real attention layer.
    """

    def __init__(self, heads: int, length: int):
        super().__init__()
        self.heads, self.length = heads, length
        self.scale = torch.nn.Parameter(torch.ones(1))

    def forward(self, hidden):
        causal = torch.tril(torch.ones(1, self.heads, self.length, self.length))
        weights = torch.softmax(causal * self.scale - 1e9 * (1 - causal), dim=-1)
        return torch.matmul(weights.mean(dim=1), hidden), weights


class SdpaAttention(torch.nn.Module):
    """Returns ``(output, None)`` like sdpa -- the vision tower's shape."""

    def forward(self, hidden):
        return hidden, None


class FakeLayer(torch.nn.Module):
    def __init__(self, heads, length):
        super().__init__()
        self.self_attn = FakeAttention(heads, length)


class FakeVisionLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = SdpaAttention()


class FakeModel(torch.nn.Module):
    """`model.language_model.layers.N.self_attn` plus a decoy vision tower."""

    def __init__(self, layers=3, heads=2, length=6, vision_layers=2):
        super().__init__()
        self.language_model = torch.nn.Module()
        self.language_model.layers = torch.nn.ModuleList(
            [FakeLayer(heads, length) for _ in range(layers)]
        )
        self.vision_tower = torch.nn.Module()
        self.vision_tower.layers = torch.nn.ModuleList(
            [FakeVisionLayer() for _ in range(vision_layers)]
        )
        self.is_gradient_checkpointing = False

    def forward(self, hidden):
        for layer in self.language_model.layers:
            hidden, _ = layer.self_attn(hidden)
        for layer in self.vision_tower.layers:
            hidden, _ = layer.self_attn(hidden)
        return hidden


@pytest.fixture
def model():
    return FakeModel()


# --------------------------------------------------------------------------
# The train split is refused, not warned about
# --------------------------------------------------------------------------


def test_train_split_raises():
    with pytest.raises(TrainSplitRefused, match="RandomAffine"):
        assert_split_allowed("train")


def test_train_split_is_refused_case_insensitively():
    for value in ("TRAIN", " Train ", "train"):
        with pytest.raises(TrainSplitRefused):
            assert_split_allowed(value)


@pytest.mark.parametrize("split", ALLOWED_SPLITS)
def test_val_and_test_are_allowed(split):
    assert assert_split_allowed(split) == split


def test_an_unknown_split_is_rejected_too():
    with pytest.raises(ValueError, match="split must be one of"):
        assert_split_allowed("holdout")


# --------------------------------------------------------------------------
# Module selection must never reach the vision tower
# --------------------------------------------------------------------------


def test_only_language_model_attention_is_selected(model):
    names = [name for name, _ in language_attention_modules(model)]
    assert len(names) == 3
    assert all(".language_model.layers." in f".{n}" for n in names)
    assert not any("vision" in n for n in names)


def test_modules_come_back_in_layer_order(model):
    names = [name for name, _ in language_attention_modules(model)]
    indices = [int(n.split(".layers.")[1].split(".")[0]) for n in names]
    assert indices == sorted(indices) == [0, 1, 2]


def test_layer_ten_does_not_sort_before_layer_two():
    """Lexicographic ordering would put layer 10 between 1 and 2."""
    big = FakeModel(layers=12)
    names = [name for name, _ in language_attention_modules(big)]
    indices = [int(n.split(".layers.")[1].split(".")[0]) for n in names]
    assert indices == list(range(12))


def test_a_model_with_no_matching_modules_raises():
    with pytest.raises(RuntimeError, match="no language-model self-attention"):
        language_attention_modules(torch.nn.Linear(2, 2))


# --------------------------------------------------------------------------
# Hooks are always removed
# --------------------------------------------------------------------------


def test_capture_collects_every_language_layer(model):
    with capture_attention(model) as captured:
        model(torch.zeros(1, 6, 4))
    assert sorted(captured) == [0, 1, 2]
    assert captured[0].shape == (1, 2, 6, 6)


def test_sdpa_style_none_weights_are_not_captured(model):
    """The vision tower returns None in slot 1; it must never enter the dict."""
    with capture_attention(model) as captured:
        model(torch.zeros(1, 6, 4))
    assert len(captured) == 3  # 3 language layers, not 5


def test_hooks_are_removed_after_the_context(model):
    before = sum(len(m._forward_hooks) for _n, m in language_attention_modules(model))
    with capture_attention(model):
        pass
    after = sum(len(m._forward_hooks) for _n, m in language_attention_modules(model))
    assert before == after == 0


def test_hooks_are_removed_even_when_the_body_raises(model):
    with pytest.raises(ZeroDivisionError):
        with capture_attention(model):
            raise ZeroDivisionError
    assert sum(len(m._forward_hooks) for _n, m in language_attention_modules(model)) == 0


# --------------------------------------------------------------------------
# Gradient checkpointing is turned off, and the fact is reported
# --------------------------------------------------------------------------


def test_disable_reports_false_when_it_was_already_off(model):
    assert disable_gradient_checkpointing(model) is False


def test_disable_turns_it_off_and_reports_true():
    class Checkpointed(FakeModel):
        def __init__(self):
            super().__init__()
            self.is_gradient_checkpointing = True
            self.disabled = False

        def gradient_checkpointing_disable(self):
            self.disabled = True
            self.is_gradient_checkpointing = False

    m = Checkpointed()
    assert disable_gradient_checkpointing(m) is True
    assert m.disabled is True
    assert m.is_gradient_checkpointing is False


# --------------------------------------------------------------------------
# stack_captured
# --------------------------------------------------------------------------


def test_stack_captured_orders_layers_and_casts_to_fp32():
    captured = {i: torch.full((1, 2, 4, 4), float(i), dtype=torch.bfloat16) for i in range(3)}
    stacked = stack_captured(captured, expected_layers=3)
    assert stacked.shape == (3, 2, 4, 4)
    assert stacked.dtype == torch.float32
    assert [float(stacked[i, 0, 0, 0]) for i in range(3)] == [0.0, 1.0, 2.0]


def test_a_partial_capture_is_refused():
    captured = {0: torch.zeros(1, 2, 4, 4), 2: torch.zeros(1, 2, 4, 4)}
    with pytest.raises(RuntimeError, match="captured 2 attention layers"):
        stack_captured(captured, expected_layers=3)


# --------------------------------------------------------------------------
# Locating the visual block
# --------------------------------------------------------------------------


def _ids(prefix, count, suffix, token_id=262144):
    return torch.tensor([[7] * prefix + [token_id] * count + [9] * suffix])


def test_locates_a_contiguous_block_like_the_real_model():
    # The real thing: 256 image tokens at [5, 260] of a 282-token sequence.
    span = locate_visual_tokens(
        _ids(5, 256, 21), 262144, expected_count=256,
        source="medgemma_image_grid", grid=GridSpec(16, 16, 56),
    )
    assert (span.start, span.end, span.length) == (5, 261, 256)
    assert span.positions[0] == 5 and span.positions[-1] == 260


def test_a_wrong_token_count_is_refused():
    with pytest.raises(RuntimeError, match="expected 256 visual tokens, found 128"):
        locate_visual_tokens(_ids(5, 128, 5), 262144, expected_count=256, source="s")


def test_a_split_block_is_refused():
    ids = torch.tensor([[7, 262144, 262144, 9, 262144, 262144, 7]])
    with pytest.raises(RuntimeError, match="not contiguous"):
        locate_visual_tokens(ids, 262144, expected_count=4, source="s")


def test_no_visual_token_at_all_is_refused():
    with pytest.raises(RuntimeError, match="no visual token"):
        locate_visual_tokens(torch.tensor([[1, 2, 3]]), 262144, expected_count=1, source="s")


def test_a_grid_that_disagrees_with_the_count_is_refused():
    with pytest.raises(ValueError, match="grid declares"):
        locate_visual_tokens(
            _ids(1, 256, 1), 262144, expected_count=256,
            source="s", grid=GridSpec(7, 7, 64),
        )


# --------------------------------------------------------------------------
# per_token_nll -- the shift is the whole point
# --------------------------------------------------------------------------


def test_nll_is_shifted_so_a_value_belongs_to_its_own_token():
    # 3 positions, 2-way vocab. logits[t] predicts labels[t+1], so position 1 is
    # scored by logits[0] (confident, correct) and position 2 by logits[1]
    # (a coin flip). logits[2] is never read -- nothing follows position 2.
    logits = torch.tensor([[[0.0, 100.0], [0.0, 0.0], [5.0, 5.0]]])
    labels = torch.tensor([[-100, 1, 0]])
    values = per_token_nll(logits, labels)
    assert torch.isnan(values[0])                      # nothing predicts position 0
    assert float(values[1]) == pytest.approx(0.0, abs=1e-4)   # predicted 1, confidently
    assert float(values[2]) == pytest.approx(0.6931, abs=1e-3)  # ln 2, a coin flip


def test_masked_positions_come_back_as_nan_not_zero():
    # 0.0 would read as "the model was certain"; nan is "not supervised".
    logits = torch.zeros(1, 4, 2)
    labels = torch.tensor([[-100, -100, 1, -100]])
    values = per_token_nll(logits, labels)
    assert torch.isnan(values[0]) and torch.isnan(values[1]) and torch.isnan(values[3])
    assert not torch.isnan(values[2])


def test_nll_rejects_a_length_mismatch():
    with pytest.raises(ValueError, match="logits cover"):
        per_token_nll(torch.zeros(1, 4, 2), torch.tensor([[1, 1]]))


# --------------------------------------------------------------------------
# The ablation gate -- a STOP condition
# --------------------------------------------------------------------------


# Real measurements, 12 MIMIC test studies, MedGemma bf16, 2026-08-30. Per-study
# mean-token-NLL deltas against the baseline. No identifiers, no report text --
# these are aggregate numbers and nothing else.
REAL_ZERO_DELTAS = [
    -0.0301, -0.0477, -0.0848, -0.2106, +0.2046, +0.1209,
    -0.1377, +0.2363, +0.1650, +0.1715, +0.1138, +0.1408,
]
REAL_MISMATCH_DELTAS = [
    +0.1305, +0.0546, +0.0119, -0.0912, +0.3383, +0.2623,
    +0.4346, +0.0017, -0.0269, +0.3825, +0.1652, +0.5776,
]


def _score(deltas, condition):
    from training.explainability.attention_capture import score_ablation

    return score_ablation([0.0] * len(deltas), deltas, condition=condition)


def test_ablation_passes_when_removing_the_image_reliably_hurts():
    result = _score([0.4, 0.5, 0.45, 0.6, 0.55, 0.5], "zeroed")
    assert result.established is True
    assert result.ci_low > 0
    assert result.fraction_worse == 1.0
    assert assert_visual_tokens_matter(result) is result


def test_ablation_raises_when_the_image_makes_no_difference():
    result = _score([0.0, 0.001, -0.001, 0.0, 0.002, -0.002], "zeroed")
    assert result.established is False
    with pytest.raises(SoftTokenAblationFailed, match="not using the image"):
        assert_visual_tokens_matter(result)


def test_ablation_raises_when_removing_the_image_HELPS():
    """A negative delta is at least as alarming as a zero one."""
    result = _score([-0.5, -0.4, -0.6, -0.45, -0.55, -0.5], "zeroed")
    assert result.established is False
    with pytest.raises(SoftTokenAblationFailed):
        assert_visual_tokens_matter(result)


def test_a_threshold_clearing_mean_with_a_ci_across_zero_is_NOT_a_pass():
    """The measurement that forced this gate to be rewritten.

    On 12 real studies the zero-ablation returned +0.0535, which clears the
    0.05 threshold, with a 95% CI of [-0.0283, +0.1314] and only 7 of 12
    studies worse. A threshold-only gate called that a pass. It is a null.
    """
    result = _score(REAL_ZERO_DELTAS, "zeroed")
    assert result.mean_delta == pytest.approx(0.0535, abs=0.001)
    assert result.mean_delta > 0.05          # clears the threshold
    assert result.ci_low < 0 < result.ci_high  # and is still indistinguishable from 0
    assert result.established is False
    with pytest.raises(SoftTokenAblationFailed, match="not established"):
        assert_visual_tokens_matter(result)


def test_the_mismatched_image_control_IS_established_on_the_same_studies():
    """Same 12 studies, sharper question, and this one clears zero."""
    result = _score(REAL_MISMATCH_DELTAS, "mismatched")
    assert result.mean_delta == pytest.approx(0.1868, abs=0.001)
    assert result.ci_low > 0
    assert result.fraction_worse == pytest.approx(10 / 12)
    assert result.established is True
    assert assert_visual_tokens_matter(result) is result


def test_the_bootstrap_is_deterministic():
    first = _score(REAL_MISMATCH_DELTAS, "mismatched")
    second = _score(REAL_MISMATCH_DELTAS, "mismatched")
    assert (first.ci_low, first.ci_high) == (second.ci_low, second.ci_high)


def test_ablation_rejects_too_few_studies():
    from training.explainability.attention_capture import score_ablation

    with pytest.raises(ValueError, match="at least 2 paired studies"):
        score_ablation([1.0], [2.0], condition="zeroed")


def test_assert_refuses_a_bare_number():
    with pytest.raises(TypeError, match="must come from score_ablation"):
        assert_visual_tokens_matter(0.5)


# --------------------------------------------------------------------------
# End-to-end attribution over the stand-in model
# --------------------------------------------------------------------------


def test_attribution_runs_over_captured_attention(model):
    with capture_attention(model) as captured:
        model(torch.zeros(1, 6, 4))
    attention = stack_captured(captured, expected_layers=3)
    span = locate_visual_tokens(
        torch.tensor([[262144, 262144, 9, 9, 9, 9]]), 262144,
        expected_count=2, source="medgemma_image_grid",
    )
    values, trace = attribute_visual_tokens(attention, span, query_positions=[5])
    assert values.shape == (2,)
    assert float(values.sum()) == pytest.approx(1.0)
    assert trace.gradient_weighted is False   # no gradients were passed


# --------------------------------------------------------------------------
# The Q-Former stage: declared, refused, and explained
# --------------------------------------------------------------------------


def test_qformer_cross_attention_raises_with_the_reason():
    with pytest.raises(QFormerCrossAttentionUnavailable, match="not in the"):
        qformer_cross_attention()


def test_soft_token_source_warns_that_it_is_not_a_heatmap():
    with pytest.warns(RuntimeWarning, match="NOT which image region"):
        warn_if_soft_token_source(SOURCE_QFORMER_SOFT_TOKEN)


def test_the_image_grid_source_does_not_warn():
    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("error")
        warn_if_soft_token_source("medgemma_image_grid")


# --------------------------------------------------------------------------
# The measured configuration is pinned
# --------------------------------------------------------------------------


def test_attention_implementation_is_eager_lm_and_sdpa_vision():
    """Both halves cost an OOM to discover; neither may drift."""
    assert ATTN_IMPLEMENTATION == {"text_config": "eager", "vision_config": "sdpa"}


# --------------------------------------------------------------------------
# gradient_weighted_layers -- shapes, which a stand-in model can still pin
# --------------------------------------------------------------------------


def test_gradients_come_back_shaped_like_the_attention(model):
    """Regression: the grads were wrapped a second time and came out 5-D.

    ``torch.autograd.grad`` already returns tensors mirroring its inputs, so
    each one is [B, H, S, S] and needs no extra axis. This went unnoticed on
    CPU because no test called the function -- it surfaced only on the GPU run.
    """
    from training.explainability.attention_capture import gradient_weighted_layers

    hidden = torch.zeros(1, 6, 4, requires_grad=True)
    with capture_attention(model) as captured:
        output = model(hidden)
    grads, reason = gradient_weighted_layers(output.sum(), captured, retain_graph=False)
    assert reason is None
    assert grads.shape == (3, 2, 6, 6)          # [L, H, S, S], matching the attention
    assert grads.dtype == torch.float32


def test_gradients_and_attention_fuse_without_a_shape_error(model):
    from training.explainability.attention_capture import gradient_weighted_layers

    hidden = torch.zeros(1, 6, 4, requires_grad=True)
    with capture_attention(model) as captured:
        output = model(hidden)
    attention = stack_captured(captured, expected_layers=3)
    grads, _ = gradient_weighted_layers(output.sum(), captured, retain_graph=False)
    assert grads.shape == attention.shape

    span = locate_visual_tokens(
        torch.tensor([[262144, 262144, 9, 9, 9, 9]]), 262144,
        expected_count=2, source="medgemma_image_grid",
    )
    values, trace = attribute_visual_tokens(attention, span, [5], gradients=grads)
    assert values.shape == (2,)
    assert trace.gradient_weighted is True


def test_visual_features_can_be_overridden_for_the_mismatch_control():
    """The mismatched-image control must go through the same scatter path.

    Zeroing asks "does anything visual matter"; substituting another study's
    features asks "does THIS image matter", which is the sharper question and
    the one a zero vector cannot pose -- zeros are out of distribution, not
    absent.
    """
    import inspect

    from training.explainability.attention_capture import (
        build_visual_inputs,
        teacher_forced_forward,
    )

    for fn in (build_visual_inputs, teacher_forced_forward):
        assert "visual_features" in inspect.signature(fn).parameters, fn.__name__
    # keyword-only, so it can never be passed by accident in positional order
    assert (
        inspect.signature(build_visual_inputs).parameters["visual_features"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )


# --------------------------------------------------------------------------
# The randomization sanity check (Adebayo et al. 2018)
# --------------------------------------------------------------------------


def test_spearman_of_a_map_with_itself_is_one():
    from training.explainability.attention_capture import spearman_correlation

    values = torch.rand(64)
    assert spearman_correlation(values, values) == pytest.approx(1.0)


def test_spearman_of_a_reversed_ranking_is_minus_one():
    from training.explainability.attention_capture import spearman_correlation

    values = torch.arange(32, dtype=torch.float32)
    assert spearman_correlation(values, -values) == pytest.approx(-1.0)


def test_spearman_is_invariant_to_a_monotone_rescale():
    """A rescaled map ranks regions identically, so it is NOT a degradation."""
    from training.explainability.attention_capture import spearman_correlation

    values = torch.rand(64)
    assert spearman_correlation(values, values * 7.5 + 3.0) == pytest.approx(1.0)


def test_spearman_against_a_constant_map_is_zero_not_one():
    # A constant map carries no ordering; reporting agreement would be false.
    from training.explainability.attention_capture import spearman_correlation

    assert spearman_correlation(torch.rand(32), torch.ones(32)) == 0.0


def test_spearman_averages_ties():
    from training.explainability.attention_capture import spearman_correlation

    a = torch.tensor([1.0, 1.0, 2.0, 2.0])
    assert spearman_correlation(a, torch.tensor([5.0, 5.0, 9.0, 9.0])) == pytest.approx(1.0)


def test_randomize_layers_changes_weights_and_restores_them(model):
    from training.explainability.attention_capture import randomize_layers

    module = model.language_model.layers[2].self_attn
    before = module.scale.detach().clone()
    restore = randomize_layers(model, [2])
    assert not torch.equal(module.scale, before)
    restore()
    assert torch.equal(module.scale, before)


def test_randomize_layers_touches_only_the_named_layers(model):
    from training.explainability.attention_capture import randomize_layers

    untouched = model.language_model.layers[0].self_attn.scale.detach().clone()
    restore = randomize_layers(model, [2])
    try:
        assert torch.equal(model.language_model.layers[0].self_attn.scale, untouched)
    finally:
        restore()


def test_randomize_layers_rejects_an_out_of_range_index(model):
    from training.explainability.attention_capture import randomize_layers

    with pytest.raises(IndexError, match="outside"):
        randomize_layers(model, [99])


def test_cascading_takes_layers_from_the_LAST_backwards():
    from training.explainability.attention_capture import cascading_randomization

    seen = []

    def attribute(indices):
        seen.append(tuple(indices))
        return torch.rand(16)

    cascading_randomization(attribute, num_layers=8, steps=[1, 2, 4])
    assert seen[0] == ()                 # the original comes first
    assert seen[1] == (7,)               # last layer
    assert seen[2] == (6, 7)
    assert seen[3] == (4, 5, 6, 7)


def test_randomization_passes_when_the_map_falls_apart():
    from training.explainability.attention_capture import (
        assert_randomization_degrades,
        cascading_randomization,
    )

    original = torch.arange(64, dtype=torch.float32)

    def attribute(indices):
        if not indices:
            return original
        # more layers randomised -> less of the original ordering survives
        noise = torch.randperm(64).to(torch.float32)
        weight = len(indices) / 8
        return original * (1 - weight) + noise * weight

    torch.manual_seed(0)
    result = cascading_randomization(attribute, num_layers=8)
    assert result.degrades is True
    assert abs(result.final_correlation) < 0.5
    assert assert_randomization_degrades(result) is result


def test_randomization_RAISES_when_the_map_is_unchanged():
    """The failure Adebayo et al. found: a map that ignores the weights."""
    from training.explainability.attention_capture import (
        RandomizationSanityFailed,
        assert_randomization_degrades,
        cascading_randomization,
    )

    fixed = torch.rand(64)
    result = cascading_randomization(lambda _indices: fixed, num_layers=8)
    assert result.final_correlation == pytest.approx(1.0)
    assert result.degrades is False
    with pytest.raises(RandomizationSanityFailed, match="does not depend on what the model learned"):
        assert_randomization_degrades(result)


def test_a_strongly_ANTI_correlated_map_also_fails():
    """|rho| is the test: a map that flips sign still tracks the weights' absence."""
    from training.explainability.attention_capture import cascading_randomization

    original = torch.arange(64, dtype=torch.float32)
    result = cascading_randomization(
        lambda indices: original if not indices else -original, num_layers=8
    )
    assert result.final_correlation == pytest.approx(-1.0)
    assert result.degrades is False


def test_default_steps_are_geometric_and_end_at_every_layer():
    from training.explainability.attention_capture import cascading_randomization

    result = cascading_randomization(lambda _i: torch.rand(16), num_layers=34)
    assert result.steps == (1, 2, 4, 8, 16, 34)


def test_assert_refuses_something_that_is_not_a_result():
    from training.explainability.attention_capture import assert_randomization_degrades

    with pytest.raises(TypeError, match="must come from cascading_randomization"):
        assert_randomization_degrades(0.1)
