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


def test_ablation_passes_when_removing_the_image_hurts():
    delta = assert_visual_tokens_matter([1.0, 1.0, 1.0], [2.0, 2.5, 1.5])
    assert delta == pytest.approx(1.0)


def test_ablation_raises_when_the_image_makes_no_difference():
    with pytest.raises(SoftTokenAblationFailed, match="not using the image"):
        assert_visual_tokens_matter([1.0, 1.0], [1.0, 1.0001])


def test_ablation_raises_when_removing_the_image_HELPS():
    # A negative delta is at least as alarming as a zero one.
    with pytest.raises(SoftTokenAblationFailed):
        assert_visual_tokens_matter([2.0, 2.0], [1.0, 1.0])


def test_ablation_ignores_nan_positions_consistently():
    delta = assert_visual_tokens_matter(
        [1.0, float("nan"), 1.0], [2.0, float("nan"), 2.0]
    )
    assert delta == pytest.approx(1.0)


def test_ablation_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="matching non-empty"):
        assert_visual_tokens_matter([1.0, 2.0], [1.0])


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
