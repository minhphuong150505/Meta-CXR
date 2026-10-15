"""Q-Former input norm and the query-diversity penalty (2026-10-15, soft-token collapse fix).

The two ways the penalty could pass a collapsed Q-Former are pinned here: a constant offset
per query, and one informative query beside 31 near-constant ones. Both must score as
collapsed (loss ~1), or the refit could report "diverse" soft tokens that carry one direction
of image information, which is the state it exists to fix.

A third way was found on the GPU (2026-10-15): computed on training-mode states, the penalty
reads dropout noise as diversity, and the first refit learned to amplify that noise. That
mechanism, and the knob and guard that close it, are pinned at the end.
"""

from __future__ import annotations

import math

import pytest
import torch

from vision_encoders.qformer_inputs import (
    QformerInputNorm,
    assert_input_norm_keys_match,
    assert_noise_free_query_path,
    batch_centred_participation_ratio,
    query_diversity_loss,
    resolve_qformer_dropout,
    resolve_qformer_input_norm,
    set_dropout_probability,
)
from vision_encoders.shared_visual_tokens import SharedVisualTokens

B, Q, D = 16, 32, 48


def _shared(batch=2, dim=8):
    torch.manual_seed(0)
    spans = {"biovil": slice(0, 3), "pubmedclip": slice(3, 5), "swin": slice(5, 9)}
    tokens = torch.randn(batch, 9, dim) * torch.tensor([1.0] * 3 + [40.0] * 2 + [5.0] * 4)[None, :, None]
    return SharedVisualTokens(tokens=tokens, spans=spans)


def test_resolve_modes():
    assert resolve_qformer_input_norm(None) is None
    assert resolve_qformer_input_norm("per_stream") == "per_stream"
    with pytest.raises(ValueError):
        resolve_qformer_input_norm("layernorm")


def test_norm_is_one_layernorm_per_stream_and_leaves_the_shared_tokens_alone():
    shared = _shared()
    before = shared.tokens.clone()
    norm = QformerInputNorm(("biovil", "pubmedclip", "swin"), visual_dim=8)
    with torch.no_grad():
        for i, name in enumerate(norm.stream_names):  # distinct affine per stream
            norm.norms[name].weight.fill_(1.0 + i)
            norm.norms[name].bias.fill_(0.1 * i)
    out = norm(shared)
    assert out.shape == shared.tokens.shape
    for i, name in enumerate(norm.stream_names):
        span = shared.spans[name]
        expected = torch.nn.functional.layer_norm(
            shared.tokens[:, span], (8,), weight=torch.full((8,), 1.0 + i), bias=torch.full((8,), 0.1 * i))
        assert torch.allclose(out[:, span], expected, atol=1e-6)
    assert torch.equal(shared.tokens, before), "MHCAC's copy must not be modified"


def test_at_initialisation_no_stream_dominates_by_magnitude():
    shared = _shared()  # pubmedclip tokens carry 40x the scale of biovil's
    out = QformerInputNorm(("biovil", "pubmedclip", "swin"), visual_dim=8)(shared)
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.full_like(norms, math.sqrt(8)), rtol=1e-3)


def test_norm_refuses_a_different_stream_set_or_a_gap():
    shared = _shared()
    with pytest.raises(ValueError, match="built for"):
        QformerInputNorm(("biovil", "pubmedclip"), 8)(shared)
    gapped = SharedVisualTokens(tokens=shared.tokens, spans={
        "biovil": slice(0, 3), "pubmedclip": slice(4, 5), "swin": slice(5, 9)})
    with pytest.raises(ValueError, match="contiguous"):
        QformerInputNorm(("biovil", "pubmedclip", "swin"), 8)(gapped)


def test_norm_trains():
    shared = _shared()
    norm = QformerInputNorm(("biovil", "pubmedclip", "swin"), 8)
    norm(shared).pow(2).mean().backward()
    assert all(p.grad is not None for p in norm.parameters())


def _image_part(seed=1):
    return torch.randn(B, 1, D, generator=torch.Generator().manual_seed(seed))


def test_collapsed_queries_score_one():
    states = _image_part().expand(B, Q, D)  # 32 identical queries, varying across studies
    assert float(query_diversity_loss(states)) == pytest.approx(1.0, abs=1e-5)


def test_constant_offsets_per_query_do_not_count_as_diversity():
    offsets = torch.randn(1, Q, D, generator=torch.Generator().manual_seed(2)) * 50.0
    states = offsets + _image_part()
    assert float(query_diversity_loss(states)) == pytest.approx(1.0, abs=1e-4)


def test_one_informative_query_beside_constant_ones_counts_as_collapsed():
    g = torch.Generator().manual_seed(3)
    states = torch.randn(1, Q, D, generator=g).expand(B, Q, D).clone()
    states[:, 0] = torch.randn(B, D, generator=g)
    states = states + 1e-4 * torch.randn(B, Q, D, generator=g)  # dropout-like jitter
    assert float(query_diversity_loss(states)) > 0.99


def test_independent_queries_score_low_and_the_loss_is_scale_free():
    states = torch.randn(B, Q, D, generator=torch.Generator().manual_seed(4))
    loss = float(query_diversity_loss(states))
    assert loss < 0.25
    assert float(query_diversity_loss(7.5 * states)) == pytest.approx(loss, abs=1e-5)
    pr = batch_centred_participation_ratio(states)
    assert pr.shape == (B,) and float(pr.min()) >= 1.0 and float(pr.max()) <= Q


def test_a_single_study_gives_a_zero_that_backpropagates():
    states = torch.randn(1, Q, D, requires_grad=True)
    loss = query_diversity_loss(states)
    assert float(loss.detach()) == 0.0
    loss.backward()
    assert states.grad is not None and float(states.grad.abs().sum()) == 0.0


def test_a_gradient_step_breaks_partial_collapse():
    g = torch.Generator().manual_seed(5)
    base = _image_part(6).expand(B, Q, D)
    states = (base + 0.05 * torch.randn(B, Q, D, generator=g)).clone().requires_grad_(True)
    loss0 = query_diversity_loss(states)
    loss0.backward()
    assert torch.isfinite(states.grad).all() and float(states.grad.abs().sum()) > 0
    with torch.no_grad():
        stepped = states - 0.5 * states.grad / states.grad.norm() * states.norm()
    assert float(query_diversity_loss(stepped)) < float(loss0.detach())
    assert math.isfinite(float(loss0.detach()))


def test_norm_keys_must_agree_between_model_and_checkpoint():
    with_norm = ["Qformer.x", "qformer_input_norm.norms.biovil.weight"]
    without = ["Qformer.x"]
    assert_input_norm_keys_match(with_norm, with_norm)
    assert_input_norm_keys_match(without, without)
    with pytest.raises(ValueError, match="built without it"):
        assert_input_norm_keys_match(without, with_norm)
    with pytest.raises(ValueError, match="LayerNorm initialisation"):
        assert_input_norm_keys_match(with_norm, without)


# --------------------------------------------------------------------------
# Training-only noise on the query path (2026-10-15 refit)
# --------------------------------------------------------------------------


def test_dropout_noise_reads_as_diversity():
    """Why the knob exists: collapsed states plus per-element dropout noise score as diverse."""
    collapsed = _image_part(7).expand(B, Q, D).contiguous()
    assert float(query_diversity_loss(collapsed)) == pytest.approx(1.0, abs=1e-5)
    # A shared component 12x the image part gives dropout noise ~4x the image-dependent part,
    # about what the refit learned (3.9x); the 32 identical queries then score as diverse.
    states = collapsed + 12.0
    torch.manual_seed(8)
    noisy = torch.nn.functional.dropout(states, p=0.1, training=True)
    ratio = (noisy - states).flatten(1).norm(dim=1) / (collapsed - collapsed.mean(0)).flatten(1).norm(dim=1)
    assert 3.5 < float(ratio.mean()) < 5.0
    assert float(query_diversity_loss(noisy)) < 0.25


def test_resolve_qformer_dropout():
    assert resolve_qformer_dropout(None) is None
    assert resolve_qformer_dropout(0) == 0.0 and resolve_qformer_dropout("0.05") == 0.05
    for bad in (1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="qformer_dropout"):
            resolve_qformer_dropout(bad)


def _with_dropout():
    torch.manual_seed(9)
    return torch.nn.Sequential(
        torch.nn.Linear(D, D), torch.nn.Dropout(0.1),
        torch.nn.Sequential(torch.nn.Linear(D, D), torch.nn.Dropout(0.1)),
    )


def test_zero_dropout_makes_the_training_forward_the_eval_one_without_leaving_train_mode():
    module = _with_dropout()
    x = torch.randn(4, D)
    module.train()
    assert not torch.equal(module(x), module(x)), "dropout should be live before the knob"
    assert set_dropout_probability(module, 0.0) == 2
    assert module.training  # gradient checkpointing is gated on self.training
    trained = module(x)
    module.eval()
    assert torch.equal(trained, module(x))


def test_set_dropout_probability_refuses_a_module_without_dropout():
    with pytest.raises(ValueError, match="no dropout layer"):
        set_dropout_probability(torch.nn.Linear(D, D), 0.0)


def test_the_diversity_term_refuses_a_noisy_query_path():
    assert_noise_free_query_path(0.0, None, 0.1)       # term off: anything goes
    assert_noise_free_query_path(1.0, 0.0, 0.0)        # the 1q configuration
    with pytest.raises(ValueError, match=r"qformer_dropout: 0\.0") as err:
        assert_noise_free_query_path(1.0, None, 0.0)   # None = BERT's 0.1
    assert "feature_mask_ratio" not in str(err.value)
    with pytest.raises(ValueError, match=r"qformer_dropout: 0\.0 and model\.feature_mask_ratio"):
        assert_noise_free_query_path(1.0, 0.1, 0.1)
    with pytest.raises(ValueError, match="feature_mask_ratio"):
        assert_noise_free_query_path(1.0, 0.0, 0.1)
