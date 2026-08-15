"""Unit tests for the multi-view auxiliary losses.

Run with `python -m tests.test_multiview_losses`. No GPU required.
"""
import torch

from mhcac.loss import MultiPositiveContrastiveLoss, view_consistency_loss
from vision_encoders.stream_adapter import (
    ContrastiveProjectionHead,
    StreamAdapter,
    pool_stream,
)


B, N, P, D = 4, 2, 5, 16


def test_mpc_zero_without_aux():
    """No auxiliary view anywhere -> no positives -> exactly zero, no NaN."""
    loss_fn = MultiPositiveContrastiveLoss()
    anchor = torch.randn(B, D)
    aux = torch.randn(B, N, D)
    out = loss_fn(anchor, aux, torch.zeros(B, N, dtype=torch.bool))
    assert torch.isfinite(out) and out.item() == 0.0, out
    assert loss_fn(anchor, torch.zeros(B, 0, D), torch.zeros(B, 0, dtype=torch.bool)).item() == 0.0
    print("ok: MPC is exactly zero when no study has an auxiliary view")


def test_mpc_rewards_same_study_similarity():
    """Aux views that match their own anchor score a lower loss than shuffled ones."""
    loss_fn = MultiPositiveContrastiveLoss()
    torch.manual_seed(0)
    anchor = torch.randn(B, D)
    aux_mask = torch.ones(B, N, dtype=torch.bool)

    aligned = anchor.unsqueeze(1).repeat(1, N, 1) + 0.01 * torch.randn(B, N, D)
    shuffled = aligned[torch.tensor([1, 2, 3, 0])]

    l_aligned = loss_fn(anchor, aligned, aux_mask)
    l_shuffled = loss_fn(anchor, shuffled, aux_mask)
    assert torch.isfinite(l_aligned) and torch.isfinite(l_shuffled)
    assert l_aligned < l_shuffled, (l_aligned.item(), l_shuffled.item())
    print(
        f"ok: MPC lower for same-study positives "
        f"({l_aligned.item():.4f} < {l_shuffled.item():.4f})"
    )


def test_mpc_partial_mask_is_finite():
    """Ragged per-study auxiliary counts stay finite and differentiable."""
    loss_fn = MultiPositiveContrastiveLoss()
    anchor = torch.randn(B, D, requires_grad=True)
    aux = torch.randn(B, N, D)
    aux_mask = torch.tensor([[True, True], [True, False], [False, False], [True, False]])
    out = loss_fn(anchor, aux, aux_mask)
    assert torch.isfinite(out), out
    out.backward()
    assert anchor.grad is not None and torch.isfinite(anchor.grad).all()
    print("ok: MPC handles ragged masks, stays finite, backprops")


def test_view_consistency_zero_when_identical():
    """Identical logits -> zero divergence; different logits -> positive."""
    logits = torch.randn(B, 14, 3)
    has_aux = torch.tensor([True, True, False, False])
    same = view_consistency_loss(logits, logits.clone(), has_aux)
    assert abs(same.item()) < 1e-6, same

    other = view_consistency_loss(logits, torch.randn(B, 14, 3), has_aux)
    assert other.item() > 0, other

    none_ = view_consistency_loss(logits, torch.randn(B, 14, 3),
                                  torch.zeros(B, dtype=torch.bool))
    assert none_.item() == 0.0, none_
    print("ok: view-consistency zero on identical logits, positive otherwise, "
          "zero when no study has auxiliaries")


def test_view_consistency_symmetric():
    a, b = torch.randn(B, 14, 3), torch.randn(B, 14, 3)
    has_aux = torch.ones(B, dtype=torch.bool)
    ab = view_consistency_loss(a, b, has_aux)
    ba = view_consistency_loss(b, a, has_aux)
    assert torch.allclose(ab, ba, atol=1e-6), (ab.item(), ba.item())
    print("ok: view-consistency is symmetric in its two arguments")


def test_view_consistency_defaults_reproduce_legacy_value():
    """The soft/conditional knobs are opt-in: defaults must be bit-compatible.

    Without this, turning the feature on becomes unfalsifiable — there would be
    no way to reproduce the previous run for ablation.
    """
    torch.manual_seed(3)
    a, b = torch.randn(B, 14, 3), torch.randn(B, 14, 3)
    has_aux = torch.ones(B, dtype=torch.bool)

    p = torch.nn.functional.log_softmax(a, dim=-1)
    q = torch.nn.functional.log_softmax(b, dim=-1)
    kl_pq = torch.nn.functional.kl_div(q, p, log_target=True, reduction="none").sum(-1)
    kl_qp = torch.nn.functional.kl_div(p, q, log_target=True, reduction="none").sum(-1)
    legacy = (0.5 * (kl_pq + kl_qp)).mean()

    assert torch.allclose(view_consistency_loss(a, b, has_aux), legacy, atol=1e-7)
    print("ok: default arguments reproduce the legacy symmetric-KL value exactly")


def test_view_consistency_margin_forgives_small_drift():
    """Divergence under the margin costs nothing; above it, only the excess."""
    torch.manual_seed(4)
    a = torch.randn(B, 14, 3)
    b = a + 0.01 * torch.randn(B, 14, 3)          # nearly identical
    has_aux = torch.ones(B, dtype=torch.bool)

    raw = view_consistency_loss(a, b, has_aux)
    assert raw.item() > 0, raw
    forgiven = view_consistency_loss(a, b, has_aux, margin=1.0)
    assert forgiven.item() == 0.0, forgiven

    far = view_consistency_loss(a, torch.randn(B, 14, 3), has_aux, margin=0.05)
    assert far.item() > 0, far
    print("ok: margin forgives small drift and still charges real disagreement")


def test_view_consistency_gate_waives_when_fusion_is_more_confident():
    """A sharper fused prediction reads as new evidence, not as contradiction."""
    # anchor: near-uniform (unsure).  fused: sharply peaked (confident).
    anchor = torch.zeros(1, 1, 3)
    fused = torch.tensor([[[8.0, 0.0, 0.0]]])
    has_aux = torch.ones(1, dtype=torch.bool)

    ungated = view_consistency_loss(fused, anchor, has_aux)
    assert ungated.item() > 0, ungated

    gated = view_consistency_loss(fused, anchor, has_aux, confidence_gate=True)
    assert gated.item() == 0.0, gated

    # Reverse it: fusion *smeared* a confident anchor -> still charged.
    smeared = view_consistency_loss(anchor, fused, has_aux, confidence_gate=True)
    assert smeared.item() > 0, smeared
    print("ok: gate waives sharpening, still charges smearing")


def test_view_consistency_gate_is_detached_but_loss_still_trains():
    """Gradient must reach the logits; the gate itself must not carry any."""
    torch.manual_seed(5)
    fused = torch.randn(B, 14, 3, requires_grad=True)
    anchor = torch.randn(B, 14, 3)
    has_aux = torch.ones(B, dtype=torch.bool)

    out = view_consistency_loss(
        fused, anchor, has_aux, margin=0.01, confidence_gate=True
    )
    out.backward()
    assert fused.grad is not None and torch.isfinite(fused.grad).all()
    assert fused.grad.abs().sum() > 0, "loss produced no gradient at all"
    print("ok: gated loss stays differentiable w.r.t. the fused logits")


def test_view_consistency_rejects_negative_knobs():
    a, b = torch.randn(B, 14, 3), torch.randn(B, 14, 3)
    has_aux = torch.ones(B, dtype=torch.bool)
    for kwargs in ({"margin": -0.1}, {"gate_tolerance": -0.1}):
        try:
            view_consistency_loss(a, b, has_aux, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kwargs}")
    print("ok: negative margin / gate_tolerance are rejected")


if __name__ == "__main__":
    test_mpc_zero_without_aux()
    test_mpc_rewards_same_study_similarity()
    test_mpc_partial_mask_is_finite()
    test_view_consistency_zero_when_identical()
    test_view_consistency_symmetric()
    test_view_consistency_defaults_reproduce_legacy_value()
    test_view_consistency_margin_forgives_small_drift()
    test_view_consistency_gate_waives_when_fusion_is_more_confident()
    test_view_consistency_gate_is_detached_but_loss_still_trains()
    test_view_consistency_rejects_negative_knobs()
    print("\nall multi-view loss tests passed")


# --------------------------------------------------------------------------
# The regression that made MPC a constant for an entire run
# --------------------------------------------------------------------------

def test_adapter_is_exactly_identity_at_initialisation():
    """Turning this on must not perturb a freshly built model."""
    adapter = StreamAdapter(dim=16)
    tokens = torch.randn(2, 5, 16)
    assert torch.allclose(adapter(tokens), tokens, atol=0), "not identity at init"


def test_adapter_stops_being_identity_once_trained():
    adapter = StreamAdapter(dim=16)
    torch.nn.init.normal_(adapter.up.weight, std=0.1)
    tokens = torch.randn(2, 5, 16)
    assert not torch.allclose(adapter(tokens), tokens, atol=1e-6)


def test_projection_head_returns_unit_vectors():
    head = ContrastiveProjectionHead(dim=16, hidden_dim=8, output_dim=4)
    out = head(torch.randn(3, 16))
    assert out.shape == (3, 4)
    assert torch.allclose(out.norm(dim=-1), torch.ones(3), atol=1e-5)


def test_pool_stream_picks_cls_or_mean():
    tokens = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    assert torch.equal(pool_stream(tokens, has_cls_token=True), tokens[:, 0, :])
    assert torch.allclose(pool_stream(tokens, has_cls_token=False), tokens.mean(dim=1))


def test_mpc_gradient_reaches_the_adapter_through_frozen_features():
    """The test that would have caught the dead loss.

    MPC used to be computed on raw frozen-encoder output stashed before the only
    trainable module, so it had no parameter upstream and its value sat at
    3.994 +/- 0.001 for four epochs while carrying 0.1 of the loss weight. What
    must hold now is that a gradient reaches the adapter and the head, *starting
    from tensors that do not require grad* -- exactly the frozen-encoder case.
    """
    frozen_anchor = torch.randn(B, P, D)                 # requires_grad False
    frozen_aux = torch.randn(B, N, P, D)                 # ditto, and no_grad in real use
    assert not frozen_anchor.requires_grad and not frozen_aux.requires_grad

    adapter = StreamAdapter(dim=D)
    torch.nn.init.normal_(adapter.up.weight, std=0.02)   # leave the identity init
    head = ContrastiveProjectionHead(dim=D, hidden_dim=8, output_dim=4)

    anchor_vec = head(pool_stream(adapter(frozen_anchor), has_cls_token=False))
    aux_vec = head(pool_stream(adapter(frozen_aux), has_cls_token=False))

    loss = MultiPositiveContrastiveLoss()(
        anchor_vec, aux_vec, torch.ones(B, N, dtype=torch.bool)
    )
    loss.backward()

    assert adapter.down.weight.grad is not None
    assert adapter.down.weight.grad.abs().sum() > 0, "adapter received no gradient"
    assert head.net[1].weight.grad.abs().sum() > 0, "head received no gradient"
