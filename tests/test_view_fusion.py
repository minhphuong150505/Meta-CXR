"""Unit tests for ViewFusionModule. Run with `python -m tests.test_view_fusion`.

No GPU and no dataset required.
"""
import torch

from mhcac.view_fusion import ViewFusionModule, real_aux_rows, scatter_aux_rows


B, P, D = 3, 7, 32


def make_module(**kw):
    torch.manual_seed(0)
    return ViewFusionModule(dim=D, num_heads=4, ffn_ratio=2, **kw)


def test_zero_init_identity():
    """Checklist 2: at step 0 the module is an exact identity on the anchor."""
    m = make_module(p_view_drop=0.0).eval()
    anchor = torch.randn(B, P, D)
    aux = torch.randn(B, 2, P, D)
    aux_mask = torch.ones(B, 2, dtype=torch.bool)
    out = m(anchor, aux, aux_mask,
            anchor_view_id=torch.tensor([0, 1, 2]),
            aux_view_ids=torch.tensor([[1, 2], [0, 3], [3, 1]]))
    max_diff = (out - anchor).abs().max().item()
    assert max_diff < 1e-6, max_diff
    print(f"ok: zero-init identity (max diff {max_diff:.2e})")


def test_no_aux_returns_anchor():
    """Checklist 3: an all-False aux_mask returns the anchor, no NaN."""
    m = make_module(p_view_drop=0.0).eval()
    # Break the zero init so the test proves the gate, not the init.
    for blk in m.blocks:
        torch.nn.init.normal_(blk.w_o.weight, std=0.5)
        torch.nn.init.normal_(blk.w_o.bias, std=0.5)
        torch.nn.init.normal_(blk.ffn[-1].weight, std=0.5)
        torch.nn.init.normal_(blk.ffn[-1].bias, std=0.5)

    anchor = torch.randn(B, P, D)
    aux = torch.randn(B, 2, P, D)
    aux_mask = torch.zeros(B, 2, dtype=torch.bool)
    out = m(anchor, aux, aux_mask)
    assert torch.isfinite(out).all(), "non-finite output for the n=0 case"
    assert torch.allclose(out, anchor, atol=1e-6), (out - anchor).abs().max()

    # Mixed batch: only row 1 has an auxiliary view.
    aux_mask = torch.tensor([[False, False], [True, False], [False, False]])
    out = m(anchor, aux, aux_mask)
    assert torch.isfinite(out).all()
    assert torch.allclose(out[0], anchor[0], atol=1e-6)
    assert torch.allclose(out[2], anchor[2], atol=1e-6)
    assert not torch.allclose(out[1], anchor[1], atol=1e-6), \
        "row with an auxiliary view should have been modified"
    print("ok: n=0 gate returns anchor exactly, finite, batch-dense")


def test_shapes():
    """Checklist 4: [B,P,D] in -> [B,P,D] out for N in {0,1,3} and mixed counts."""
    m = make_module(p_view_drop=0.0).eval()
    anchor = torch.randn(B, P, D)
    for n in (0, 1, 3):
        aux = torch.randn(B, n, P, D)
        aux_mask = torch.ones(B, n, dtype=torch.bool)
        out = m(anchor, aux, aux_mask)
        assert out.shape == (B, P, D), (n, out.shape)

    # Per-sample counts 2 / 1 / 0, padded to N_max = 2.
    aux = torch.randn(B, 2, P, D)
    aux_mask = torch.tensor([[True, True], [True, False], [False, False]])
    out = m(anchor, aux, aux_mask)
    assert out.shape == (B, P, D)
    assert torch.isfinite(out).all()
    print("ok: shape contract holds for N in {0,1,3} and ragged counts")


def test_gradient_reaches_kv():
    """Gradient flows into the fusion weights despite the detached aux features.

    At exactly step 0 the zero-init W_O makes dL/dW_K and dL/dW_V vanish -- W_O
    is the only weight in the attention branch that moves. Once W_O is nonzero
    (i.e. from step 1 on) K/V start learning, so both facts are asserted.
    """
    anchor = torch.randn(B, P, D)
    aux = torch.randn(B, 2, P, D).detach()  # frozen-encoder output, no_grad
    aux_mask = torch.ones(B, 2, dtype=torch.bool)

    m = make_module(p_view_drop=0.0).train()
    m(anchor, aux, aux_mask).sum().backward()
    g_o = m.blocks[0].w_o.weight.grad
    assert g_o is not None and g_o.abs().sum() > 0, "W_O received no gradient at init"

    m = make_module(p_view_drop=0.0).train()
    torch.nn.init.normal_(m.blocks[0].w_o.weight, std=0.1)  # simulate step >= 1
    m(anchor, aux, aux_mask).sum().backward()
    for name in ("w_k", "w_v"):
        g = getattr(m.blocks[0], name).weight.grad
        assert g is not None and g.abs().sum() > 0, f"{name} received no gradient"
    print("ok: W_O learns at step 0; W_K / W_V learn once W_O is nonzero")


def test_view_dropout_train_only():
    """Checklist 6: view dropout is active in train() and inert in eval().

    p_view_drop=1.0 makes this deterministic: under train() every auxiliary is
    dropped, so the n=0 gate returns the anchor; under eval() the views survive
    and the output must differ.
    """
    m = make_module(p_view_drop=1.0, dropout=0.0)
    for blk in m.blocks:  # break the zero init so fusion has a visible effect
        torch.nn.init.normal_(blk.w_o.weight, std=0.5)
    anchor = torch.randn(B, P, D)
    aux = torch.randn(B, 4, P, D)
    aux_mask = torch.ones(B, 4, dtype=torch.bool)

    m.train()
    out_train = m(anchor, aux, aux_mask)
    assert torch.allclose(out_train, anchor, atol=1e-6), \
        "train() with p_view_drop=1.0 should drop every view and return the anchor"

    m.eval()
    out_eval = m(anchor, aux, aux_mask)
    assert not torch.allclose(out_eval, anchor, atol=1e-6), \
        "eval() must ignore view dropout and actually fuse"
    print("ok: view dropout active under train(), inert under eval()")


def test_real_aux_rows_selects_only_padded_slots():
    """The filter keeps exactly the real views and nothing else."""
    aux_mask = torch.tensor([[True, False], [False, False], [True, True]])
    keep = real_aux_rows(aux_mask, total=6, device=torch.device("cpu"))
    assert keep is not None
    assert keep.tolist() == [True, False, False, False, True, True]
    assert int(keep.sum()) == 3
    print("ok: real_aux_rows selects only the real auxiliary slots")


def test_real_aux_rows_returns_none_when_nothing_to_filter():
    """A fully real mask, or no mask at all, keeps the dense path."""
    full = torch.ones(2, 3, dtype=torch.bool)
    assert real_aux_rows(full, total=6, device=torch.device("cpu")) is None
    assert real_aux_rows(None, total=6, device=torch.device("cpu")) is None
    print("ok: real_aux_rows is a no-op when every slot is real")


def test_scatter_aux_rows_round_trip_leaves_padding_at_zero():
    """Encoding only the real rows reproduces the dense result exactly.

    This is the invariant the auxiliary-encoder filter rests on: a padded slot
    is all-zero input whose output ViewFusionModule gates away, so replacing it
    with literal zeros must be indistinguishable downstream.
    """
    Bx, Nx, Px, Dx = 3, 2, 5, 4
    aux_mask = torch.tensor([[True, False], [False, False], [True, True]])
    keep = real_aux_rows(aux_mask, total=Bx * Nx, device=torch.device("cpu"))

    encoded_real = torch.randn(int(keep.sum()), Px, Dx)
    out = scatter_aux_rows(encoded_real, keep, Bx, Nx)

    assert out.shape == (Bx, Nx, Px, Dx)
    torch.testing.assert_close(out.reshape(Bx * Nx, Px, Dx)[keep], encoded_real)
    assert out.reshape(Bx * Nx, Px, Dx)[~keep].abs().max().item() == 0.0
    print("ok: scatter_aux_rows round-trips and zeroes padding")


def test_filtered_aux_fuses_identically_to_dense_aux():
    """End-to-end: fusion output is unchanged by dropping the padded rows.

    Stands in for the frozen encoders -- a padded slot carries *some* value in
    the dense path (the encoder's response to an all-zero image) and exactly
    zero in the filtered path. The fused anchor must not be able to tell.
    """
    m = make_module(p_view_drop=0.0, dropout=0.0).eval()
    for blk in m.blocks:  # break the zero init so fusion actually does something
        torch.nn.init.normal_(blk.w_o.weight, std=0.5)
        torch.nn.init.normal_(blk.ffn[-1].weight, std=0.5)

    N = 2
    anchor = torch.randn(B, P, D)
    aux_mask = torch.tensor([[True, False], [False, False], [True, True]])
    aux_view_ids = torch.tensor([[1, 0], [0, 0], [2, 3]])
    keep = real_aux_rows(aux_mask, total=B * N, device=torch.device("cpu"))

    real = torch.randn(int(keep.sum()), P, D)
    filtered = scatter_aux_rows(real, keep, B, N)

    # Dense path: padded slots hold arbitrary junk instead of zeros.
    dense = filtered.reshape(B * N, P, D).clone()
    dense[~keep] = torch.randn(int((~keep).sum()), P, D)
    dense = dense.reshape(B, N, P, D)

    out_filtered = m(anchor, filtered, aux_mask, torch.tensor([0, 1, 2]), aux_view_ids)
    out_dense = m(anchor, dense, aux_mask, torch.tensor([0, 1, 2]), aux_view_ids)

    torch.testing.assert_close(out_filtered, out_dense)
    # And study 1, which has no real view at all, still gets the anchor back.
    torch.testing.assert_close(out_filtered[1], anchor[1])
    print("ok: filtered auxiliary rows fuse identically to the dense batch")


if __name__ == "__main__":
    test_zero_init_identity()
    test_no_aux_returns_anchor()
    test_shapes()
    test_gradient_reaches_kv()
    test_view_dropout_train_only()
    test_real_aux_rows_selects_only_padded_slots()
    test_real_aux_rows_returns_none_when_nothing_to_filter()
    test_scatter_aux_rows_round_trip_leaves_padding_at_zero()
    test_filtered_aux_fuses_identically_to_dense_aux()
    print("\nall view-fusion tests passed")
