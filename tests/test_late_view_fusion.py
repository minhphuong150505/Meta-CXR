"""Late multi-view fusion (``model.view_fusion.mode: late``, 2026-10-26).

Pins the guarantees of docs/handoff/PLAN-2026-10-26-late-view-fusion.md that a
CPU box can check: a single-view study gets exactly its anchor logits, padded
slots are never read, zero-init is the mean of the real views, the refusals,
the loss of a single-view study is the OFF loss, and the default (absent key)
path builds nothing new. Also the aux-pass gradient split of LD
(``late_detach_aux``) and of the separate aux head (``late_aux_head``,
docs/handoff/PLAN-2026-10-27-late-aux-head.md), on tiny stand-in modules.
No GPU and no dataset required.
"""
from pathlib import Path

import pytest
import torch
import yaml

from mhcac.late_view_pool import (
    LateViewPooling,
    late_aux_head_loss,
    late_aux_view_logits,
    late_classification_loss,
    resolve_view_fusion_mode,
)
from mhcac.loss import ClassificationLoss
from mhcac.view_fusion import real_aux_rows

_ROOT = Path(__file__).resolve().parents[1]
_CONFIG = _ROOT / "pretraining" / "configs" / "mimic_cxr_full.yaml"
_QFORMER = _ROOT / "model" / "lavis" / "models" / "blip2_models" / "blip2_qformer.py"
_TRAIN = _ROOT / "pretraining" / "train.py"

B, N, F, C = 4, 2, 14, 3


def _pool(trained=True):
    torch.manual_seed(0)
    pool = LateViewPooling(num_findings=F, num_classes=C, num_view_types=4)
    if trained:
        # Past step 0, so the tests do not pass merely because the weights are zero.
        torch.nn.init.normal_(pool.weight, std=1.0)
        torch.nn.init.normal_(pool.bias, std=1.0)
    return pool


def _inputs(seed=1):
    torch.manual_seed(seed)
    anchor = torch.randn(B, F, C)
    aux = torch.randn(B, N, F, C)
    # study 0: no aux; study 1: one aux; study 2: two; study 3: second slot only
    aux_mask = torch.tensor([[False, False], [True, False], [True, True], [False, True]])
    anchor_view_id = torch.tensor([0, 1, 0, 1])
    aux_view_ids = torch.tensor([[0, 0], [2, 0], [2, 3], [0, 2]])
    return anchor, aux, aux_mask, anchor_view_id, aux_view_ids


def test_a_single_view_study_gets_its_anchor_logits_bit_for_bit():
    pool = _pool()
    anchor, aux, aux_mask, av, xv = _inputs()
    out = pool(anchor, aux, aux_mask, av, xv)
    assert torch.equal(out[0], anchor[0])
    # whole batch without an aux view
    none = torch.zeros(B, N, dtype=torch.bool)
    assert torch.equal(pool(anchor, aux, none, av, xv), anchor)


def test_junk_in_padded_slots_is_never_read():
    pool = _pool()
    anchor, aux, aux_mask, av, xv = _inputs()
    out = pool(anchor, aux, aux_mask, av, xv)
    junk = aux.clone()
    junk[~aux_mask] = float("nan")
    junk_ids = xv.clone()
    junk_ids[~aux_mask] = 3
    out_junk = pool(anchor, junk, aux_mask, av, junk_ids)
    assert torch.equal(out, out_junk)
    assert torch.isfinite(out_junk).all()


def test_zero_init_is_the_mean_of_the_real_views():
    pool = _pool(trained=False)
    anchor, aux, aux_mask, av, xv = _inputs()
    out = pool(anchor, aux, aux_mask, av, xv)
    assert torch.allclose(out[1], (anchor[1] + aux[1, 0]) / 2, atol=1e-6)
    assert torch.allclose(out[2], (anchor[2] + aux[2, 0] + aux[2, 1]) / 3, atol=1e-6)
    assert torch.allclose(out[3], (anchor[3] + aux[3, 1]) / 2, atol=1e-6)


def test_weights_sum_to_one_over_real_views_and_vanish_on_padding():
    pool = _pool()
    anchor, aux, aux_mask, av, xv = _inputs()
    w = pool.weights(anchor, aux, aux_mask, av, xv)            # [B, 1 + N, F]
    assert w.shape == (B, 1 + N, F)
    assert torch.allclose(w.sum(dim=1), torch.ones(B, F), atol=1e-6)
    padded = torch.cat([torch.zeros(B, 1, dtype=torch.bool), ~aux_mask], dim=1)
    assert (w[padded] == 0).all()


def test_the_weights_depend_on_view_type_and_confidence():
    pool = _pool()
    anchor, aux, aux_mask, av, xv = _inputs()
    w = pool.weights(anchor, aux, aux_mask, av, xv)
    xv2 = xv.clone()
    xv2[1, 0] = 3
    assert not torch.allclose(w[1], pool.weights(anchor, aux, aux_mask, av, xv2)[1])
    aux2 = aux.clone()
    aux2[1, 0] = aux2[1, 0] * 5
    assert not torch.allclose(w[1], pool.weights(anchor, aux2, aux_mask, av, xv)[1])


def test_gradient_reaches_the_pool_and_every_real_view():
    pool = _pool()
    anchor, aux, aux_mask, av, xv = _inputs()
    anchor.requires_grad_(True)
    aux.requires_grad_(True)
    pool(anchor, aux, aux_mask, av, xv).sum().backward()
    assert pool.weight.grad is not None and pool.weight.grad.abs().sum() > 0
    assert anchor.grad[0].abs().sum() > 0           # single-view study: through the anchor
    assert aux.grad[2].abs().sum() > 0
    assert (aux.grad[~aux_mask] == 0).all()         # padded slots get nothing


def test_shape_errors_fail_loud():
    pool = _pool()
    anchor, aux, aux_mask, av, xv = _inputs()
    with pytest.raises(ValueError):
        pool(anchor[:, :13], aux, aux_mask, av, xv)
    with pytest.raises(ValueError):
        pool(anchor, aux, aux_mask[:, :1], av, xv)
    bad = xv.clone()
    bad[2, 1] = 4
    with pytest.raises(ValueError):
        pool(anchor, aux, aux_mask, av, bad)


def test_the_loss_of_a_single_view_study_is_the_off_loss():
    torch.manual_seed(2)
    loss_fn = ClassificationLoss(class_weights=[[1.0, 0.8, 1.5]] * F)
    anchor = torch.randn(B, F, C)
    labels = torch.randint(0, C, (B, F))
    labels[0, 3] = -100
    # pooled == anchor, as for a batch of single-view studies
    late = late_classification_loss(loss_fn, anchor, anchor, labels, anchor_weight=0.5)
    off = loss_fn(anchor, labels)
    assert torch.allclose(late, off, atol=1e-6)
    pooled = torch.randn(B, F, C)
    mix = late_classification_loss(loss_fn, pooled, anchor, labels, anchor_weight=0.5)
    assert torch.allclose(mix, 0.5 * loss_fn(pooled, labels) + 0.5 * off, atol=1e-6)
    with pytest.raises(ValueError):
        late_classification_loss(loss_fn, pooled, anchor, labels, anchor_weight=1.5)


def test_mode_resolution_and_refusals():
    assert resolve_view_fusion_mode(None, multi_view=True, lambda_mpc=0.02) == "early"
    assert resolve_view_fusion_mode("early", multi_view=False) == "early"
    assert resolve_view_fusion_mode("late", multi_view=True) == "late"
    with pytest.raises(ValueError, match="one of"):
        resolve_view_fusion_mode("mid", multi_view=True)
    with pytest.raises(ValueError, match="multi_view"):
        resolve_view_fusion_mode("late", multi_view=False)
    for key in ("lambda_mpc", "lambda_view_consistency",
                "lambda_explanation", "lambda_explanation_strong"):
        with pytest.raises(ValueError, match=key):
            resolve_view_fusion_mode("late", multi_view=True, **{key: 0.05})


def test_default_path_builds_nothing_new():
    """Absent key = early: no late module, the anchor logits are the prediction."""
    source = _QFORMER.read_text()
    init = source[source.index("        self.view_fusion = None\n"):]
    init = init[: init.index("        self._last_raddino_patches = None")]
    assert "self.late_view_pool = None" in init
    assert 'if self.multi_view and self.view_fusion_mode == "late":' in init
    assert init.count("LateViewPooling(") == 1
    assert "elif self.multi_view:" in init          # early fusion, unchanged
    body = source[source.index("    def forward(self, samples):"):]
    body = body[: body.index("    @torch.no_grad()")]
    assert "classification_logits = student_logits\n" in body
    assert "classification_logits=classification_logits," in body
    helper = source[source.index("    def _late_view_logits("):]
    helper = helper[: helper.index("    def initialize_expert_tokens")]
    assert 'if getattr(self, "late_view_pool", None) is None:\n            return anchor_logits' in helper
    image = source[source.index("    def forward_image("):]
    image = image[: image.index("    def forward_text(")]
    assert "classification_logits = self._late_view_logits(classification_logits)" in image
    assert '"mode": view_fusion_cfg_raw.get("mode"),' in source


def test_shipped_yaml_keeps_early_fusion_and_trains_the_pool_when_late():
    cfg = yaml.safe_load(_CONFIG.read_text())
    assert "mode" not in cfg["model"]["view_fusion"]
    phases = cfg["run"]["phases"]
    for name in ("phase1b", "phase1c"):
        assert "late_view_pool" in phases[name]["trainable"]
        assert "view_fusion" in phases[name]["trainable"]


def _late_helper_source():
    source = _QFORMER.read_text()
    helper = source[source.index("    def _late_view_logits("):]
    return source, helper[: helper.index("    def initialize_expert_tokens")]


def test_late_detach_aux_is_off_by_default_and_refused_outside_late():
    """docs/handoff/PLAN-2026-10-26-late-fusion-detach-aux.md: absent = the LF path."""
    source, _ = _late_helper_source()
    assert 'self.late_detach_aux = bool(vf_cfg.pop("late_detach_aux", False))' in source
    assert ('if self.late_detach_aux and not (self.multi_view and self.view_fusion_mode == "late"):'
            in source)
    assert '"late_detach_aux": cfg_bool(view_fusion_cfg_raw.get("late_detach_aux", False)),' in source
    cfg = yaml.safe_load(_CONFIG.read_text())
    assert "late_detach_aux" not in cfg["model"]["view_fusion"]


def test_the_aux_pass_goes_through_the_shared_helper():
    _, helper = _late_helper_source()
    assert helper.count("late_aux_view_logits(") == 1
    assert "detach_aux=bool(getattr(self, \"late_detach_aux\", False))," in helper
    assert "separate_head=separate," in helper
    assert "self.mhcac_aux if separate else self.mhcac," in helper


D, P, T = 8, 5, 3


class _Head(torch.nn.Module):
    """MHCAC stand-in: same call signature and 5-tuple return, reads tokens and text."""

    def __init__(self, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.lin = torch.nn.Linear(D, F * C)

    def forward(self, shared, text_embeddings=None, text_attention_mask=None,
                labels=None, text_row_mask=None):
        x = shared.mean(1)
        if text_embeddings is not None:
            x = x + text_embeddings.mean(1)
        return self.lin(x).view(-1, F, C), None, None, None, None


def _late_setup(separate=False, detach=False, seed=3, anchor_grad=True):
    """Projector, text tower, anchor head, aux head and pool wired as Blip2Qformer does.

    ``anchor_grad=False`` computes the anchor logits without gradient, so whatever
    gradient the modules then receive came from the aux side alone.
    """
    torch.manual_seed(seed)
    proj = torch.nn.Linear(D, D)
    text_tower = torch.nn.Linear(D, D)
    anchor_head = _Head(10)
    aux_head = _Head(11) if separate else anchor_head
    pool = _pool()
    _, _, aux_mask, av, xv = _inputs()
    anchor_raw = torch.randn(B, P, D)
    aux_raw = torch.randn(B * N, P, D)                 # frozen encoder output
    text = text_tower(torch.randn(B, T, D))
    with torch.enable_grad() if anchor_grad else torch.no_grad():
        anchor_logits, *_ = anchor_head(proj(anchor_raw), text_embeddings=text)
    keep = real_aux_rows(aux_mask, B * N, anchor_logits.device)
    rows = (torch.arange(B * N) // N)[keep]
    pooled, aux_logits = late_aux_view_logits(
        anchor_logits, lambda: proj(aux_raw[keep]), aux_head, pool, aux_mask, av, xv,
        keep, rows, text_embeddings=text, detach_aux=detach, separate_head=separate,
    )
    mods = {"proj": proj, "text": text_tower, "anchor": anchor_head,
            "aux": aux_head, "pool": pool}
    return mods, anchor_logits, pooled, aux_logits, aux_mask


def _grads(loss, module):
    g = torch.autograd.grad(loss, list(module.parameters()), retain_graph=True, allow_unused=True)
    return sum(0.0 if x is None else float(x.abs().sum()) for x in g)


def _loss_and_labels():
    torch.manual_seed(5)
    loss_fn = ClassificationLoss(class_weights=[[1.0, 0.8, 1.5]] * F)
    labels = torch.randint(0, C, (B, F))
    labels[1, 2] = -100
    return loss_fn, labels


def test_aux_head_side_never_reaches_the_anchor_path():
    """late_aux_head: CE(aux) + CE(pooled) train mhcac_aux and the pool, nothing else."""
    mods, anchor_logits, pooled, aux_logits, aux_mask = _late_setup(separate=True)
    loss_fn, labels = _loss_and_labels()
    total, (l_anchor, l_aux, l_pooled) = late_aux_head_loss(
        loss_fn, pooled, anchor_logits, aux_logits, labels, aux_mask)
    side = l_aux + l_pooled
    for name in ("proj", "text", "anchor"):
        assert _grads(side, mods[name]) == 0.0, name
    assert _grads(side, mods["aux"]) > 0
    assert _grads(side, mods["pool"]) > 0
    # so the anchor path's gradient under the full loss is exactly OFF's CE(anchor)
    for name in ("proj", "text", "anchor"):
        full = torch.autograd.grad(total, list(mods[name].parameters()), retain_graph=True)
        off = torch.autograd.grad(l_anchor, list(mods[name].parameters()), retain_graph=True)
        assert all(torch.equal(a, b) for a, b in zip(full, off)), name


def test_aux_head_keeps_single_view_studies_at_their_anchor_logits():
    _, anchor_logits, pooled, aux_logits, aux_mask = _late_setup(separate=True)
    assert torch.equal(pooled[0], anchor_logits[0])          # study 0 has no aux view
    assert not torch.equal(pooled[2], anchor_logits[2])
    assert aux_logits.shape == (B, N, F, C)
    assert torch.equal(aux_logits[0], torch.zeros(N, F, C))   # padded slots stay zero


def test_late_detach_aux_trains_only_the_pool_from_the_aux_side():
    mods, _, pooled, aux_logits, _ = _late_setup(detach=True, anchor_grad=False)
    assert not aux_logits.requires_grad
    loss = pooled[1:].sum()
    for name in ("proj", "text", "anchor"):
        assert _grads(loss, mods[name]) == 0.0, name
    assert _grads(loss, mods["pool"]) > 0


def test_lf_path_sends_the_aux_gradient_into_the_shared_path():
    mods, _, pooled, aux_logits, _ = _late_setup(anchor_grad=False)
    assert aux_logits.requires_grad
    loss = pooled[1:].sum()
    for name in ("proj", "text", "anchor", "pool"):         # mhcac is the aux head in LF
        assert _grads(loss, mods[name]) > 0, name


def test_aux_head_and_detach_aux_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        _late_setup(separate=True, detach=True)


def test_aux_head_loss_is_the_sum_of_its_three_terms():
    loss_fn, labels = _loss_and_labels()
    torch.manual_seed(6)
    anchor, aux, aux_mask, _, _ = _inputs()
    pooled = torch.randn(B, F, C)
    mask = torch.tensor([True, True, True, False])
    total, (la, lx, lp) = late_aux_head_loss(loss_fn, pooled, anchor, aux, labels, aux_mask,
                                             sample_mask=mask)
    assert torch.allclose(la, loss_fn(anchor, labels, sample_mask=mask))
    assert torch.allclose(lp, loss_fn(pooled, labels, sample_mask=mask & aux_mask.any(1)))
    flat_mask = aux_mask.reshape(-1) & mask.repeat_interleave(N)
    flat_labels = labels.repeat_interleave(N, dim=0)
    assert torch.allclose(lx, loss_fn(aux.flatten(0, 1), flat_labels, sample_mask=flat_mask))
    # slot (b, n) is scored against study b's labels
    assert torch.equal(flat_labels[2 * N + 1], labels[2])
    assert torch.allclose(total, la + lx + lp)
    # a batch with no aux view: exactly the OFF loss
    off, parts = late_aux_head_loss(loss_fn, anchor, anchor, None, labels, None, sample_mask=mask)
    assert torch.equal(off, loss_fn(anchor, labels, sample_mask=mask))
    assert float(parts[1]) == 0.0 and float(parts[2]) == 0.0


def test_late_aux_head_is_off_by_default_and_refused_outside_late():
    source, _ = _late_helper_source()
    assert 'self.late_aux_head = bool(vf_cfg.pop("late_aux_head", False))' in source
    assert ('if self.late_aux_head and not (self.multi_view and self.view_fusion_mode == "late"):'
            in source)
    assert "if self.late_aux_head and self.late_detach_aux:" in source
    assert "self.mhcac_aux = copy.deepcopy(self.mhcac) if self.late_aux_head else None" in source
    assert '"late_aux_head": cfg_bool(view_fusion_cfg_raw.get("late_aux_head", False)),' in source
    body = source[source.index("    def forward(self, samples):"):]
    assert 'elif late_on and getattr(self, "mhcac_aux", None) is not None:' in body
    cfg = yaml.safe_load(_CONFIG.read_text())
    assert "late_aux_head" not in cfg["model"]["view_fusion"]
    train = _TRAIN.read_text()
    assert 'name.startswith("mhcac_aux.") for name in state' in train
    assert "model.sync_aux_head_from_anchor()" in train


def test_no_grad_pass_still_trains_the_pool():
    """The mechanism on the pool itself: detached aux logits, gradient to the pool and anchor only."""
    pool = _pool()
    anchor, aux, aux_mask, av, xv = _inputs()
    anchor.requires_grad_(True)
    aux.requires_grad_(True)
    with torch.no_grad():
        aux_detached = aux * 1.0
    pool(anchor, aux_detached, aux_mask, av, xv).sum().backward()
    assert pool.weight.grad.abs().sum() > 0
    assert anchor.grad[2].abs().sum() > 0
    assert aux.grad is None
