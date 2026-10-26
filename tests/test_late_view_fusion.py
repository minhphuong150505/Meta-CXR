"""Late multi-view fusion (``model.view_fusion.mode: late``, 2026-10-26).

Pins the guarantees of docs/handoff/PLAN-2026-10-26-late-view-fusion.md that a
CPU box can check: a single-view study gets exactly its anchor logits, padded
slots are never read, zero-init is the mean of the real views, the refusals,
the loss of a single-view study is the OFF loss, and the default (absent key)
path builds nothing new. No GPU and no dataset required.
"""
from pathlib import Path

import pytest
import torch
import yaml

from mhcac.late_view_pool import (
    LateViewPooling,
    late_classification_loss,
    resolve_view_fusion_mode,
)
from mhcac.loss import ClassificationLoss

_ROOT = Path(__file__).resolve().parents[1]
_CONFIG = _ROOT / "pretraining" / "configs" / "mimic_cxr_full.yaml"
_QFORMER = _ROOT / "model" / "lavis" / "models" / "blip2_models" / "blip2_qformer.py"

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


def test_late_detach_aux_wraps_the_aux_pass_and_not_the_pool():
    _, helper = _late_helper_source()
    guard = "        with torch.no_grad() if detach else contextlib.nullcontext():\n"
    assert helper.count(guard) == 1
    inside = helper[helper.index(guard) + len(guard):]
    inside = inside[: inside.index("\n        if detach and aux_logits.requires_grad:")]
    # every line of the aux pass is indented under the guard
    assert all(line.startswith("            ") or not line.strip() for line in inside.splitlines())
    assert "self.shared_visual_projector(raw)" in inside
    assert "self.mhcac(" in inside
    assert "late_view_pool" not in inside
    after = helper[helper.index("if detach and aux_logits.requires_grad:"):]
    assert "return self.late_view_pool(" in after
    assert 'raise RuntimeError("late_detach_aux: the aux-view logits still require grad")' in after


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
