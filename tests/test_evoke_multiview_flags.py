"""The two EVOKE-alignment flags: ``view_fusion.detach_aux`` and ``loss.mpc_temperature``.

EVOKE (arXiv 2411.10224, released code) detaches the auxiliary-view K/V inside
its fusion attention and runs its multi-positive contrastive loss at
temperature 0.5. Both are opt-in here; the defaults must reproduce every
recorded run. No GPU and no dataset required.
"""
import re
from pathlib import Path

import torch
import yaml

from mhcac.loss import MultiPositiveContrastiveLoss
from mhcac.view_fusion import ViewFusionModule

_ROOT = Path(__file__).resolve().parents[1]
_CONFIG = _ROOT / "pretraining" / "configs" / "mimic_cxr_full.yaml"
_QFORMER = _ROOT / "model" / "lavis" / "models" / "blip2_models" / "blip2_qformer.py"

B, N, P, D = 3, 2, 5, 16


def _fusion(detach_aux):
    torch.manual_seed(0)
    m = ViewFusionModule(dim=D, num_heads=4, ffn_ratio=2, p_view_drop=0.0,
                         detach_aux=detach_aux).train()
    # Past step 0: with the zero-init W_O no gradient reaches K/V at all, so
    # the test would pass for the wrong reason.
    torch.nn.init.normal_(m.blocks[0].w_o.weight, std=0.1)
    return m


def _run(m):
    torch.manual_seed(1)
    anchor = torch.randn(B, P, D, requires_grad=True)
    aux = torch.randn(B, N, P, D, requires_grad=True)
    aux_mask = torch.ones(B, N, dtype=torch.bool)
    out = m(anchor, aux, aux_mask,
            anchor_view_id=torch.tensor([0, 1, 0]),
            aux_view_ids=torch.tensor([[2, 2], [2, 3], [1, 2]]))
    out.sum().backward()
    return out, anchor, aux


def test_detach_aux_defaults_off():
    assert ViewFusionModule(dim=D, num_heads=4).detach_aux is False


def test_detach_aux_stops_gradient_into_aux_features_only():
    out_live, anchor_live, aux_live = _run(_fusion(detach_aux=False))
    m = _fusion(detach_aux=True)
    out_det, anchor_det, aux_det = _run(m)

    # Same forward value: detach changes gradients, never the output.
    torch.testing.assert_close(out_det, out_live)
    assert aux_live.grad is not None and aux_live.grad.abs().sum() > 0
    assert aux_det.grad is None
    # The anchor (query) path and the fusion weights still learn.
    assert anchor_det.grad is not None and anchor_det.grad.abs().sum() > 0
    for name in ("w_q", "w_k", "w_v", "w_o"):
        g = getattr(m.blocks[0], name).weight.grad
        assert g is not None and g.abs().sum() > 0, name
    # Only the aux FEATURES are detached; the view-type embedding still trains.
    assert m.view_emb.weight.grad is not None
    assert m.view_emb.weight.grad[2].abs().sum() > 0


def test_mpc_temperature_is_honoured():
    torch.manual_seed(0)
    anchor = torch.randn(B, D)
    aux = anchor.unsqueeze(1).repeat(1, N, 1) + 0.05 * torch.randn(B, N, D)
    mask = torch.ones(B, N, dtype=torch.bool)
    assert MultiPositiveContrastiveLoss().temperature == 0.07
    sharp = MultiPositiveContrastiveLoss(temperature=0.07)(anchor, aux, mask)
    soft = MultiPositiveContrastiveLoss(temperature=0.5)(anchor, aux, mask)
    # Well-aligned positives: a softer softmax cannot score them as confidently.
    assert soft > sharp, (soft.item(), sharp.item())


def test_shipped_yaml_keeps_recorded_behaviour():
    cfg = yaml.safe_load(_CONFIG.read_text(encoding="utf-8"))["model"]
    assert cfg["view_fusion"]["detach_aux"] is False
    assert cfg["loss"]["mpc_temperature"] == 0.07


def test_from_config_reads_both_keys():
    """from_config silently drops keys it does not read, so pin that it reads these."""
    src = _QFORMER.read_text(encoding="utf-8")
    assert re.search(r'view_fusion_cfg_raw\.get\("detach_aux"', src)
    assert re.search(r'loss_cfg\.get\("mpc_temperature"', src)
    assert "MultiPositiveContrastiveLoss(temperature=mpc_temperature)" in src
