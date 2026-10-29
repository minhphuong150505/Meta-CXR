"""Asymmetric focusing on Negative cells in ClassificationLoss (Ridnik et al., ICCV 2021).

docs/handoff/PLAN-2026-10-29-stage1-asymmetric-loss.md. Off (the default) must be the
historical weighted cross entropy bit for bit; on, a Negative cell costs
``-stopgrad((1 - q) ** gamma) * log(q)`` with ``q = min(p_neg + clip, 1)`` and a
Positive / Uncertain cell keeps ``-log p_y``. Three classes throughout.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mhcac.loss import ClassificationLoss, asymmetric_loss_settings  # noqa: E402

WEIGHTS = [[1.0, 3.0, 10.0], [1.0, 2.0, 5.0]]


def _batch(seed=0, n=12):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(n, 2, 3, generator=g) * 2
    labels = torch.randint(0, 3, (n, 2), generator=g)
    labels[0, 0] = -100
    labels[3, 1] = -100
    mask = torch.ones(n, dtype=torch.bool)
    mask[5] = False
    return logits, labels, mask


@pytest.mark.parametrize("weights", [None, WEIGHTS])
def test_off_is_the_historical_loss_and_gradient(weights):
    logits, labels, mask = _batch()
    a = logits.clone().requires_grad_(True)
    b = logits.clone().requires_grad_(True)
    old = ClassificationLoss(num_abnormalities=2, class_weights=weights)(a, labels, mask)
    new = ClassificationLoss(num_abnormalities=2, class_weights=weights,
                             asymmetric_gamma_neg=0.0, asymmetric_clip=0.0)(b, labels, mask)
    old.backward(), new.backward()
    assert torch.equal(old, new)
    assert torch.equal(a.grad, b.grad)


def test_hand_computed_example():
    # one finding, three cells: Negative, Positive, Uncertain; weights [1, 3, 10]
    logits = torch.tensor([[[0.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]], [[1.0, 0.0, 2.0]]])
    labels = torch.tensor([[0], [1], [2]])
    loss = ClassificationLoss(num_abnormalities=1, class_weights=[[1.0, 3.0, 10.0]],
                              asymmetric_gamma_neg=4.0, asymmetric_clip=0.05)(logits, labels)
    q = 1 / 3 + 0.05
    neg = -((1 - q) ** 4) * math.log(q)
    e = math.e
    pos = -math.log(e / (e + 2))
    unc = -math.log(e ** 2 / (e + 1 + e ** 2))
    expected = (1 * neg + 3 * pos + 10 * unc) / (1 + 3 + 10)
    assert loss.item() == pytest.approx(expected, rel=1e-6)


def test_easy_negative_costs_nothing_and_has_no_gradient():
    logits = torch.tensor([[[6.0, 0.0, 0.0]]], requires_grad=True)  # p_neg ~ 0.995 >= 0.95
    loss = ClassificationLoss(num_abnormalities=1, asymmetric_gamma_neg=4.0,
                              asymmetric_clip=0.05)(logits, torch.tensor([[0]]))
    loss.backward()
    assert loss.item() == 0.0
    assert torch.count_nonzero(logits.grad) == 0


def test_positive_and_uncertain_cells_are_plain_weighted_cross_entropy():
    logits, labels, _ = _batch(seed=3)
    labels = labels.clamp(min=1)  # only Positive / Uncertain
    on = ClassificationLoss(num_abnormalities=2, class_weights=WEIGHTS,
                            asymmetric_gamma_neg=4.0, asymmetric_clip=0.05)(logits, labels)
    off = ClassificationLoss(num_abnormalities=2, class_weights=WEIGHTS)(logits, labels)
    assert on.item() == pytest.approx(off.item(), rel=1e-6)


def test_focus_weight_carries_no_gradient():
    torch.manual_seed(1)
    logits = torch.randn(1, 1, 3, requires_grad=True)
    gamma, clip = 4.0, 0.05
    ClassificationLoss(num_abnormalities=1, asymmetric_gamma_neg=gamma,
                       asymmetric_clip=clip)(logits, torch.tensor([[0]])).backward()
    p = F.softmax(logits.detach()[0, 0], dim=-1)
    q = min(p[0].item() + clip, 1.0)
    w = (1 - q) ** gamma
    # d/dz of -w*log(p_neg + clip) with w held constant
    dlogp0 = torch.tensor([1.0, 0.0, 0.0]) - p
    expected = -w * (p[0].item() / q) * dlogp0
    assert torch.allclose(logits.grad[0, 0], expected, atol=1e-6)


def test_harder_negative_gets_more_weight_than_easier_one():
    loss = ClassificationLoss(num_abnormalities=1, asymmetric_gamma_neg=4.0, asymmetric_clip=0.05)
    hard = loss(torch.tensor([[[0.0, 1.0, 0.0]]]), torch.tensor([[0]]))
    easy = loss(torch.tensor([[[2.0, 0.0, 0.0]]]), torch.tensor([[0]]))
    plain = ClassificationLoss(num_abnormalities=1)
    ratio_asl = hard / easy
    ratio_ce = plain(torch.tensor([[[0.0, 1.0, 0.0]]]), torch.tensor([[0]])) / \
        plain(torch.tensor([[[2.0, 0.0, 0.0]]]), torch.tensor([[0]]))
    assert ratio_asl > ratio_ce


@pytest.mark.parametrize("kwargs, match", [
    (dict(asymmetric_gamma_neg=-1.0), "gamma_neg"),
    (dict(asymmetric_gamma_neg=4.0, asymmetric_clip=1.0), "clip"),
    (dict(asymmetric_gamma_neg=4.0, label_smoothing=0.05), "label_smoothing"),
    (dict(asymmetric_gamma_neg=4.0, logit_adjust_counts=[[6, 3, 1], [9, 1, 0]]), "exclusive"),
])
def test_bad_settings_are_refused(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ClassificationLoss(num_abnormalities=2, **kwargs)


def test_config_settings():
    assert asymmetric_loss_settings({}) == (0.0, 0.0)
    assert asymmetric_loss_settings(None) == (0.0, 0.0)
    assert asymmetric_loss_settings({"asymmetric_loss": {"gamma_neg": 4, "clip": 0.05}}) == (4.0, 0.05)
    assert asymmetric_loss_settings({"asymmetric_loss": {}}) == (4.0, 0.05)
    with pytest.raises(ValueError, match="unknown keys"):
        asymmetric_loss_settings({"asymmetric_loss": {"gamma_pos": 1}})


def test_config_reaches_the_model_loss():
    """from_config must read the key and pass it to cls_loss_fn (source check; CPU box lacks LAVIS)."""
    src = (_REPO_ROOT / "model/lavis/models/blip2_models/blip2_qformer.py").read_text()
    assert "asymmetric_loss_settings(mhcac_cfg)" in src
    assert src.count("asymmetric_gamma_neg=asymmetric_gamma_neg") == 2  # cls(...) and ClassificationLoss(...)


def test_bf16_logits_are_scored_in_float32():
    logits, labels, mask = _batch(seed=5)
    loss = ClassificationLoss(num_abnormalities=2, class_weights=WEIGHTS, asymmetric_gamma_neg=4.0,
                              asymmetric_clip=0.05)
    a = loss(logits.to(torch.bfloat16), labels, mask)
    b = loss(logits.to(torch.bfloat16).float(), labels, mask)
    assert a.dtype == torch.float32
    assert a.item() == pytest.approx(b.item(), rel=1e-6)
