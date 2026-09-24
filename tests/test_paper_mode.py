"""The paper-faithful pieces added 2026-09-24 (D-020).

* 10% per-encoder feature mask;
* MHCAC single path: self -> masked text -> image, Bernoulli text mask per
  element with no rescale, text only while training and only for studies with
  a report;
* ITC label smoothing that stays finite when invalid candidates are -inf;
* the ITC gate's R@1 / R@5 against a binomial test of chance;
* 296 native tokens with MedCLIP Swin, one tensor for both branches.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from mhcac.loss import smoothed_cross_entropy
from mhcac.mhcac_12 import AbnormalityClassificationModel, ExpertTokenCrossAttention, StreamLayout
from pretraining.itc_gate import binomial_upper_quantile, recall_at_k, recall_gate, score_itc
from vision_encoders.feature_mask import mask_aux_tokens, mask_encoder_tokens, num_masked
from vision_encoders.shared_visual_tokens import SharedVisualTokens

REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# 10% feature mask
# --------------------------------------------------------------------------


@pytest.mark.parametrize("count, expected", [(196, 20), (50, 5), (49, 5), (10, 1), (4, 0)])
def test_ten_percent_of_each_encoders_tokens(count, expected):
    assert num_masked(count, 0.1) == expected


def test_mask_zeroes_whole_tokens_independently_per_sample():
    torch.manual_seed(0)
    x = torch.randn(64, 50, 16) + 5.0  # no token is zero by chance
    y = mask_encoder_tokens(x, 0.1)
    zero = (y == 0).all(dim=-1)
    assert (zero.sum(dim=1) == 5).all()
    untouched = ~zero
    assert torch.equal(y[untouched], x[untouched])
    assert len({tuple(row.nonzero().flatten().tolist()) for row in zero}) > 1


def test_mask_off_returns_the_input_itself():
    x = torch.randn(2, 50, 8)
    assert mask_encoder_tokens(x, 0.0) is x
    with pytest.raises(ValueError):
        mask_encoder_tokens(x, 1.0)


def test_aux_views_are_masked_per_view():
    x = torch.randn(3, 2, 50, 8) + 5.0
    y = mask_aux_tokens(x, 0.1)
    assert ((y == 0).all(-1).sum(-1) == 5).all()


# --------------------------------------------------------------------------
# MHCAC, paper mode
# --------------------------------------------------------------------------


def _layer(**kw):
    torch.manual_seed(0)
    return ExpertTokenCrossAttention(32, 4, dropout=0.0, text_dropout_rate=0.2, **kw)


def test_element_text_mask_is_bernoulli_without_rescale():
    layer = _layer(text_mask_mode="element").train()
    torch.manual_seed(1)
    t = torch.ones(8, 40, 32)
    masked = layer._mask_text(t)
    values = set(masked.unique().tolist())
    assert values <= {0.0, 1.0}, "no 1/(1-p) rescale"
    assert 0.15 < float((masked == 0).float().mean()) < 0.25
    # Not whole-report: the zeros are scattered, not one row at a time.
    assert (masked == 0).any(dim=-1).any(dim=-1).all()
    assert not layer.eval()._mask_text(t).eq(0).any(), "no mask at eval"


def test_self_first_matches_the_papers_equations():
    layer = _layer(layer_order="self_first", text_mask_mode="element").eval()
    e = torch.randn(2, 6, 32)
    img = torch.randn(2, 10, 32)
    txt = torch.randn(2, 7, 32)
    out, _ = layer(e, img, txt)
    with torch.no_grad():
        e_self = layer.norm_self_attention(e + layer.self_attention(e, e, e)[0])
        e_text = layer.norm_expert_text(e_self + layer.expert_to_text_attention(e_self, txt, txt)[0])
        e_img = layer.norm_expert_image(e_text + layer.expert_to_image_attention(e_text, img, img)[0])
        expected = layer.norm_ff(layer.ffn_expert(e_img) + e_img)
    assert torch.allclose(out, expected, atol=1e-6)


def test_studies_without_a_report_take_the_image_only_path():
    layer = _layer(layer_order="self_first", text_mask_mode="element").eval()
    e = torch.randn(3, 6, 32)
    img = torch.randn(3, 10, 32)
    txt = torch.randn(3, 7, 32)
    rows = torch.tensor([True, False, True])
    with_text, _ = layer(e, img, txt, text_row_mask=rows)
    no_text, _ = layer(e, img, None)
    assert torch.allclose(with_text[1], no_text[1], atol=1e-6)
    assert not torch.allclose(with_text[0], no_text[0], atol=1e-4)


def test_the_historical_layer_is_unchanged():
    """text_first + report, no row mask: byte-for-byte the pre-2026-09-24 forward."""
    layer = _layer().train()
    e, img, txt = torch.randn(2, 6, 32), torch.randn(2, 10, 32), torch.randn(2, 7, 32)
    torch.manual_seed(7)
    out, _ = layer(e, img, txt)
    torch.manual_seed(7)
    keep = torch.rand(2, 1, 1) >= 0.2
    t = txt * keep.to(txt.dtype)
    et = layer.norm_expert_text(layer.expert_to_text_attention(e, t, t)[0] + e)
    expected, _ = layer._self_then_image(et, img)
    assert torch.allclose(out, expected, atol=1e-6)


def test_unknown_modes_raise():
    with pytest.raises(ValueError, match="layer_order"):
        _layer(layer_order="image_first")
    with pytest.raises(ValueError, match="text_mask_mode"):
        _layer(text_mask_mode="token")


def test_mhcac_keeps_all_296_native_tokens_with_medclip_swin():
    layouts = {
        "biovil": StreamLayout(196),
        "pubmedclip": StreamLayout(50, num_global_tokens=1),
        "swin": StreamLayout(50, num_global_tokens=1),
    }
    torch.manual_seed(0)
    model = AbnormalityClassificationModel(
        embed_dim=64, num_heads=4, num_layers=2, num_commmon_tokens=4, visual_dim=96,
        txt_dim=64, stream_layouts=layouts, layer_order="self_first",
        text_mask_mode="element",
    ).eval()
    assert model.cnn_downsampler is None, "BioViL must not be pooled to 7x7"
    tokens = torch.randn(2, 296, 96)
    shared = SharedVisualTokens(
        tokens=tokens,
        spans={"biovil": slice(0, 196), "pubmedclip": slice(196, 246), "swin": slice(246, 296)},
    )
    logits, *_ = model(shared)
    assert logits.shape == (2, 14, 3)
    bad = SharedVisualTokens(tokens=torch.randn(2, 295, 96),
                             spans={"biovil": slice(0, 196), "pubmedclip": slice(196, 246),
                                    "swin": slice(246, 295)})
    with pytest.raises(ValueError, match="layout declares"):
        model(bad)


# --------------------------------------------------------------------------
# ITC label smoothing
# --------------------------------------------------------------------------


def test_smoothing_equals_torch_when_every_candidate_is_valid():
    torch.manual_seed(0)
    logits = torch.randn(6, 9)
    targets = torch.randint(0, 9, (6,))
    ours = smoothed_cross_entropy(logits, targets, 0.1)
    ref = F.cross_entropy(logits, targets, label_smoothing=0.1)
    assert float(ours) == pytest.approx(float(ref), rel=1e-6)
    assert float(smoothed_cross_entropy(logits, targets, 0.0)) == pytest.approx(
        float(F.cross_entropy(logits, targets))
    )


def test_smoothing_stays_finite_with_masked_candidates():
    logits = torch.randn(4, 6)
    logits[:, 5] = float("-inf")  # a study without usable FINDINGS
    targets = torch.tensor([0, 1, 2, 3])
    assert not torch.isfinite(F.cross_entropy(logits, targets, label_smoothing=0.1))
    ours = smoothed_cross_entropy(logits, targets, 0.1)
    assert torch.isfinite(ours)
    ref = F.cross_entropy(logits[:, :5], targets, label_smoothing=0.1)
    assert float(ours) == pytest.approx(float(ref), rel=1e-6)


def test_the_shipped_config_turns_the_queue_off_and_smooths_itc():
    import yaml

    cfg = yaml.safe_load((REPO / "pretraining/configs/mimic_cxr_full.yaml").read_text())
    assert cfg["model"]["loss"]["itc_queue_size"] == 0
    assert cfg["model"]["loss"]["itc_label_smoothing"] == pytest.approx(0.1)
    assert cfg["model"]["feature_mask_ratio"] == pytest.approx(0.1)
    assert cfg["model"]["mhcac"]["text_guidance"] == "single_path"
    assert cfg["model"]["mhcac"]["layer_order"] == "self_first"
    assert cfg["model"]["mhcac"]["text_mask"] == "element"
    assert cfg["model"]["encoders"]["swin"] is True
    assert cfg["model"]["swin"]["backend"] == "medclip"
    assert cfg["model"]["swin"]["normalize"] is False


# --------------------------------------------------------------------------
# ITC gate: R@k
# --------------------------------------------------------------------------


def test_recall_at_k_on_a_perfect_and_a_reversed_ranking():
    n = 8
    perfect = torch.eye(n)
    assert recall_at_k(perfect, 1) == 1.0
    reversed_ = -torch.eye(n)
    assert recall_at_k(reversed_, 1) == 0.0
    assert recall_at_k(reversed_, n) == 1.0


def test_binomial_threshold_matches_a_brute_force_tail():
    n, p, alpha = 256, 5 / 256, 0.01
    c = binomial_upper_quantile(n, p, alpha)

    def tail(k):
        return sum(math.comb(n, j) * p**j * (1 - p) ** (n - j) for j in range(k, n + 1))

    assert tail(c) <= alpha < tail(c - 1)


def test_chance_similarities_fail_and_aligned_ones_pass():
    torch.manual_seed(0)
    n, q, d = 256, 4, 32
    img = F.normalize(torch.randn(n, q, d), dim=-1)
    txt = F.normalize(torch.randn(n, d), dim=-1)
    chance = score_itc(img, txt, temperature=0.07)
    assert not chance["recall_above_chance"] and not chance["meets_threshold"]
    aligned_txt = F.normalize(img[:, 0] + 0.3 * torch.randn(n, d), dim=-1)
    good = score_itc(img, aligned_txt, temperature=0.07)
    assert good["recall_above_chance"] and good["meets_threshold"]
    assert good["R@5_i2t"] > good["R@5_chance"]


def test_recall_gate_needs_both_directions():
    n = 256
    sim_good = torch.eye(n)
    sim_bad = -torch.eye(n)  # every true pair ranks last
    out = recall_gate(sim_good, sim_bad, n)
    assert out["R@5_i2t_above_chance"] and not out["R@5_t2i_above_chance"]
    assert not out["recall_above_chance"]


# --------------------------------------------------------------------------
# Host only: the real Blip2Qformer layout / branch-sharing contract
# --------------------------------------------------------------------------


def _qformer_class():
    pytest.importorskip("torchvision")
    try:
        from model.lavis.models.blip2_models.blip2_qformer import Blip2Qformer
    except Exception as exc:
        pytest.skip(f"Blip2Qformer not importable here ({type(exc).__name__})")
    return Blip2Qformer


def test_native_layouts_cover_medclip_swin():
    cls = _qformer_class()
    fake = SimpleNamespace(
        use_swin=True, use_raddino=False, use_biovil=True, use_pubmedclip=True,
        swin=SimpleNamespace(num_tokens=50, has_global_token=True),
        pubmedclip=SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(
            vision_config=SimpleNamespace(image_size=224, patch_size=32)))),
    )
    layouts = cls._native_stream_layouts(fake, 448)
    assert {k: (v.num_tokens, v.num_global_tokens) for k, v in layouts.items()} == {
        "biovil": (196, 0), "pubmedclip": (50, 1), "swin": (50, 1),
    }
    fake.swin = SimpleNamespace(num_tokens=None, has_global_token=False)  # SwinV2 hf path
    assert cls._native_stream_layouts(fake, 448) is None


def test_qformer_and_mhcac_read_the_same_token_tensor():
    """forward() hands shared.tokens to the Q-Former and shared itself to MHCAC."""
    source = (REPO / "model/lavis/models/blip2_models/blip2_qformer.py").read_text()
    body = source[source.index("    def forward(self, samples):"):]
    body = body[: body.index("    @torch.no_grad()")]
    assert "shared_visual = self.encode_samples(samples)" in body
    assert "image_embeds = shared_visual.tokens" in body
    assert "encoder_hidden_states=image_embeds" in body
    assert body.count("self.mhcac(\n                    shared_visual,") >= 1


def test_phase1a_does_not_run_mhcac():
    cls = _qformer_class()
    lambdas = dict(
        lambda_cls=0.0, lambda_teacher_cls=0.0, lambda_distill=0.0,
        lambda_mhcac_contrastive=0.0, lambda_orthogonality=0.0, lambda_sparsity=0.0,
        lambda_gate=0.0, lambda_mention_conditioned_cls=0.0, lambda_view_consistency=0.0,
        lambda_explanation=0.0, lambda_explanation_strong=0.0,
    )
    assert cls.needs_mhcac(SimpleNamespace(**lambdas)) is False
    lambdas["lambda_cls"] = 1.0
    assert cls.needs_mhcac(SimpleNamespace(**lambdas)) is True
