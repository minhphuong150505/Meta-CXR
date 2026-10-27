"""``model.view_fusion.mode: tokens`` (2026-10-27): aux-view tokens on MHCAC's key axis.

Plan and reading rule: ``docs/handoff/PLAN-2026-10-27-aux-tokens-mhcac.md``.

Invariants pinned here, all on CPU in fp32:

* mode resolution: same requirements and refusals as ``late``;
* other configurations build no aux parameters, and building them does not move
  the RNG draws of anything else;
* a study without a real aux view gets the logits it gets with no aux input at all,
  and zero attention weight on aux keys;
* a batch without any real aux view runs the historical path bit for bit;
* the sparsity term is a mean over REAL attention cells;
* gradient reaches the aux parameters and the aux input tokens.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mhcac.late_view_pool import VIEW_FUSION_MODES, resolve_view_fusion_mode
from mhcac.loss import AbnormalitySpecificLoss
from mhcac.mhcac_12 import AbnormalityClassificationModel, StreamLayout
from vision_encoders.shared_visual_tokens import SharedVisualTokens

REPO = Path(__file__).resolve().parents[1]
_QFORMER = REPO / "model" / "lavis" / "models" / "blip2_models" / "blip2_qformer.py"

VISUAL_DIM = 24
LAYOUTS = {"biovil": StreamLayout(9), "pubmedclip": StreamLayout(5, num_global_tokens=1)}
SPANS = {"biovil": slice(0, 9), "pubmedclip": slice(9, 14)}
TOKENS = 14


def _model(aux=True, seed=0):
    torch.manual_seed(seed)
    model = AbnormalityClassificationModel(
        embed_dim=32,
        num_heads=4,
        num_abnormalities=14,
        num_layers=3,
        num_commmon_tokens=8,
        dropout=0.0,
        visual_dim=VISUAL_DIM,
        text_dropout_rate=0.0,
        num_text_teacher_layers=0,
        stream_layouts=LAYOUTS,
        layer_order="self_first",
        text_mask_mode="element",
        aux_view_tokens=aux,
    )
    return model.eval()


def _shared(batch):
    return SharedVisualTokens(tokens=torch.randn(batch, TOKENS, VISUAL_DIM), spans=dict(SPANS))


def test_mode_resolution_matches_late():
    assert "tokens" in VIEW_FUSION_MODES
    assert resolve_view_fusion_mode("tokens", multi_view=True) == "tokens"
    with pytest.raises(ValueError, match="multi_view"):
        resolve_view_fusion_mode("tokens", multi_view=False)
    for key in ("lambda_mpc", "lambda_view_consistency",
                "lambda_explanation", "lambda_explanation_strong"):
        with pytest.raises(ValueError, match=key):
            resolve_view_fusion_mode("tokens", multi_view=True, **{key: 0.05})


def test_other_configurations_build_no_aux_parameters_and_keep_their_rng():
    plain, aux = _model(aux=False), _model(aux=True)
    assert not hasattr(plain, "aux_pos_enc") and not hasattr(plain, "aux_view_emb")
    plain_state, aux_state = plain.state_dict(), aux.state_dict()
    extra = set(aux_state) - set(plain_state)
    assert extra and all(k.startswith(("aux_pos_enc.", "aux_view_emb.")) for k in extra)
    assert set(plain_state) <= set(aux_state)
    for key, value in plain_state.items():
        assert torch.equal(value, aux_state[key]), key
    with pytest.raises(ValueError, match="stream_layouts"):
        AbnormalityClassificationModel(
            embed_dim=32, num_heads=4, visual_dim=VISUAL_DIM, aux_view_tokens=True,
            num_text_teacher_layers=0,
        )


def test_aux_inputs_refused_without_aux_parameters():
    model = _model(aux=False)
    torch.manual_seed(1)
    with pytest.raises(ValueError, match="aux_view_tokens"):
        model(_shared(2), aux_visual_tokens=_shared(2),
              aux_slot_mask=torch.tensor([[True], [False]]))


def test_single_view_study_is_unchanged_by_other_studies_aux_tokens():
    model = _model()
    torch.manual_seed(2)
    anchors, aux = _shared(3), _shared(3)
    slot_mask = torch.tensor([[True], [False], [False]])
    view_ids = torch.tensor([[2], [0], [0]])
    with torch.no_grad():
        mixed, attn, *_ = model(anchors, aux_visual_tokens=aux, aux_slot_mask=slot_mask,
                                aux_view_ids=view_ids)
        alone, *_ = model(anchors)
    assert torch.allclose(mixed[1:], alone[1:], atol=1e-6, rtol=0)
    assert not torch.allclose(mixed[0], alone[0], atol=1e-4)
    for layer_attention in attn:
        assert layer_attention.shape[-1] == 2 * TOKENS
        assert torch.equal(layer_attention[1:, :, TOKENS:],
                           torch.zeros_like(layer_attention[1:, :, TOKENS:]))
        assert layer_attention[0, :, TOKENS:].sum() > 0


def test_batch_without_real_aux_runs_the_historical_path_bit_for_bit():
    model = _model()
    torch.manual_seed(3)
    anchors, aux = _shared(2), _shared(2)
    labels = torch.randint(0, 3, (2, 14))
    with torch.no_grad():
        with_args = model(anchors, labels=labels, aux_visual_tokens=aux,
                          aux_slot_mask=torch.zeros(2, 1, dtype=torch.bool))
        without = model(anchors, labels=labels)
    assert torch.equal(with_args[0], without[0])
    for a, b in zip(with_args[1], without[1]):
        assert torch.equal(a, b)
    for a, b in zip(with_args[2:], without[2:]):
        assert torch.equal(torch.as_tensor(a), torch.as_tensor(b))


def test_view_type_embedding_changes_the_aux_study_only():
    model = _model()
    torch.manual_seed(4)
    anchors, aux = _shared(2), _shared(2)
    slot_mask = torch.tensor([[True], [False]])
    with torch.no_grad():
        lateral, *_ = model(anchors, aux_visual_tokens=aux, aux_slot_mask=slot_mask,
                            aux_view_ids=torch.tensor([[2], [0]]))
        frontal, *_ = model(anchors, aux_visual_tokens=aux, aux_slot_mask=slot_mask,
                            aux_view_ids=torch.tensor([[0], [0]]))
    assert not torch.allclose(lateral[0], frontal[0], atol=1e-6)
    assert torch.equal(lateral[1], frontal[1])


def test_gradient_reaches_aux_parameters_and_aux_tokens():
    model = _model().train()
    torch.manual_seed(5)
    anchors = _shared(2)
    aux_tokens = torch.randn(2, TOKENS, VISUAL_DIM, requires_grad=True)
    aux = SharedVisualTokens(tokens=aux_tokens, spans=dict(SPANS))
    logits, *_ = model(anchors, aux_visual_tokens=aux,
                       aux_slot_mask=torch.tensor([[True], [False]]),
                       aux_view_ids=torch.tensor([[2], [0]]))
    logits.sum().backward()
    for name, param in model.named_parameters():
        if name.startswith(("aux_pos_enc.", "aux_view_emb.")):
            assert param.grad is not None, name
    assert model.aux_pos_enc["biovil"].positional_encoding.grad.abs().sum() > 0
    assert model.aux_view_emb.weight.grad[2].abs().sum() > 0
    assert model.aux_view_emb.weight.grad[1].abs().sum() == 0   # no slot uses id 1
    assert aux_tokens.grad[0].abs().sum() > 0
    assert aux_tokens.grad[1].abs().sum() == 0                  # masked slot


def test_sparsity_is_a_mean_over_real_cells():
    loss = AbnormalitySpecificLoss(d_embedding=32)
    torch.manual_seed(6)
    weights = [torch.softmax(torch.randn(3, 8, 10), dim=-1) for _ in range(3)]
    base = loss.compute_weighted_sparsity_loss(weights)
    no_mask = loss.compute_weighted_sparsity_loss(
        weights, attention_key_mask=torch.zeros(3, 10, dtype=torch.bool)
    )
    assert torch.allclose(base, no_mask, atol=1e-7, rtol=0)
    padded = [torch.cat([w, torch.zeros(3, 8, 6)], dim=-1) for w in weights]
    mask = torch.cat([torch.zeros(3, 10, dtype=torch.bool), torch.ones(3, 6, dtype=torch.bool)], 1)
    assert torch.allclose(
        loss.compute_weighted_sparsity_loss(padded, attention_key_mask=mask), base,
        atol=1e-7, rtol=0,
    )


def test_qformer_wires_tokens_mode_into_both_student_calls():
    source = _QFORMER.read_text()
    init = source[source.index("        self.view_fusion = None\n"):]
    init = init[: init.index("        self._last_raddino_patches = None")]
    assert 'self.aux_tokens_on = bool(self.multi_view and self.view_fusion_mode == "tokens")' in init
    assert "elif self.aux_tokens_on:" in init
    assert init.index("elif self.aux_tokens_on:") < init.index("elif self.multi_view:")
    assert "aux_view_tokens=self.aux_tokens_on," in source
    body = source[source.index("    def forward(self, samples):"):]
    body = body[: body.index("    @torch.no_grad()")]
    assert body.index("aux_token_inputs = self._aux_token_inputs()") < body.index("self.mhcac(")
    assert body.count("**aux_token_inputs,") == 1
    image = source[source.index("    def forward_image("):]
    image = image[: image.index("    def forward_text(")]
    assert "aux_token_inputs = self._aux_token_inputs()" in image
    assert image.count("**aux_token_inputs,") == 1
    # late_detach_aux / late_aux_head still demand mode: late, so tokens refuses them.
    assert 'self.late_aux_head and not (self.multi_view and self.view_fusion_mode == "late")' in source
    assert 'self.late_detach_aux and not (self.multi_view and self.view_fusion_mode == "late")' in source
