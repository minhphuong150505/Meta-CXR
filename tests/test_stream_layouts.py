"""Each encoder must reach MHCAC at its own scale, intact.

The point of running two frozen encoders is that they see different things:
BioViL-T resolves a 14x14 grid over the full 448 image, PubMedCLIP contributes a
coarse 7x7 grid plus the CLS token it was contrastively trained to make
meaningful. The previous code pooled BioViL to 7x7 and dropped the CLS, leaving
two interchangeable coarse maps and no global token at all.

Every failure mode here is silent -- the loss still falls with the wrong tokens,
the wrong grid, or a Grad-CAM handle that is detached from the graph -- so each
one is pinned.
"""

import pytest
import torch

from mhcac.mhcac_12 import AbnormalityClassificationModel, StreamLayout
from vision_encoders.shared_visual_tokens import SharedVisualTokens

BIOVIL_TOKENS = 196  # 448 / 32 -> 14x14
PMC_TOKENS = 50  # 224 / 32 -> 7x7, plus CLS
LAYOUTS = {
    "biovil": StreamLayout(BIOVIL_TOKENS),
    "pubmedclip": StreamLayout(PMC_TOKENS, num_global_tokens=1),
}
VISUAL_DIM = 1408


def _model(**kwargs):
    return AbnormalityClassificationModel(
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        num_commmon_tokens=14,
        visual_dim=VISUAL_DIM,
        num_text_teacher_layers=0,
        **kwargs,
    )


def _tokens(batch=2):
    total = BIOVIL_TOKENS + PMC_TOKENS
    return SharedVisualTokens(
        tokens=torch.randn(batch, total, VISUAL_DIM),
        spans={
            "biovil": slice(0, BIOVIL_TOKENS),
            "pubmedclip": slice(BIOVIL_TOKENS, total),
        },
    )


def test_native_layout_builds_no_downsampler():
    """The 196 -> 49 pooling is the thing being removed; it must not exist."""
    assert _model(stream_layouts=LAYOUTS, use_cnn=True).cnn_downsampler is None


def test_legacy_path_keeps_the_downsampler():
    assert _model(stream_layouts=None, use_cnn=True).cnn_downsampler is not None


def test_each_encoder_gets_its_own_positional_encoding():
    model = _model(stream_layouts=LAYOUTS, use_cnn=True)

    sizes = {
        name: model.pos_enc[name].positional_encoding.shape[1] for name in LAYOUTS
    }

    assert sizes == {"biovil": BIOVIL_TOKENS, "pubmedclip": PMC_TOKENS}


def test_positional_encoding_does_not_swamp_the_token_content():
    """Tokens arrive L2-normalised to 1.0; a randn init would be ~28x that."""
    model = _model(stream_layouts=LAYOUTS, use_cnn=True)

    norm = model.pos_enc["biovil"].positional_encoding.norm(dim=-1).mean()

    assert norm < 1.0, f"positional encoding norm {norm:.2f} drowns unit-norm content"


def test_capture_grids_are_native_and_exclude_the_global_token():
    model = _model(stream_layouts=LAYOUTS, use_cnn=True)
    model.capture_streams = True

    model(_tokens())

    captured = {k: (v[0].shape[1], v[1]) for k, v in model._last_cam_streams.items()}
    assert captured["biovil"] == (196, (14, 14)), "BioViL must keep its fine grid"
    assert captured["pubmedclip"] == (49, (7, 7)), "CLS must not enter the CAM grid"


def test_captured_activations_stay_on_the_path_to_the_output():
    """Grad-CAM differentiates the logits w.r.t. these tensors.

    Slicing the spatial tokens off and then feeding the *unsliced* tensor
    downstream would leave the captured handle on a dead branch: autograd.grad
    raises, or with allow_unused returns None, and the explanation loss silently
    becomes a no-op.
    """
    model = _model(stream_layouts=LAYOUTS, use_cnn=True)
    model.capture_streams = True

    logits = model(_tokens())[0]
    for name, (activation, _) in model._last_cam_streams.items():
        grad = torch.autograd.grad(
            logits.square().sum(), activation, retain_graph=True, allow_unused=True
        )[0]

        assert grad is not None, f"{name} activation is detached from the graph"
        assert torch.any(grad != 0), f"{name} activation receives no gradient"


def test_global_token_is_carried_through_not_dropped():
    """A regression guard on the CLS: it used to be silently sliced away."""
    model = _model(stream_layouts=LAYOUTS, use_cnn=True)
    model.capture_streams = True

    model(_tokens())

    spatial = model._last_cam_streams["pubmedclip"][0].shape[1]
    assert PMC_TOKENS - spatial == 1, "the one global token must survive the layout"


def test_declared_layout_must_match_the_actual_span():
    """The layout is derived from encoder config; a mismatch must not be quiet."""
    model = _model(
        stream_layouts={"biovil": StreamLayout(99)}, use_cnn=True
    )
    tokens = SharedVisualTokens(
        tokens=torch.randn(2, BIOVIL_TOKENS, VISUAL_DIM),
        spans={"biovil": slice(0, BIOVIL_TOKENS)},
    )

    with pytest.raises(ValueError, match="layout declares"):
        model(tokens)
