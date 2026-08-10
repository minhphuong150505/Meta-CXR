"""Tests for inference-only encoder selection at the shared-token boundary."""

from types import SimpleNamespace

import pytest
import torch

from model.lavis.models.blip2_models.blip2_qformer import (
    Blip2Qformer,
    _resolve_encoder_ablation,
)
from vision_encoders.shared_visual_tokens import SharedVisualTokenProjector


STREAM_DIMS = {"biovil": 8, "pubmedclip": 4, "swin": 6}


def _shared_tokens():
    projector = SharedVisualTokenProjector(STREAM_DIMS, visual_dim=8)
    streams = {
        "biovil": torch.randn(2, 3, 8),
        "pubmedclip": torch.randn(2, 2, 4),
        "swin": torch.randn(2, 4, 6),
    }
    return projector(streams)


def _apply_ablation(shared, ablate_encoders, *, training=False):
    """Exercise the model's exact post-projection ablation branch."""
    model = SimpleNamespace(
        ablate_encoders=ablate_encoders,
        training=training,
    )
    # Invoke the real method without constructing any heavyweight encoders.
    return Blip2Qformer._apply_encoder_ablation(model, shared)


def test_absent_active_set_preserves_the_original_object_and_values():
    shared = _shared_tokens()
    result = _apply_ablation(shared, _resolve_encoder_ablation(shared.spans, None))
    assert result is shared
    torch.testing.assert_close(result.tokens, shared.tokens, rtol=0, atol=0)


def test_single_active_encoder_zeros_exactly_the_other_spans():
    shared = _shared_tokens()
    ablated = _resolve_encoder_ablation(shared.spans, ["pubmedclip"])
    assert ablated == ("biovil", "swin")

    result = _apply_ablation(shared, ablated)
    torch.testing.assert_close(
        result.stream("pubmedclip"), shared.stream("pubmedclip"), rtol=0, atol=0
    )
    assert torch.count_nonzero(result.stream("biovil")) == 0
    assert torch.count_nonzero(result.stream("swin")) == 0
    assert result.tokens.shape == shared.tokens.shape
    assert result.spans == shared.spans


def test_all_active_encoders_is_the_unablated_path():
    built = ("biovil", "pubmedclip", "swin")
    assert _resolve_encoder_ablation(built, built) == ()


def test_unknown_active_encoder_fails_closed():
    with pytest.raises(ValueError, match="raddino"):
        _resolve_encoder_ablation(("biovil", "pubmedclip", "swin"), ["raddino"])


def test_ablation_is_rejected_while_training():
    with pytest.raises(RuntimeError, match="inference-only"):
        _apply_ablation(_shared_tokens(), ("swin",), training=True)
