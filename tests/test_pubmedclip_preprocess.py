"""PubMedCLIP's own preprocessing (D-022) and how it is wired.

The reference in tests/fixtures/pubmedclip_preprocess_reference.npz was produced
on the training host by the real slow ``CLIPImageProcessor`` of
``flaviagiammarino/pubmed-clip-vit-base-patch32``, on SYNTHETIC images, by
scripts/make_pubmedclip_preprocess_reference.py.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from vision_encoders.pubmedclip.preprocess import (
    PUBMEDCLIP_IMG_SIZE,
    pubmedclip_preprocess,
    resolve_pubmedclip_preprocess,
    shortest_edge_size,
)

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests/fixtures/pubmedclip_preprocess_reference.npz"


def _reference():
    spec = importlib.util.spec_from_file_location(
        "_make_clip_ref", REPO / "scripts/make_pubmedclip_preprocess_reference.py"
    )
    source = spec.loader.get_source("_make_clip_ref")
    start, end = source.index("def synthetic("), source.index("# Portrait")
    from PIL import Image

    namespace = {"np": np, "Image": Image}
    exec(source[start:end], namespace)  # noqa: S102 - trusted repo file
    return namespace["synthetic"]


def test_matches_the_real_clip_image_processor():
    synthetic = _reference()
    ref = np.load(FIXTURE)
    index = 0
    while f"out_{index}" in ref:
        h, w, seed = (int(v) for v in ref[f"shape_{index}"])
        ours = pubmedclip_preprocess(synthetic(h, w, seed)).numpy()
        expected = ref[f"out_{index}"]
        assert ours.shape == expected.shape == (3, 224, 224)
        np.testing.assert_allclose(ours, expected, atol=1e-5)
        index += 1
    assert index == 5


@pytest.mark.parametrize(
    ("size", "expected"),
    [((200, 300), (224, 336)), ((320, 180), (398, 224)), ((224, 224), (224, 224))],
)
def test_shortest_edge_resize_keeps_the_aspect(size, expected):
    assert shortest_edge_size(*size) == expected


def test_output_is_three_identical_channels_of_a_grayscale_input_before_norm():
    synthetic = _reference()
    out = pubmedclip_preprocess(synthetic(300, 200, 0))
    assert out.dtype == torch.float32
    assert out.shape == (3, PUBMEDCLIP_IMG_SIZE, PUBMEDCLIP_IMG_SIZE)
    # Channels differ only by the per-channel mean/std.
    from vision_encoders.pubmedclip.preprocess import PUBMEDCLIP_MEAN, PUBMEDCLIP_STD

    raw = [out[c] * PUBMEDCLIP_STD[c] + PUBMEDCLIP_MEAN[c] for c in range(3)]
    assert torch.allclose(raw[0], raw[1], atol=1e-5)
    assert torch.allclose(raw[1], raw[2], atol=1e-5)


def test_mode_resolution():
    assert resolve_pubmedclip_preprocess(None) == "biovil_tensor"
    assert resolve_pubmedclip_preprocess("NATIVE") == "native"
    with pytest.raises(ValueError, match="preprocess"):
        resolve_pubmedclip_preprocess("clip")


def test_shipped_config_uses_native_preprocessing():
    yaml = pytest.importorskip("yaml")
    cfg = yaml.safe_load((REPO / "pretraining/configs/mimic_cxr_full.yaml").read_text())
    assert cfg["model"]["pubmedclip"]["preprocess"] == "native"


def test_model_never_hands_the_biovil_tensor_to_native_pubmedclip():
    """blip2_qformer cannot be imported on the CPU box; read its source."""
    source = (REPO / "model/lavis/models/blip2_models/blip2_qformer.py").read_text()
    # Every PubMedCLIP forward goes through the one helper that checks the mode.
    calls = [line for line in source.splitlines() if "self.pubmedclip(" in line]
    assert len(calls) == 2, calls
    helper = source[source.index("def _pubmedclip_tokens"):source.index("def _mask_stream")]
    assert "preprocessed=True" in helper and "raise ValueError" in helper
    for key in ("pubmedclip_image", "aux_pubmedclip_image"):
        assert f'samples.get("{key}")' in source
