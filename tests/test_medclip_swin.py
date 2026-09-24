"""MedCLIP Swin-Tiny: preprocessing, token layout, and the inherited wrapper bug.

The preprocessing reference in tests/fixtures/medclip_preprocess_reference.npz
was produced by ``MedCLIPFeatureExtractor`` (medclip 0.0.3) running under
transformers 4.24.0 -- the version medclip pins -- on SYNTHETIC images, by
scripts/make_medclip_preprocess_reference.py. The package's own processor
cannot run under this repo's transformers 4.53 (its positional arguments land in
the wrong slots), which is why the reference is a stored fixture.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from vision_encoders.swin.medclip_swin import (
    MEDCLIP_IMG_MEAN,
    MEDCLIP_IMG_STD,
    MEDCLIP_NUM_TOKENS,
    join_pooled_and_patches,
    medclip_preprocess,
    pad_to_square,
    resolve_weights_path,
    vision_state_dict,
)

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests/fixtures/medclip_preprocess_reference.npz"


def _reference_module():
    """The synthetic-image generator the fixture was built with."""
    spec = importlib.util.spec_from_file_location(
        "_make_ref", REPO / "scripts/make_medclip_preprocess_reference.py"
    )
    source = spec.loader.get_source("_make_ref")
    # Only the generator is needed; executing the whole script would import
    # transformers 4.24 and write a file.
    start = source.index("def synthetic(")
    end = source.index("SHAPES = ")
    namespace = {"np": np, "Image": Image}
    exec(source[start:end], namespace)  # noqa: S102 - trusted repo file
    return namespace["synthetic"]


def test_preprocessing_matches_medclip_processor_under_transformers_4_24():
    synthetic = _reference_module()
    ref = np.load(FIXTURE)
    assert str(ref["transformers_version"]) == "4.24.0"
    index = 0
    while f"out_{index}" in ref:
        h, w, seed = (int(v) for v in ref[f"shape_{index}"])
        expected = ref[f"out_{index}"]                       # [1, 224, 224]
        got = medclip_preprocess(synthetic(h, w, seed))       # [3, 224, 224]
        assert got.shape == (3, 224, 224)
        for channel in range(3):
            np.testing.assert_allclose(got[channel].numpy(), expected[0], atol=1e-5)
        index += 1
    assert index >= 4, "the fixture should cover portrait, landscape, square and small"


def test_padding_centres_on_black_and_never_crops():
    image = Image.fromarray(np.full((300, 100), 200, dtype=np.uint8), mode="L")
    square = pad_to_square(image)
    assert square.size == (300, 300)
    array = np.asarray(square)
    assert array[:, :100].max() == 0 and array[:, 200:].max() == 0
    assert (array[:, 100:200] == 200).all()


def test_normalisation_constants_are_medclips():
    assert MEDCLIP_IMG_MEAN == pytest.approx(0.5862785803043838)
    assert MEDCLIP_IMG_STD == pytest.approx(0.27950088968644304)
    black = medclip_preprocess(Image.new("L", (224, 224), 0))
    assert float(black.max()) == pytest.approx(-MEDCLIP_IMG_MEAN / MEDCLIP_IMG_STD, abs=1e-5)


def test_fifty_tokens_with_the_pooled_vector_first():
    pooled = torch.randn(2, 768)
    patches = torch.randn(2, 49, 768)
    joined = join_pooled_and_patches(pooled, patches)
    assert joined.shape == (2, MEDCLIP_NUM_TOKENS, 768)
    assert torch.equal(joined[:, 0], pooled)
    assert torch.equal(joined[:, 1:], patches)
    with pytest.raises(ValueError):
        join_pooled_and_patches(pooled, patches[:1])


def test_only_the_vision_backbone_is_taken_from_a_medclip_checkpoint():
    state = {
        "vision_model.model.embeddings.patch_embeddings.projection.weight": torch.zeros(1),
        "vision_model.projection_head.weight": torch.zeros(1),
        "text_model.model.embeddings.position_ids": torch.zeros(1),
        "logit_scale": torch.zeros(()),
    }
    assert list(vision_state_dict(state)) == ["embeddings.patch_embeddings.projection.weight"]


def test_missing_weights_fail_loudly(tmp_path):
    with pytest.raises(ValueError, match="weights_path is required"):
        resolve_weights_path(None)
    with pytest.raises(FileNotFoundError, match="MedCLIP weights not found"):
        resolve_weights_path(tmp_path / "nope.bin")


# --------------------------------------------------------------------------
# The inherited wrapper: vision_encoders/medclip/medclip.py
# --------------------------------------------------------------------------


class _FakeVisionModelViT(torch.nn.Module):
    """Returns what MedCLIPVisionModelViT.forward returns: ONE [B, 512] tensor."""

    def forward(self, pixel_values, project=True):
        return torch.randn(pixel_values.shape[0], 512)


def _load_old_wrapper(monkeypatch):
    fake = types.ModuleType("medclip")

    class MedCLIPModel(torch.nn.Module):
        def __init__(self, vision_cls=None):
            super().__init__()
            self.vision_model = _FakeVisionModelViT()

        def from_pretrained(self):
            return None

    class MedCLIPProcessor:
        def __call__(self, images, return_tensors="pt"):
            return {"pixel_values": images}

    fake.MedCLIPModel = MedCLIPModel
    fake.MedCLIPVisionModelViT = object
    fake.MedCLIPProcessor = MedCLIPProcessor
    monkeypatch.setitem(sys.modules, "medclip", fake)
    spec = importlib.util.spec_from_file_location(
        "_old_medclip_wrapper", REPO / "vision_encoders/medclip/medclip.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Medclip(device="cpu")


@pytest.mark.parametrize("batch", [2, 3])
def test_the_inherited_wrapper_cannot_produce_patch_tokens(monkeypatch, batch):
    """It unpacks ONE [B, 512] tensor into (pool, patches), i.e. along the batch.

    Batch 2 "succeeds" at the unpack -- two 512-vectors from different studies --
    and fails at the concat; batch 3 fails at the unpack. Neither path can ever
    yield 49 patch tokens. Recorded in DECISIONS as inherited from the
    original META-CXR code; the repo's medclip backend reads the HF SwinModel
    inside instead.
    """
    wrapper = _load_old_wrapper(monkeypatch)
    with pytest.raises((ValueError, RuntimeError, IndexError)):
        wrapper(torch.randn(batch, 3, 224, 224))


def test_real_weights_give_fifty_medclip_tokens_when_available():
    """Host-only: needs transformers and the downloaded MedCLIP weights."""
    pytest.importorskip("transformers")
    weights = Path("~/medclip_work/pretrained/medclip-vit/pytorch_model.bin").expanduser()
    if not weights.is_file():
        pytest.skip("MedCLIP weights not downloaded on this machine")
    from vision_encoders.swin.swin_encoder import SwinEncoder

    encoder = SwinEncoder(
        model_name="microsoft/swin-tiny-patch4-window7-224",
        backend="medclip",
        normalize=False,
        weights_path=str(weights),
    ).eval()
    assert encoder.load_report["vision_keys_loaded"] == 231
    x = torch.stack([medclip_preprocess(Image.new("L", (300, 200), 90))] * 2)
    with torch.no_grad():
        tokens = encoder(x)
        out = encoder.model(pixel_values=x)
    assert tokens.shape == (2, 50, 768)
    assert torch.allclose(tokens[:, 0], out.pooler_output)
    with pytest.raises(ValueError, match="must not be fed"):
        encoder(torch.randn(2, 3, 448, 448))
