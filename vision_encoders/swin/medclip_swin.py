"""MedCLIP Swin-Tiny: the Swin stream the META-CXR paper uses.

Wang et al., "MedCLIP: Contrastive Learning from Unpaired Medical Images and
Text", EMNLP 2022 (github.com/RyanWangZf/MedCLIP). Its vision tower is HF
``SwinModel`` from ``microsoft/swin-tiny-patch4-window7-224`` (``VIT_TYPE`` in
``medclip/constants.py``), fine-tuned on MIMIC-CXR + CheXpert, followed by a
768 -> 512 ``projection_head`` that this repo does NOT use.

Two things here are easy to get wrong, and both were wrong in
``vision_encoders/medclip/medclip.py`` (inherited from the original META-CXR
code, never wired in):

* ``MedCLIPVisionModelViT.forward`` returns ONE tensor, the 512-d projected
  pooled vector. Unpacking it as ``pool, patches = ...`` splits the batch along
  dim 0. The tokens come from the inner HF ``SwinModel`` instead:
  ``last_hidden_state`` [B, 49, 768] and ``pooler_output`` [B, 768], joined here
  as [B, 50, 768] with the pooled vector at position 0 -- the paper's "50 tokens
  including the feature vector aggregated through global pooling".
* The ``medclip`` package cannot preprocess under transformers >= 4.25: its
  ``MedCLIPFeatureExtractor`` passes ``CLIPFeatureExtractor`` arguments by
  position, and the positions moved (measured under 4.53.2: ``rescale_factor``
  receives the mean, ``image_mean`` receives ``False``, and ``__call__`` dies on
  ``convert_rgb``). :func:`medclip_preprocess` reimplements the 4.24 semantics;
  ``tests/test_medclip_swin.py`` pins it against outputs of the real processor
  captured under transformers 4.24.0.

Pure torch/numpy/PIL at import time; transformers is imported only when a model
is built.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

MEDCLIP_VIT_TYPE = "microsoft/swin-tiny-patch4-window7-224"
MEDCLIP_IMG_SIZE = 224
MEDCLIP_IMG_MEAN = 0.5862785803043838
MEDCLIP_IMG_STD = 0.27950088968644304
MEDCLIP_WEIGHTS_URL_FILE = (
    "https://github.com/RyanWangZf/MedCLIP/raw/main/medclip/medclip_vit_weight.txt"
)
#: Tokens per image: 1 pooled + 7x7 patches.
MEDCLIP_NUM_TOKENS = 50
MEDCLIP_EMBED_DIM = 768
_VISION_PREFIX = "vision_model.model."


def pad_to_square(image: Image.Image, min_size: int = MEDCLIP_IMG_SIZE) -> Image.Image:
    """``MedCLIPFeatureExtractor.pad_img``: centre the image on a black square."""
    if image.mode != "L":
        image = image.convert("L")
    width, height = image.size
    size = max(min_size, width, height)
    padded = Image.new("L", (size, size), 0)
    padded.paste(image, (int((size - width) / 2), int((size - height) / 2)))
    return padded


def medclip_preprocess(image: Image.Image) -> torch.Tensor:
    """Grayscale PIL radiograph -> [3, 224, 224] float32, MedCLIP-normalised.

    pad to square (black) -> resize 224x224 bicubic -> centre crop 224 (a no-op
    on a square) -> /255 -> (x - mean) / std -> repeat to 3 channels, which is
    what ``MedCLIPVisionModelViT.forward`` does to a 1-channel input.
    """
    square = pad_to_square(image)
    resized = square.resize((MEDCLIP_IMG_SIZE, MEDCLIP_IMG_SIZE), Image.BICUBIC)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    array = (array - MEDCLIP_IMG_MEAN) / MEDCLIP_IMG_STD
    tensor = torch.from_numpy(np.ascontiguousarray(array))
    return tensor.unsqueeze(0).repeat(3, 1, 1)


def join_pooled_and_patches(pooled: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
    """[B, D] + [B, P, D] -> [B, 1 + P, D], pooled vector first."""
    if pooled.ndim != 2 or patches.ndim != 3 or pooled.shape[0] != patches.shape[0]:
        raise ValueError(
            f"expected pooled [B, D] and patches [B, P, D], got "
            f"{tuple(pooled.shape)} and {tuple(patches.shape)}"
        )
    return torch.cat([pooled.unsqueeze(1), patches], dim=1)


def vision_state_dict(medclip_state: dict) -> dict:
    """Keep ``vision_model.model.*`` from a MedCLIP checkpoint, prefix stripped."""
    return {
        key[len(_VISION_PREFIX):]: value
        for key, value in medclip_state.items()
        if key.startswith(_VISION_PREFIX)
    }


def resolve_weights_path(weights_path: str | os.PathLike | None) -> Path:
    if not weights_path:
        raise ValueError(
            "model.swin.weights_path is required for backend 'medclip': the "
            "MedCLIP-ViT pytorch_model.bin (unzipped from the URL listed in "
            f"{MEDCLIP_WEIGHTS_URL_FILE})"
        )
    path = Path(os.path.expandvars(os.path.expanduser(str(weights_path))))
    if not path.is_file():
        raise FileNotFoundError(
            f"MedCLIP weights not found at {path}. Download the zip named in "
            f"{MEDCLIP_WEIGHTS_URL_FILE}, unzip it, and point "
            "model.swin.weights_path at its pytorch_model.bin."
        )
    return path


def build_medclip_swin(weights_path, backbone: str = MEDCLIP_VIT_TYPE):
    """HF ``SwinModel`` with MedCLIP's fine-tuned weights loaded, strictly.

    Every backbone key must be present in the checkpoint and vice versa; a
    missing key would leave that tensor at its ImageNet value with no error.
    Returns ``(model, report)``; ``report`` records the key accounting.
    """
    from transformers import AutoModel

    model = AutoModel.from_pretrained(backbone)
    path = resolve_weights_path(weights_path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    vision = vision_state_dict(state)
    if not vision:
        raise ValueError(f"{path} holds no '{_VISION_PREFIX}*' keys; not a MedCLIP-ViT checkpoint")
    result = model.load_state_dict(vision, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(
            f"MedCLIP Swin weights do not match {backbone}: "
            f"missing {result.missing_keys[:10]} ({len(result.missing_keys)}), "
            f"unexpected {result.unexpected_keys[:10]} ({len(result.unexpected_keys)})"
        )
    report = {
        "weights_path": str(path),
        "backbone": backbone,
        "vision_keys_loaded": len(vision),
        "missing_keys": 0,
        "unexpected_keys": 0,
        "other_checkpoint_keys_ignored": len(state) - len(vision),
    }
    return model, report
