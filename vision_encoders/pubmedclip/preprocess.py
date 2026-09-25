"""PubMedCLIP's own preprocessing, applied to the raw radiograph (D-022).

Until 2026-09-25 PubMedCLIP read the BioViL tensor: resize shorter side 512,
centre crop 448, [0,1], 1 channel copied to 3 -- then ``CLIPImageProcessorFast``
downscaled that 448 square to 224 and normalised it. So the encoder saw BioViL's
crop, not its own. The paper preprocesses each frozen encoder its own way, and
MedCLIP Swin already does (``vision_encoders/swin/medclip_swin.py``).

This reimplements the slow ``CLIPImageProcessor`` exactly as configured by
``flaviagiammarino/pubmed-clip-vit-base-patch32/preprocessor_config.json``
(read on the host 2026-09-25):

    size 224 (shortest edge), resample 3 (bicubic), centre crop 224,
    rescale 1/255, normalise with the OpenAI CLIP mean/std, RGB.

Pure numpy/PIL/torch so the dataset worker needs no transformers import.
``tests/test_pubmedclip_preprocess.py`` pins it against outputs of the real
processor captured on the host.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

PUBMEDCLIP_IMG_SIZE = 224
PUBMEDCLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
PUBMEDCLIP_STD = (0.26862954, 0.26130258, 0.27577711)

#: ``model.pubmedclip.preprocess`` values. ``biovil_tensor`` is the historical
#: path and what a config without the key gets; ``native`` is this module.
PREPROCESS_MODES = ("biovil_tensor", "native")


def resolve_pubmedclip_preprocess(value) -> str:
    mode = "biovil_tensor" if value is None else str(value).lower()
    if mode not in PREPROCESS_MODES:
        raise ValueError(
            f"model.pubmedclip.preprocess must be one of {PREPROCESS_MODES}, got {value!r}"
        )
    return mode


def shortest_edge_size(width: int, height: int, size: int = PUBMEDCLIP_IMG_SIZE):
    """transformers ``get_resize_output_image_size(default_to_square=False)``."""
    short, long = (width, height) if width <= height else (height, width)
    new_short, new_long = size, int(size * long / short)
    return (new_short, new_long) if width <= height else (new_long, new_short)


def pubmedclip_preprocess(image: Image.Image) -> torch.Tensor:
    """PIL radiograph -> [3, 224, 224] float32, CLIP-normalised."""
    rgb = image.convert("RGB")
    new_w, new_h = shortest_edge_size(*rgb.size)
    resized = np.asarray(rgb.resize((new_w, new_h), resample=Image.BICUBIC))
    top = (new_h - PUBMEDCLIP_IMG_SIZE) // 2
    left = (new_w - PUBMEDCLIP_IMG_SIZE) // 2
    crop = resized[top:top + PUBMEDCLIP_IMG_SIZE, left:left + PUBMEDCLIP_IMG_SIZE]
    array = crop.astype(np.float32) * np.float32(1 / 255)
    array = (array - np.asarray(PUBMEDCLIP_MEAN, dtype=np.float32)) / np.asarray(
        PUBMEDCLIP_STD, dtype=np.float32
    )
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))
