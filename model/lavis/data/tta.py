"""One fixed test-time affine on the raw radiograph (opt-in, evaluation splits only).

Only PIL, so the CPU test suite can pin it without torchvision or LAVIS.

Enabled by ``datasets.mimic_cxr.vis_processor.eval.tta`` in the run config;
absent means no transform at all, bit-identical to before::

    datasets:
      mimic_cxr:
        vis_processor:
          eval:
            tta: {degrees: 5.0, scale: 1.0}

The transform is applied in ``MIMIC_CXR_Dataset._row_visual`` to the decoded
image BEFORE every stream's own preprocessing (BioViL resize + crop, PubMedCLIP,
MedCLIP Swin), so all encoders see the same transformed radiograph. One
evaluation pass applies one transform; test-time augmentation is the mean of
the per-pass probabilities, computed outside the model
(``docs/handoff/PLAN-2026-10-30-stage1-tta.md``). Never applied to ``train``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from PIL import Image

TTA_KEYS = ("degrees", "scale")


@dataclass(frozen=True)
class TTAAffine:
    degrees: float  # counter-clockwise, as PIL ``Image.rotate``
    scale: float  # about the image centre; > 1 zooms in


def resolve_tta_affine(raw, split: str) -> TTAAffine | None:
    """``None`` when the key is absent; refuses a train split and unknown keys."""
    if raw is None:
        return None
    if split == "train":
        raise ValueError("vis_processor tta is evaluation-only; it was configured for the train split")
    if not isinstance(raw, Mapping):
        raise ValueError(f"vis_processor tta must be a mapping with keys {TTA_KEYS}, got {raw!r}")
    unknown = set(raw) - set(TTA_KEYS)
    if unknown:
        raise ValueError(f"unknown vis_processor tta keys {sorted(unknown)}; expected {TTA_KEYS}")
    degrees = float(raw.get("degrees", 0.0))
    scale = float(raw.get("scale", 1.0))
    if not math.isfinite(degrees) or not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"vis_processor tta needs finite degrees and scale > 0, got {degrees}, {scale}")
    return TTAAffine(degrees=degrees, scale=scale)


def apply_tta_affine(image: Image.Image, tta: TTAAffine) -> Image.Image:
    """Rotate and scale about the centre in ONE bilinear resampling; corners filled with 0.

    The inverse map mirrors ``Image.rotate``'s own matrix, divided by ``scale``.
    """
    w, h = image.size
    cx, cy = w * 0.5, h * 0.5
    a = -math.radians(tta.degrees)
    m = [math.cos(a) / tta.scale, math.sin(a) / tta.scale, 0.0,
         -math.sin(a) / tta.scale, math.cos(a) / tta.scale, 0.0]
    m[2] = cx - (m[0] * cx + m[1] * cy)
    m[5] = cy - (m[3] * cx + m[4] * cy)
    return image.transform((w, h), Image.AFFINE, m, resample=Image.BILINEAR, fillcolor=0)
