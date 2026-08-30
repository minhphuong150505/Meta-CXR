"""Feature-grid coordinates to image coordinates. Pure tensor.

Like :mod:`training.explainability.rollout` this module imports nothing but
``torch``, so the geometry is checkable on a CPU box.

Which path produces a spatial map, and which does not
=====================================================

``medgemma_direct``
    MedGemma's own vision tower emits a square patch grid over the input image.
    Rolling attention from a generated token back to those patches lands on
    real image coordinates, so stage 1 alone yields a valid heatmap. **This is
    the only route in this branch that produces a spatial map**, which is why
    the module is built around it.

Q-Former routes (``meta_cxr_qformer*``)
    The 32 soft tokens are the Q-Former's ``query_output.last_hidden_state``.
    They carry NO position: each one has already cross-attended over all 246
    visual tokens, and the cross-attention weights that would relate a query
    token to an image region are not part of the cached Stage-2 record. Stage 1
    alone therefore says which *soft token* a sentence used, not which *region*.
    Turning that into a heatmap needs stage 2, which this branch does not
    implement. :func:`assert_spatial_projection_supported` refuses the
    conversion rather than emitting a map that would look meaningful and not be.

The Stage-1 coordinate frame
============================

The dataset applies ``Resize(512)`` on the shorter side then
``CenterCrop(448)``, producing one 3x448x448 tensor that every encoder reads.
BioViL-T tiles it 14x14 at 32 px per cell; PubMedCLIP resizes that same square
to 224 and tiles it 7x7 at 64 px per cell, plus one non-spatial CLS token. Both
therefore describe the *same* field of view.

VERIFIED on the training host 2026-08-30. PubMedCLIP's
``preprocessor_config.json`` reads ``do_resize: true, size: 224,
do_center_crop: true, crop_size: 224, resample: 3``. The dataset hands it a
SQUARE 448x448 tensor, so the resize takes it to 224x224 and the centre crop is
a no-op: a pure downscale of the same crop, no second crop, no offset. The two
encoders describe one field of view.

That is now evidence rather than inference, but it is still not left as a
comment: :func:`assert_shared_coordinate_frame` checks the arithmetic at import
time for the declared Stage-1 grids, and should be called again with whatever
geometry a live model actually reports -- a future config change to the crop
size would break the frame without touching this file.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

STAGE1_IMAGE_SIZE = 448

MODE_BILINEAR = "bilinear"
MODE_NEAREST = "nearest"
MODES = (MODE_BILINEAR, MODE_NEAREST)


class SpatialProjectionUnsupported(RuntimeError):
    """Raised when a token span has no spatial interpretation.

    A distinct type so a caller can tell "the Q-Former route cannot produce a
    heatmap" apart from an ordinary shape bug, and neither can be swallowed by
    a bare ``except RuntimeError``.
    """


@dataclass(frozen=True)
class GridSpec:
    """A feature grid and the image square it tiles."""

    height: int
    width: int
    patch_px: int

    def __post_init__(self) -> None:
        for name in ("height", "width", "patch_px"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")

    @property
    def num_tokens(self) -> int:
        return self.height * self.width

    @property
    def covered_px(self) -> tuple[int, int]:
        return (self.height * self.patch_px, self.width * self.patch_px)

    def to_dict(self) -> dict[str, int]:
        return {"height": self.height, "width": self.width, "patch_px": self.patch_px}


# What Stage 1 declares today. ``blip2_qformer._native_stream_layouts`` is the
# source of truth at runtime; these are the values it is expected to produce,
# and the import-time check below is what turns that expectation into an error
# rather than a comment.
STAGE1_GRIDS: dict[str, GridSpec] = {
    "biovil": GridSpec(height=14, width=14, patch_px=32),
    "pubmedclip": GridSpec(height=7, width=7, patch_px=64),
}


def assert_shared_coordinate_frame(
    grids: dict[str, GridSpec],
    image_size: int = STAGE1_IMAGE_SIZE,
) -> None:
    """Every grid must tile the SAME square exactly, or the frames disagree.

    14*32 == 7*64 == 448. If a future config breaks that -- a different crop
    size, an encoder reading an independently cropped image, a grid that leaves
    a remainder -- the two streams no longer share a coordinate system and any
    map overlaid on the image is wrong by an unknown offset. Fail here.
    """
    if not isinstance(image_size, int) or isinstance(image_size, bool) or image_size <= 0:
        raise ValueError(f"image_size must be a positive integer, got {image_size!r}")
    if not grids:
        raise ValueError("at least one grid is required")

    for name, spec in sorted(grids.items()):
        height_px, width_px = spec.covered_px
        if height_px != image_size or width_px != image_size:
            raise ValueError(
                f"grid {name!r} tiles {height_px}x{width_px} px but the image is "
                f"{image_size}x{image_size}: {spec.height}x{spec.width} cells of "
                f"{spec.patch_px} px do not cover the crop exactly, so this grid "
                f"does not share a coordinate frame with the others"
            )


# Run it on the declared Stage-1 geometry at import time. Cheap integer
# arithmetic, and it means the "shared coordinate frame" claim cannot rot into
# a stale comment while the constants drift.
assert_shared_coordinate_frame(STAGE1_GRIDS)


def infer_square_grid(num_tokens: int, patch_px: int | None = None) -> GridSpec:
    """Derive a square grid from a token count, refusing to guess.

    MedGemma's patch count is read from the live model rather than hardcoded
    here: the planning box cannot load it, and a wrong constant would silently
    reshape the map. ``patch_px`` is optional because the token count alone
    fixes the grid shape; supply it when the image size is known so
    :func:`assert_shared_coordinate_frame` can check the tiling too.
    """
    if not isinstance(num_tokens, int) or isinstance(num_tokens, bool) or num_tokens <= 0:
        raise ValueError(f"num_tokens must be a positive integer, got {num_tokens!r}")
    side = int(round(num_tokens**0.5))
    if side * side != num_tokens:
        raise ValueError(
            f"{num_tokens} visual tokens is not a perfect square, so the grid shape "
            "cannot be inferred; pass an explicit GridSpec. A non-square count "
            "usually means a global/CLS token is still attached and must be split "
            "off first"
        )
    return GridSpec(height=side, width=side, patch_px=patch_px if patch_px else 1)


def split_global_tokens(
    attribution: torch.Tensor,
    num_global_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Peel non-spatial tokens off the FRONT of a stream.

    Matches ``StreamLayout.num_global_tokens`` in ``mhcac/mhcac_12.py``:
    PubMedCLIP's CLS sits at index 0 and carries the global view, so it is kept
    and reported separately rather than smeared across the grid.
    """
    if not torch.is_tensor(attribution):
        raise TypeError("attribution must be a torch.Tensor")
    if attribution.ndim != 1:
        raise ValueError(f"attribution must be 1-D [N], got {tuple(attribution.shape)}")
    if (
        not isinstance(num_global_tokens, int)
        or isinstance(num_global_tokens, bool)
        or num_global_tokens < 0
    ):
        raise ValueError(f"num_global_tokens must be >= 0, got {num_global_tokens!r}")
    if num_global_tokens > attribution.shape[0]:
        raise ValueError(
            f"cannot split {num_global_tokens} global tokens off a stream of "
            f"{attribution.shape[0]}"
        )
    return attribution[:num_global_tokens], attribution[num_global_tokens:]


def assert_spatial_projection_supported(source: str) -> None:
    """Refuse to project a token span that has no spatial meaning.

    ``qformer_soft_token`` is rejected on purpose. Producing a 448x448 picture
    from 32 position-free query vectors would yield something that renders and
    means nothing, which is worse than an error.
    """
    if source == "qformer_soft_token":
        raise SpatialProjectionUnsupported(
            "Q-Former soft tokens carry no position: each has already "
            "cross-attended over every visual token, and those cross-attention "
            "weights are not in the Stage-2 record. Stage 1 alone identifies "
            "which soft token a sentence used, not which image region. A "
            "spatial map needs the Q-Former cross-attention stage, which this "
            "branch does not implement"
        )
    if not isinstance(source, str) or not source:
        raise ValueError("source must be a non-empty string")


def normalize_map(attribution: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Min-max normalise each map independently; a constant map becomes zero.

    Deliberately identical in behaviour to ``_normalize_cam`` in
    ``scripts/evaluate_explanation.py``, including the "constant maps become
    zero" rule -- a flat map carries no ranking, and stretching it to [0, 1]
    would invent one. Pinned against that function by
    ``tests/explainability/test_projection.py`` so the two cannot drift.
    """
    if not torch.is_tensor(attribution):
        raise TypeError("attribution must be a torch.Tensor")
    if attribution.ndim not in (2, 3):
        raise ValueError(
            f"attribution must be [H, W] or [B, H, W], got {tuple(attribution.shape)}"
        )
    if not torch.isfinite(attribution).all():
        raise ValueError("attribution contains non-finite values")

    squeeze = attribution.ndim == 2
    value = attribution.to(torch.float32)
    if squeeze:
        value = value[None]

    flat = value.reshape(value.shape[0], -1)
    minimum = flat.min(dim=1).values[:, None, None]
    maximum = flat.max(dim=1).values[:, None, None]
    scale = maximum - minimum
    normalized = torch.where(
        scale > eps,
        (value - minimum) / torch.where(scale > eps, scale, torch.ones_like(scale)),
        torch.zeros_like(value),
    )
    normalized = normalized.clamp(0.0, 1.0)
    return normalized[0] if squeeze else normalized


def project_to_image(
    attribution: torch.Tensor,
    grid: GridSpec,
    image_size: int = STAGE1_IMAGE_SIZE,
    *,
    mode: str = MODE_BILINEAR,
    normalize: bool = True,
    source: str = "vision_patch_grid",
) -> torch.Tensor:
    """Upsample a per-patch attribution vector to the image square.

    ``attribution`` is ``[N]`` or ``[B, N]`` with ``N == grid.num_tokens``, in
    row-major order -- the order the vision tower emits patches, and the order
    ``mhcac_12`` reshapes with (``spatial.reshape(B, h, w)``).

    Returns ``[image_size, image_size]`` or ``[B, image_size, image_size]``.
    Bilinear with ``align_corners=False`` matches
    ``scripts/evaluate_explanation.py``, so a Stage-1 CAM and a Stage-2 map are
    resampled the same way and can be compared.
    """
    assert_spatial_projection_supported(source)
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if not isinstance(grid, GridSpec):
        raise TypeError("grid must be a GridSpec")
    if not isinstance(image_size, int) or isinstance(image_size, bool) or image_size <= 0:
        raise ValueError(f"image_size must be a positive integer, got {image_size!r}")
    if not torch.is_tensor(attribution):
        raise TypeError("attribution must be a torch.Tensor")
    if attribution.ndim not in (1, 2):
        raise ValueError(
            f"attribution must be [N] or [B, N], got {tuple(attribution.shape)}"
        )
    if not torch.isfinite(attribution).all():
        raise ValueError("attribution contains non-finite values")

    squeeze = attribution.ndim == 1
    value = attribution.to(torch.float32)
    if squeeze:
        value = value[None]
    if value.shape[1] != grid.num_tokens:
        raise ValueError(
            f"attribution carries {value.shape[1]} values but the grid declares "
            f"{grid.height}x{grid.width} = {grid.num_tokens} patches; a mismatch "
            "here silently transposes or rolls the map, so it is refused"
        )

    maps = value.reshape(value.shape[0], 1, grid.height, grid.width)
    interpolate_kwargs = {"size": (image_size, image_size), "mode": mode}
    if mode == MODE_BILINEAR:
        interpolate_kwargs["align_corners"] = False
    upsampled = torch.nn.functional.interpolate(maps, **interpolate_kwargs).squeeze(1)

    if normalize:
        upsampled = normalize_map(upsampled)
    return upsampled[0] if squeeze else upsampled
