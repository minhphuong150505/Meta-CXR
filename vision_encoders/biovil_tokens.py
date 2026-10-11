"""Turn BioViL-T's projected patch grid into a token sequence.

``ImageModel.forward(...).projected_patch_embeddings`` is channel-first,
``[B, D, H, W]``. Two layouts exist:

* ``legacy_reshape`` -- ``grid.reshape(B, -1, D)``, what the reference META-CXR
  code and every checkpoint up to 2026-10-11 use. It does NOT move the channel
  axis, so each "token" is a flattened group of ``D / (H*W)`` channel maps and a
  true patch is spread over all tokens. Kept as the default so existing
  checkpoints and feature caches keep their meaning.
* ``patch`` -- ``grid.flatten(2).transpose(1, 2)``: token ``k`` is the embedding
  of patch ``k`` in row-major order, which is what the positional encodings and
  the 14 x 14 layout assume.

A checkpoint is only valid under the layout it was trained with.
"""

from __future__ import annotations

LEGACY_RESHAPE = "legacy_reshape"
PATCH = "patch"
BIOVIL_TOKEN_LAYOUTS = (LEGACY_RESHAPE, PATCH)


def resolve_biovil_token_layout(value) -> str:
    """``None`` (key absent) is the historical layout; anything unknown raises."""
    if value is None:
        return LEGACY_RESHAPE
    value = str(value)
    if value not in BIOVIL_TOKEN_LAYOUTS:
        raise ValueError(
            f"model.biovil.token_layout must be one of {BIOVIL_TOKEN_LAYOUTS}, got {value!r}"
        )
    return value


def biovil_grid_to_tokens(grid, layout: str):
    """``[B, D, H, W]`` -> ``[B, H*W, D]`` under ``layout``."""
    if grid.ndim != 4:
        raise ValueError(f"expected a [B, D, H, W] grid, got shape {tuple(grid.shape)}")
    if layout == PATCH:
        return grid.flatten(2).transpose(1, 2)
    if layout == LEGACY_RESHAPE:
        return grid.reshape(grid.shape[0], -1, grid.shape[1])
    raise ValueError(f"unknown BioViL token layout {layout!r}")
