"""Image perturbations for counterfactual auditing.

Each perturbation destroys a different kind of visual information, so that the
change in the generated report tells you *which* kind the model was actually
using:

* ``no_image`` / ``blank_image`` / ``constant_image`` -- remove the image
  entirely. A report that barely changes is being written from the language
  prior and the prompt, not from the radiograph.
* ``shuffled_pixels`` -- destroys all spatial structure, preserves the intensity
  histogram. Isolates "is it using anatomy, or just overall exposure?".
* ``shuffled_patches`` -- destroys global anatomy, preserves local texture.
* ``random_image_swap`` / ``hard_negative_swap`` -- substitute a *different real*
  study. The strongest test: output should change substantially, and a hard
  negative (same anatomy, different finding) should still move the report.
* ``region_occlusion`` -- blanks one region, for localisation claims.

Every perturbation is deterministic given its seed, so an audit is reproducible.
Depends on torch only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

NO_IMAGE = "no_image"
BLANK_IMAGE = "blank_image"
CONSTANT_IMAGE = "constant_image"
SHUFFLED_PIXELS = "shuffled_pixels"
SHUFFLED_PATCHES = "shuffled_patches"
RANDOM_IMAGE_SWAP = "random_image_swap"
HARD_NEGATIVE_SWAP = "hard_negative_swap"
REGION_OCCLUSION = "region_occlusion"

#: Perturbations that need only the sample's own image.
SELF_CONTAINED = (
    NO_IMAGE,
    BLANK_IMAGE,
    CONSTANT_IMAGE,
    SHUFFLED_PIXELS,
    SHUFFLED_PATCHES,
    REGION_OCCLUSION,
)
#: Perturbations that need a donor image from elsewhere in the cohort.
NEEDS_DONOR = (RANDOM_IMAGE_SWAP, HARD_NEGATIVE_SWAP)

ALL_PERTURBATIONS = SELF_CONTAINED + NEEDS_DONOR


def _check_image(image: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(image):
        raise TypeError(f"expected an image tensor, got {type(image).__name__}")
    if image.dim() != 3:
        raise ValueError(f"expected an image of shape [C, H, W], got {tuple(image.shape)}")
    return image


def blank_image(image: torch.Tensor) -> torch.Tensor:
    """All zeros, same shape/dtype/device."""
    return torch.zeros_like(_check_image(image))


def constant_image(image: torch.Tensor) -> torch.Tensor:
    """Uniform fill at the image's own mean.

    Unlike ``blank_image`` this keeps the overall exposure level, so a model
    that keys off mean intensity alone is not disturbed by it.
    """
    image = _check_image(image)
    return torch.full_like(image, float(image.mean()))


def shuffled_pixels(image: torch.Tensor, seed: int) -> torch.Tensor:
    """Permute pixels within each channel: histogram preserved, anatomy gone."""
    image = _check_image(image)
    generator = torch.Generator().manual_seed(int(seed))
    channels, height, width = image.shape
    flat = image.reshape(channels, height * width)
    order = torch.randperm(height * width, generator=generator)
    return flat[:, order].reshape(channels, height, width)


def shuffled_patches(image: torch.Tensor, seed: int, patch: int = 16) -> torch.Tensor:
    """Permute non-overlapping patches: local texture kept, global layout gone.

    Any right/bottom remainder that does not fill a whole patch is left in place
    rather than silently cropped, so the output is always the input's shape.
    """
    image = _check_image(image)
    if patch < 1:
        raise ValueError("patch size must be positive")
    channels, height, width = image.shape
    rows, cols = height // patch, width // patch
    if rows < 1 or cols < 1:
        raise ValueError(f"image {height}x{width} is smaller than one {patch}x{patch} patch")

    generator = torch.Generator().manual_seed(int(seed))
    order = torch.randperm(rows * cols, generator=generator)
    out = image.clone()
    grid = [
        image[:, r * patch : (r + 1) * patch, c * patch : (c + 1) * patch]
        for r in range(rows)
        for c in range(cols)
    ]
    for destination, source in enumerate(order.tolist()):
        r, c = divmod(destination, cols)
        out[:, r * patch : (r + 1) * patch, c * patch : (c + 1) * patch] = grid[source]
    return out


def region_occlusion(
    image: torch.Tensor, fraction: float = 0.25, corner: str = "top_left"
) -> torch.Tensor:
    """Zero one rectangular region covering ``fraction`` of the area."""
    image = _check_image(image)
    if not 0 < fraction < 1:
        raise ValueError("fraction must be in (0, 1)")
    _, height, width = image.shape
    side = fraction**0.5
    box_h, box_w = max(1, int(height * side)), max(1, int(width * side))
    out = image.clone()
    if corner == "top_left":
        out[:, :box_h, :box_w] = 0
    elif corner == "top_right":
        out[:, :box_h, width - box_w :] = 0
    elif corner == "bottom_left":
        out[:, height - box_h :, :box_w] = 0
    elif corner == "bottom_right":
        out[:, height - box_h :, width - box_w :] = 0
    else:
        raise ValueError(f"unknown corner {corner!r}")
    return out


@dataclass(frozen=True)
class Donor:
    """A different study's image, substituted for the original."""

    sample_key: str
    image: torch.Tensor


def pick_random_donor(
    sample_key: str, cohort: Sequence[Donor], seed: int
) -> Donor:
    """Deterministically choose a donor that is not the sample itself.

    Raises rather than returning the sample's own image: a "swap" that silently
    swapped nothing would report the model as image-insensitive when in fact no
    swap occurred.
    """
    candidates = [donor for donor in cohort if donor.sample_key != sample_key]
    if not candidates:
        raise ValueError(
            f"no donor available for {sample_key!r}: a swap perturbation needs at "
            "least one other study in the cohort"
        )
    generator = torch.Generator().manual_seed(int(seed))
    index = int(torch.randint(len(candidates), (1,), generator=generator).item())
    return candidates[index]


def apply_self_contained(name: str, image: torch.Tensor, seed: int) -> torch.Tensor | None:
    """Apply a perturbation that needs only this sample's image."""
    if name == NO_IMAGE:
        return None
    if name == BLANK_IMAGE:
        return blank_image(image)
    if name == CONSTANT_IMAGE:
        return constant_image(image)
    if name == SHUFFLED_PIXELS:
        return shuffled_pixels(image, seed)
    if name == SHUFFLED_PATCHES:
        return shuffled_patches(image, seed)
    if name == REGION_OCCLUSION:
        return region_occlusion(image)
    raise ValueError(f"unknown or donor-requiring perturbation {name!r}")
