"""NumPy-only metrics for evaluating normalized Grad-CAM explanations.

The two supervision populations intentionally stay separate throughout this
module: ``mask_source=0`` is a lung anatomical prior, while ``mask_source=1``
is an expert MS-CXR bounding-box annotation.  There is deliberately no
``overall`` aggregate because averaging those populations would overstate the
clinical meaning of the much larger lung-mask cohort.

All CAM inputs must already be normalized to ``[0, 1]``.  Bounding boxes may be
provided either as binary masks with shape ``[num_boxes, H, W]`` or as
half-open ``(x0, y0, x1, y1)`` coordinates on the CAM grid.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

LUNG_MASK_SOURCE = 0
BBOX_MASK_SOURCE = 1


@dataclass(frozen=True)
class MetricResult:
    """One aggregate metric and the amount of evidence used to compute it."""

    value: float | None
    num_samples: int
    num_boxes: int | None = None
    unavailable_reason: str | None = None

    @property
    def available(self) -> bool:
        return self.value is not None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "value": self.value,
            "available": self.available,
            "num_samples": self.num_samples,
        }
        if self.num_boxes is not None:
            payload["num_boxes"] = self.num_boxes
        if self.unavailable_reason is not None:
            payload["unavailable_reason"] = self.unavailable_reason
        return payload


@dataclass(frozen=True)
class PopulationSummary:
    """Metrics for exactly one mask-source population."""

    mask_source: int
    population: str
    interpretation: str
    top_saliency_precision: MetricResult
    all_saliency_precision: MetricResult
    annotation_coverage: MetricResult

    def to_dict(self) -> dict[str, object]:
        return {
            "mask_source": self.mask_source,
            "population": self.population,
            "interpretation": self.interpretation,
            "top_saliency_precision": self.top_saliency_precision.to_dict(),
            "all_saliency_precision": self.all_saliency_precision.to_dict(),
            "annotation_coverage": self.annotation_coverage.to_dict(),
        }


@dataclass(frozen=True)
class ExplanationSummary:
    """A report with no mixed lung/bbox aggregate by design."""

    lung: PopulationSummary
    bbox: PopulationSummary

    def to_dict(self) -> dict[str, dict[str, object]]:
        return {
            "lung": self.lung.to_dict(),
            "bbox": self.bbox.to_dict(),
        }


def _validate_fraction(value: float, name: str, *, allow_zero: bool = False) -> float:
    value = float(value)
    lower_ok = value >= 0.0 if allow_zero else value > 0.0
    if not math.isfinite(value) or not lower_ok or value > 1.0:
        interval = "[0, 1]" if allow_zero else "(0, 1]"
        raise ValueError(f"{name} must be in {interval}")
    return value


def _validate_cam(cam: np.ndarray) -> np.ndarray:
    array = np.asarray(cam, dtype=np.float64)
    if array.ndim != 2 or array.size == 0:
        raise ValueError("cam must be a non-empty 2-D array")
    if not np.isfinite(array).all():
        raise ValueError("cam must contain only finite values")
    tolerance = 1e-7
    if np.any(array < -tolerance) or np.any(array > 1.0 + tolerance):
        raise ValueError("cam must already be normalized to [0, 1]")
    return np.clip(array, 0.0, 1.0)


def _validate_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(mask)
    if array.shape != shape:
        raise ValueError(f"mask must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("mask must contain only finite values")
    binary = array > 0
    if not binary.any():
        raise ValueError("mask must contain at least one annotated pixel")
    return binary


def _validate_mask_source(mask_source: int) -> int:
    if isinstance(mask_source, (bool, np.bool_)) or not isinstance(
        mask_source, (int, np.integer)
    ):
        raise ValueError("mask_source must be 0 (lung) or 1 (bbox)")
    value = int(mask_source)
    if value not in (LUNG_MASK_SOURCE, BBOX_MASK_SOURCE):
        raise ValueError("mask_source must be 0 (lung) or 1 (bbox)")
    return value


def _top_saliency_mask(cam: np.ndarray, k: float) -> np.ndarray:
    """Return exactly ``ceil(k * H * W)`` pixels, resolving ties by flat index."""

    cam = _validate_cam(cam)
    k = _validate_fraction(k, "k")
    flat = cam.reshape(-1)
    keep = max(1, int(math.ceil(k * flat.size)))

    # Equation (5) is binary.  Selecting by rank rather than ``>= quantile``
    # makes the requested cardinality exact even when many CAM values tie at
    # the cutoff.  The flat index is a deterministic secondary key only.
    order = np.lexsort((np.arange(flat.size, dtype=np.int64), -flat))
    selected = np.zeros(flat.size, dtype=bool)
    selected[order[:keep]] = True
    return selected.reshape(cam.shape)


def top_saliency_precision(
    cam: np.ndarray,
    mask: np.ndarray,
    k: float = 0.5,
) -> float:
    """Return Eq. (7): binary top-k salient pixels falling inside ``mask``."""

    cam_array = _validate_cam(cam)
    mask_array = _validate_mask(mask, cam_array.shape)
    salient = _top_saliency_mask(cam_array, k)
    return float(np.count_nonzero(salient & mask_array) / np.count_nonzero(salient))


def all_saliency_precision(cam: np.ndarray, mask: np.ndarray) -> float | None:
    """Return Eq. (8), or ``None`` when the CAM has no saliency mass."""

    cam_array = _validate_cam(cam)
    mask_array = _validate_mask(mask, cam_array.shape)
    total_mass = float(cam_array.sum(dtype=np.float64))
    if total_mass <= 0.0:
        return None
    inside_mass = float(cam_array[mask_array].sum(dtype=np.float64))
    return inside_mass / total_mass


def _coordinate_box_masks(
    coordinates: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    if coordinates.ndim != 2 or coordinates.shape[1] != 4:
        raise ValueError("box coordinates must have shape [num_boxes, 4]")
    if not np.isfinite(coordinates).all():
        raise ValueError("box coordinates must contain only finite values")

    height, width = shape
    masks = np.zeros((len(coordinates), height, width), dtype=bool)
    for index, (x0, y0, x1, y1) in enumerate(coordinates):
        if x1 <= x0 or y1 <= y0:
            raise ValueError("boxes must have positive width and height")
        left = max(0, min(width, math.floor(float(x0))))
        top = max(0, min(height, math.floor(float(y0))))
        right = max(0, min(width, math.ceil(float(x1))))
        bottom = max(0, min(height, math.ceil(float(y1))))
        if right <= left or bottom <= top:
            raise ValueError("a box does not overlap the CAM grid")
        masks[index, top:bottom, left:right] = True
    return masks


def _coerce_box_masks(
    boxes: Sequence[np.ndarray] | np.ndarray | None,
    shape: tuple[int, int],
) -> np.ndarray:
    if boxes is None:
        return np.zeros((0, *shape), dtype=bool)

    array = np.asarray(boxes)
    if array.size == 0:
        return np.zeros((0, *shape), dtype=bool)
    if array.ndim == 1 and array.shape[0] == 4:
        array = array.reshape(1, 4)
    if array.ndim == 2 and array.shape[1] == 4:
        return _coordinate_box_masks(array.astype(np.float64, copy=False), shape)
    if array.ndim != 3 or tuple(array.shape[1:]) != shape:
        raise ValueError(
            "boxes must be [num_boxes,4] coordinates or [num_boxes,H,W] masks"
        )
    if not np.isfinite(array).all():
        raise ValueError("box masks must contain only finite values")
    masks = array > 0
    if np.any(masks.reshape(len(masks), -1).sum(axis=1) == 0):
        raise ValueError("every box mask must contain at least one pixel")
    return masks


def _annotation_coverage_counts(
    cam: np.ndarray,
    boxes: Sequence[np.ndarray] | np.ndarray | None,
    *,
    k: float,
    tau: float,
) -> tuple[int, int]:
    cam_array = _validate_cam(cam)
    tau = _validate_fraction(tau, "tau", allow_zero=True)
    box_masks = _coerce_box_masks(boxes, cam_array.shape)
    if len(box_masks) == 0:
        return 0, 0

    salient = _top_saliency_mask(cam_array, k)
    box_areas = box_masks.reshape(len(box_masks), -1).sum(axis=1)
    salient_inside = (box_masks & salient[None, ...]).reshape(len(box_masks), -1).sum(axis=1)
    covered = int(np.count_nonzero(salient_inside / box_areas >= tau))
    return covered, len(box_masks)


def annotation_coverage(
    cam: np.ndarray,
    boxes: Sequence[np.ndarray] | np.ndarray | None,
    k: float = 0.5,
    tau: float = 0.01,
    *,
    mask_source: int = BBOX_MASK_SOURCE,
) -> float | None:
    """Return Eq. (9), evaluating each expert box independently.

    Lung masks carry no expert boxes, so ``mask_source=0`` is explicitly
    unavailable and returns ``None``.  Missing/empty boxes also return ``None``
    rather than the clinically misleading score ``0``.
    """

    source = _validate_mask_source(mask_source)
    if source == LUNG_MASK_SOURCE:
        return None
    covered, total = _annotation_coverage_counts(cam, boxes, k=k, tau=tau)
    return covered / total if total else None


def _aggregate(values: list[float]) -> MetricResult:
    if not values:
        return MetricResult(
            value=None,
            num_samples=0,
            unavailable_reason="no samples had a defined value",
        )
    return MetricResult(value=float(np.mean(values, dtype=np.float64)), num_samples=len(values))


def _population_summary(
    cams: np.ndarray,
    masks: np.ndarray,
    sources: np.ndarray,
    boxes_by_sample: Sequence[Sequence[np.ndarray] | np.ndarray | None],
    *,
    source: int,
    k: float,
    tau: float,
) -> PopulationSummary:
    indices = np.flatnonzero(sources == source)
    top_values: list[float] = []
    all_values: list[float] = []
    covered_boxes = 0
    total_boxes = 0
    coverage_samples = 0

    for index in indices:
        top_values.append(top_saliency_precision(cams[index], masks[index], k=k))
        all_value = all_saliency_precision(cams[index], masks[index])
        if all_value is not None:
            all_values.append(all_value)
        if source == BBOX_MASK_SOURCE:
            covered, total = _annotation_coverage_counts(
                cams[index], boxes_by_sample[index], k=k, tau=tau
            )
            if total:
                covered_boxes += covered
                total_boxes += total
                coverage_samples += 1

    if source == LUNG_MASK_SOURCE:
        population = "lung"
        interpretation = "anatomical prior: saliency remains inside the lungs"
        coverage = MetricResult(
            value=None,
            num_samples=0,
            num_boxes=0,
            unavailable_reason="annotation coverage requires separate expert boxes",
        )
    else:
        population = "bbox"
        interpretation = "expert annotation: saliency aligns with MS-CXR pathology boxes"
        coverage = MetricResult(
            value=(covered_boxes / total_boxes if total_boxes else None),
            num_samples=coverage_samples,
            num_boxes=total_boxes,
            unavailable_reason=(
                None if total_boxes else "no separate expert boxes were available"
            ),
        )

    return PopulationSummary(
        mask_source=source,
        population=population,
        interpretation=interpretation,
        top_saliency_precision=_aggregate(top_values),
        all_saliency_precision=_aggregate(all_values),
        annotation_coverage=coverage,
    )


def summarize(
    cams: np.ndarray,
    masks: np.ndarray,
    mask_sources: Sequence[int] | np.ndarray,
    boxes_by_sample: Sequence[Sequence[np.ndarray] | np.ndarray | None] | None = None,
    *,
    k: float = 0.5,
    tau: float = 0.01,
) -> ExplanationSummary:
    """Summarize metrics without ever pooling lung and bbox populations."""

    cam_array = np.asarray(cams)
    mask_array = np.asarray(masks)
    sources = np.asarray(mask_sources)
    if cam_array.ndim != 3:
        raise ValueError("cams must have shape [N,H,W]")
    if mask_array.shape != cam_array.shape:
        raise ValueError("masks must have the same [N,H,W] shape as cams")
    if sources.ndim != 1 or len(sources) != len(cam_array):
        raise ValueError("mask_sources must contain one value per CAM")
    for source in sources:
        _validate_mask_source(source)

    if boxes_by_sample is None:
        boxes: list[Sequence[np.ndarray] | np.ndarray | None] = [None] * len(cam_array)
    else:
        boxes = list(boxes_by_sample)
        if len(boxes) != len(cam_array):
            raise ValueError("boxes_by_sample must contain one entry per CAM")

    # Validate parameters even for an empty population, so a typo never hides
    # behind a split that happens not to contain bbox samples.
    k = _validate_fraction(k, "k")
    tau = _validate_fraction(tau, "tau", allow_zero=True)

    return ExplanationSummary(
        lung=_population_summary(
            cam_array,
            mask_array,
            sources,
            boxes,
            source=LUNG_MASK_SOURCE,
            k=k,
            tau=tau,
        ),
        bbox=_population_summary(
            cam_array,
            mask_array,
            sources,
            boxes,
            source=BBOX_MASK_SOURCE,
            k=k,
            tau=tau,
        ),
    )


__all__ = [
    "BBOX_MASK_SOURCE",
    "LUNG_MASK_SOURCE",
    "ExplanationSummary",
    "MetricResult",
    "PopulationSummary",
    "all_saliency_precision",
    "annotation_coverage",
    "summarize",
    "top_saliency_precision",
]
