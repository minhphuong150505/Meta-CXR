from __future__ import annotations

import numpy as np
import pytest

from training.evaluation.explanation_metrics import (
    BBOX_MASK_SOURCE,
    LUNG_MASK_SOURCE,
    all_saliency_precision,
    annotation_coverage,
    summarize,
    top_saliency_precision,
)


def test_saliency_precision_is_one_inside_and_zero_outside():
    cam = np.array([[0.9, 0.8], [0.0, 0.0]], dtype=np.float64)
    inside = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    outside = 1 - inside

    assert top_saliency_precision(cam, inside, k=0.5) == pytest.approx(1.0)
    assert all_saliency_precision(cam, inside) == pytest.approx(1.0)
    assert top_saliency_precision(cam, outside, k=0.5) == pytest.approx(0.0)
    assert all_saliency_precision(cam, outside) == pytest.approx(0.0)


def test_top_saliency_precision_keeps_exactly_k_percent_even_with_ties():
    cam = np.ones((4, 4), dtype=np.float64)
    first_six = np.zeros_like(cam, dtype=np.uint8)
    first_six.reshape(-1)[:6] = 1

    # k=3/8 means exactly six of sixteen pixels. With every value tied, the
    # deterministic flat-index tiebreak makes those six pixels observable.
    assert top_saliency_precision(cam, first_six, k=3 / 8) == pytest.approx(1.0)


def test_annotation_coverage_uses_inclusive_tau_threshold_per_box():
    cam = np.zeros((10, 10), dtype=np.float64)
    cam[0, 0] = 1.0
    full_grid_box = np.ones((1, 10, 10), dtype=np.uint8)

    assert annotation_coverage(cam, full_grid_box, k=0.01, tau=0.01) == pytest.approx(1.0)
    assert annotation_coverage(cam, full_grid_box, k=0.01, tau=0.011) == pytest.approx(0.0)


def test_annotation_coverage_counts_each_box_instead_of_their_union():
    cam = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    boxes = np.array(
        [
            [[1, 0], [0, 0]],
            [[0, 0], [0, 1]],
        ],
        dtype=np.uint8,
    )

    assert annotation_coverage(cam, boxes, k=0.25, tau=0.01) == pytest.approx(0.5)


def test_annotation_coverage_is_unavailable_for_lung_masks():
    cam = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float64)

    assert annotation_coverage(
        cam,
        boxes=None,
        mask_source=LUNG_MASK_SOURCE,
    ) is None


def test_summary_keeps_lung_and_bbox_populations_separate():
    cams = np.array(
        [
            [[0.9, 0.8], [0.0, 0.0]],
            [[0.0, 0.0], [0.8, 0.9]],
        ],
        dtype=np.float64,
    )
    masks = np.array(
        [
            [[1, 1], [0, 0]],
            [[1, 1], [0, 0]],
        ],
        dtype=np.uint8,
    )
    sources = np.array([LUNG_MASK_SOURCE, BBOX_MASK_SOURCE], dtype=np.int64)
    boxes = [None, np.array([[[1, 1], [0, 0]]], dtype=np.uint8)]

    report = summarize(cams, masks, sources, boxes, k=0.5, tau=0.01)

    assert report.lung.mask_source == LUNG_MASK_SOURCE
    assert report.lung.top_saliency_precision.value == pytest.approx(1.0)
    assert report.lung.top_saliency_precision.num_samples == 1
    assert report.lung.annotation_coverage.value is None
    assert report.lung.annotation_coverage.num_samples == 0

    assert report.bbox.mask_source == BBOX_MASK_SOURCE
    assert report.bbox.top_saliency_precision.value == pytest.approx(0.0)
    assert report.bbox.top_saliency_precision.num_samples == 1
    assert report.bbox.annotation_coverage.value == pytest.approx(0.0)
    assert report.bbox.annotation_coverage.num_samples == 1
    assert report.bbox.annotation_coverage.num_boxes == 1

    payload = report.to_dict()
    assert set(payload) == {"lung", "bbox"}
    assert "overall" not in payload


def test_all_saliency_precision_without_mass_is_unavailable_not_zero():
    cam = np.zeros((2, 2), dtype=np.float64)
    mask = np.ones((2, 2), dtype=np.uint8)

    assert all_saliency_precision(cam, mask) is None
