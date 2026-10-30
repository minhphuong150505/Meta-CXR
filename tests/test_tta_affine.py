"""Pins ``model/lavis/data/tta.py``: the opt-in test-time affine (2026-10-30)."""

import math

import numpy as np
import pytest
from PIL import Image

from model.lavis.data.tta import TTAAffine, apply_tta_affine, resolve_tta_affine


def _image(w=101, h=121):
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 256, (h, w), dtype=np.uint8), mode="L")


def test_absent_key_means_no_transform():
    assert resolve_tta_affine(None, "val") is None
    assert resolve_tta_affine(None, "train") is None


def test_train_split_is_refused():
    with pytest.raises(ValueError, match="evaluation-only"):
        resolve_tta_affine({"degrees": 5.0}, "train")


@pytest.mark.parametrize("raw", [{"degree": 5.0}, {"scale": 0.0}, {"scale": -1.0},
                                 {"degrees": math.nan}, [5.0, 1.0]])
def test_bad_configs_are_refused(raw):
    with pytest.raises(ValueError):
        resolve_tta_affine(raw, "val")


def test_defaults():
    assert resolve_tta_affine({"degrees": 5}, "test") == TTAAffine(5.0, 1.0)
    assert resolve_tta_affine({"scale": 1.05}, "val") == TTAAffine(0.0, 1.05)


def test_identity_parameters_return_the_image():
    img = _image()
    out = apply_tta_affine(img, TTAAffine(0.0, 1.0))
    assert out.size == img.size and out.mode == "L"
    assert np.array_equal(np.asarray(out), np.asarray(img))


def test_rotation_matches_pil_rotate():
    img = _image(100, 100)
    for deg in (5.0, -5.0, 90.0):
        ours = np.asarray(apply_tta_affine(img, TTAAffine(deg, 1.0)), dtype=np.int16)
        ref = np.asarray(img.rotate(deg, resample=Image.BILINEAR, fillcolor=0), dtype=np.int16)
        assert np.abs(ours - ref).max() <= 1


def test_scale_moves_points_away_from_the_centre():
    w = h = 101
    arr = np.zeros((h, w), dtype=np.uint8)
    arr[50, 60] = 255  # 10 px right of the centre pixel
    out = np.asarray(apply_tta_affine(Image.fromarray(arr, mode="L"), TTAAffine(0.0, 2.0)))
    y, x = np.unravel_index(out.argmax(), out.shape)
    assert y == 50 and abs(x - 70) <= 1  # 20 px right after a 2x zoom
    small = np.asarray(apply_tta_affine(Image.fromarray(arr, mode="L"), TTAAffine(0.0, 0.5)))
    y, x = np.unravel_index(small.argmax(), small.shape)
    assert y == 50 and abs(x - 55) <= 1


def test_zoom_out_fills_borders_with_zero():
    img = Image.new("L", (64, 64), 200)
    out = np.asarray(apply_tta_affine(img, TTAAffine(0.0, 0.9)))
    assert out[0, 0] == 0 and out[32, 32] == 200
