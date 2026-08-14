"""The uint8 fast path in ``remap_to_uint8`` must be bit-identical.

It replaced the float64 min-max stretch that dominated the training data
pipeline (37.9 ms of an 83.6 ms study, and bandwidth-bound enough that twelve
workers starved each other). A preprocessing change here would be silent: the
model would simply train on slightly different pixels, and nothing would fail.
So every case is pinned against the original implementation, which is
reproduced here rather than imported so the test still means something if the
optimised path is edited.
"""

import numpy as np
import pytest

try:
    from model.lavis.data.ReportDataset import MIMIC_CXR_Dataset
except Exception as exc:  # torchvision, or a private configs/env_config.yaml
    pytest.skip(
        f"ReportDataset is not importable here ({type(exc).__name__}); "
        "run this on the training host",
        allow_module_level=True,
    )

remap = MIMIC_CXR_Dataset.remap_to_uint8


def reference(array, percentiles=None):
    """The float64 implementation, verbatim, before the fast path."""
    array = array.astype(float)
    if percentiles is not None:
        cutoff = np.percentile(array, percentiles)
        array = np.clip(array, *cutoff)
    array -= array.min()
    value_range = array.max()
    if value_range == 0:
        return np.zeros_like(array, dtype=np.uint8)
    array /= value_range
    array *= 255
    return array.astype(np.uint8)


@pytest.mark.parametrize(
    "low,high",
    [(0, 255), (0, 200), (37, 255), (10, 240), (0, 1), (128, 129), (7, 7)],
)
def test_uint8_fast_path_matches_the_float64_reference(low, high):
    rng = np.random.default_rng(0)
    array = rng.integers(low, high + 1, size=(64, 48), dtype=np.uint8)
    # Pin the extremes so the intended range is actually present.
    array[0, 0], array[-1, -1] = low, high

    assert np.array_equal(remap(None, array), reference(array))


def test_full_range_input_is_returned_unchanged():
    """The identity shortcut: this is what every MIMIC-CXR-JPG image hits."""
    array = np.arange(256, dtype=np.uint8).reshape(16, 16)

    out = remap(None, array)

    assert np.array_equal(out, array)
    assert np.array_equal(out, reference(array))


def test_constant_image_maps_to_zeros():
    array = np.full((8, 8), 91, dtype=np.uint8)

    assert np.array_equal(remap(None, array), np.zeros((8, 8), dtype=np.uint8))


def test_output_dtype_is_uint8_on_every_path():
    rng = np.random.default_rng(1)
    for array in (
        rng.integers(0, 256, size=(32, 32), dtype=np.uint8),
        rng.integers(3, 199, size=(32, 32), dtype=np.uint8),
        np.full((4, 4), 5, dtype=np.uint8),
    ):
        assert remap(None, array).dtype == np.uint8


def test_non_uint8_input_still_uses_the_float_path():
    """16-bit DICOM is what the original was written for; do not break it."""
    array = (np.arange(64, dtype=np.uint16) * 500).reshape(8, 8)

    assert np.array_equal(remap(None, array), reference(array))


def test_percentiles_bypass_the_fast_path():
    rng = np.random.default_rng(2)
    array = rng.integers(0, 256, size=(40, 40), dtype=np.uint8)

    assert np.array_equal(
        remap(None, array, percentiles=(5, 95)),
        reference(array, percentiles=(5, 95)),
    )
