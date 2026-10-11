"""BioViL's channel-first grid: the historical reshape vs the true patch layout."""

import pytest
import torch

from vision_encoders.biovil_tokens import (
    LEGACY_RESHAPE,
    PATCH,
    biovil_grid_to_tokens,
    resolve_biovil_token_layout,
)


def _grid(b=2, d=1408, h=14, w=14):
    return torch.arange(b * d * h * w, dtype=torch.float32).reshape(b, d, h, w)


def test_absent_key_is_the_historical_layout():
    assert resolve_biovil_token_layout(None) == LEGACY_RESHAPE
    assert resolve_biovil_token_layout("patch") == PATCH
    with pytest.raises(ValueError):
        resolve_biovil_token_layout("spatial")


def test_legacy_layout_is_bit_identical_to_the_old_reshape():
    grid = _grid()
    old = grid.reshape(grid.shape[0], -1, 1408)
    assert torch.equal(biovil_grid_to_tokens(grid, LEGACY_RESHAPE), old)


def test_patch_layout_puts_one_patch_in_one_token_row_major():
    grid = _grid()
    tokens = biovil_grid_to_tokens(grid, PATCH)
    assert tokens.shape == (2, 196, 1408)
    for row, col in ((0, 0), (3, 5), (13, 13)):
        assert torch.equal(tokens[1, row * 14 + col], grid[1, :, row, col])
    # the mean over tokens is BioViL's own global embedding
    assert torch.allclose(tokens.mean(1), grid.mean((2, 3)))


def test_legacy_layout_is_not_a_patch_layout():
    grid = _grid()
    legacy = biovil_grid_to_tokens(grid, LEGACY_RESHAPE)
    assert not torch.equal(legacy, biovil_grid_to_tokens(grid, PATCH))
    # legacy token 0 is the first 1408 values of the [D, H, W] block:
    # seven whole channel maps and part of an eighth, not one patch
    assert torch.equal(legacy[0, 0], grid[0].reshape(-1)[:1408])


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        biovil_grid_to_tokens(torch.zeros(2, 196, 1408), PATCH)
    with pytest.raises(ValueError):
        biovil_grid_to_tokens(_grid(), "other")
