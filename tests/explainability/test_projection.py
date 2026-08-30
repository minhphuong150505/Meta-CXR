"""Geometry checks for the grid-to-image projection.

Two things are pinned here. First, the synthetic-square test: a grid that is
hot in one named corner must produce an image map that is hot in the SAME
corner. That is the test that catches a transpose, a row/column-major flip or a
vertical mirror -- none of which change any shape, so nothing else would.

Second, the coordinate-frame arithmetic: 14*32 == 7*64 == 448. The repository's
claim that BioViL-T and PubMedCLIP share one coordinate frame is unverified
against the live PubMedCLIP preprocessor config, so it is asserted rather than
commented.

This is the CPU tier. The end-to-end version -- push a real image with a
contrast square through the real encoders -- needs the training host and lives
behind the ``gpu`` marker.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from training.explainability.projection import (  # noqa: E402
    MODE_NEAREST,
    STAGE1_GRIDS,
    STAGE1_IMAGE_SIZE,
    GridSpec,
    SpatialProjectionUnsupported,
    assert_shared_coordinate_frame,
    assert_spatial_projection_supported,
    infer_square_grid,
    normalize_map,
    project_to_image,
    split_global_tokens,
)

CORNERS = {
    # name: (row slice, column slice) of the quadrant that must light up
    "top_left": (slice(0, 224), slice(0, 224)),
    "top_right": (slice(0, 224), slice(224, 448)),
    "bottom_left": (slice(224, 448), slice(0, 224)),
    "bottom_right": (slice(224, 448), slice(224, 448)),
}


def _corner_grid(grid: GridSpec, corner: str) -> torch.Tensor:
    """A grid that is 1.0 in one corner cell and 0.0 everywhere else."""
    cells = torch.zeros(grid.height, grid.width)
    row = 0 if corner.startswith("top") else grid.height - 1
    column = 0 if corner.endswith("left") else grid.width - 1
    cells[row, column] = 1.0
    return cells.reshape(-1)


# --------------------------------------------------------------------------
# The coordinate frame -- 14*32 == 7*64 == 448
# --------------------------------------------------------------------------


def test_declared_stage1_grids_tile_the_448_crop_exactly():
    assert_shared_coordinate_frame(STAGE1_GRIDS, STAGE1_IMAGE_SIZE)
    assert STAGE1_GRIDS["biovil"].covered_px == (448, 448)
    assert STAGE1_GRIDS["pubmedclip"].covered_px == (448, 448)
    # Written out so the arithmetic is visible in the test, not only the code.
    assert 14 * 32 == 448
    assert 7 * 64 == 448


def test_a_grid_that_does_not_tile_the_crop_is_rejected():
    # 7x7 cells of 32 px cover 224, not 448: this is the exact failure mode of
    # an encoder that crops its own image instead of resizing the shared crop.
    broken = {"pubmedclip": GridSpec(height=7, width=7, patch_px=32)}
    with pytest.raises(ValueError, match="do not cover the crop exactly"):
        assert_shared_coordinate_frame(broken, STAGE1_IMAGE_SIZE)


def test_a_grid_that_overruns_the_crop_is_rejected():
    with pytest.raises(ValueError, match="do not cover the crop exactly"):
        assert_shared_coordinate_frame(
            {"biovil": GridSpec(height=16, width=16, patch_px=32)}, STAGE1_IMAGE_SIZE
        )


def test_mixed_grids_must_agree_with_each_other():
    # Individually plausible, jointly incompatible: 448 vs 224.
    with pytest.raises(ValueError, match="pubmedclip"):
        assert_shared_coordinate_frame(
            {
                "biovil": GridSpec(14, 14, 32),
                "pubmedclip": GridSpec(7, 7, 32),
            },
            STAGE1_IMAGE_SIZE,
        )


def test_grid_spec_rejects_non_positive_dimensions():
    for kwargs in ({"height": 0}, {"width": -1}, {"patch_px": 0}):
        base = {"height": 7, "width": 7, "patch_px": 64} | kwargs
        with pytest.raises(ValueError, match="positive integer"):
            GridSpec(**base)


# --------------------------------------------------------------------------
# Synthetic square: the map must light up in the corner it was fed
# --------------------------------------------------------------------------


@pytest.mark.parametrize("corner", sorted(CORNERS))
@pytest.mark.parametrize("grid_name", sorted(STAGE1_GRIDS))
def test_synthetic_square_lights_the_correct_corner(grid_name, corner):
    grid = STAGE1_GRIDS[grid_name]
    projected = project_to_image(_corner_grid(grid, corner), grid, STAGE1_IMAGE_SIZE)
    assert projected.shape == (STAGE1_IMAGE_SIZE, STAGE1_IMAGE_SIZE)

    masses = {
        name: float(projected[rows, columns].sum()) for name, (rows, columns) in CORNERS.items()
    }
    hottest = max(masses, key=masses.__getitem__)
    assert hottest == corner, f"{grid_name}: fed {corner}, map peaked in {hottest} ({masses})"

    # And the peak pixel itself, which a mirror would move even if the
    # quadrant mass happened to survive.
    peak = int(projected.argmax())
    peak_row, peak_column = divmod(peak, STAGE1_IMAGE_SIZE)
    rows, columns = CORNERS[corner]
    assert rows.start <= peak_row < rows.stop
    assert columns.start <= peak_column < columns.stop


@pytest.mark.parametrize("grid_name", sorted(STAGE1_GRIDS))
def test_a_transposed_grid_is_detected_by_the_off_diagonal_corners(grid_name):
    """Row-major vs column-major differ only off the diagonal.

    Feeding top_right and reading top_right proves the reshape order, because a
    transpose sends top_right to bottom_left while leaving top_left and
    bottom_right exactly where they were.
    """
    grid = STAGE1_GRIDS[grid_name]
    projected = project_to_image(_corner_grid(grid, "top_right"), grid, STAGE1_IMAGE_SIZE)
    rows, columns = CORNERS["top_right"]
    mirrored_rows, mirrored_columns = CORNERS["bottom_left"]
    assert float(projected[rows, columns].sum()) > float(
        projected[mirrored_rows, mirrored_columns].sum()
    )


def test_nearest_mode_keeps_each_patch_a_solid_block():
    grid = GridSpec(height=2, width=2, patch_px=224)
    projected = project_to_image(
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
        grid,
        448,
        mode=MODE_NEAREST,
        normalize=False,
    )
    assert torch.allclose(projected[:224, :224], torch.ones(224, 224))
    assert float(projected[224:, :].sum()) == 0.0
    assert float(projected[:, 224:].sum()) == 0.0


def test_a_uniform_grid_projects_to_a_uniform_map():
    grid = STAGE1_GRIDS["pubmedclip"]
    projected = project_to_image(
        torch.full((grid.num_tokens,), 0.3), grid, STAGE1_IMAGE_SIZE, normalize=False
    )
    assert torch.allclose(projected, torch.full_like(projected, 0.3))


def test_batched_projection_matches_the_single_map_path():
    grid = STAGE1_GRIDS["biovil"]
    first = _corner_grid(grid, "top_left")
    second = _corner_grid(grid, "bottom_right")
    batched = project_to_image(torch.stack([first, second]), grid, STAGE1_IMAGE_SIZE)
    assert batched.shape == (2, STAGE1_IMAGE_SIZE, STAGE1_IMAGE_SIZE)
    assert torch.allclose(batched[0], project_to_image(first, grid, STAGE1_IMAGE_SIZE))
    assert torch.allclose(batched[1], project_to_image(second, grid, STAGE1_IMAGE_SIZE))


def test_a_token_count_mismatch_is_refused_rather_than_reshaped():
    grid = STAGE1_GRIDS["pubmedclip"]
    # 50 tokens is the PubMedCLIP stream WITH its CLS still attached -- the
    # exact mistake this guard exists for.
    with pytest.raises(ValueError, match="49 patches"):
        project_to_image(torch.zeros(50), grid, STAGE1_IMAGE_SIZE)


def test_non_finite_attribution_is_refused():
    grid = STAGE1_GRIDS["pubmedclip"]
    values = torch.zeros(grid.num_tokens)
    values[0] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        project_to_image(values, grid, STAGE1_IMAGE_SIZE)


# --------------------------------------------------------------------------
# Global/CLS token handling
# --------------------------------------------------------------------------


def test_split_global_tokens_peels_the_cls_off_the_front():
    stream = torch.arange(50, dtype=torch.float32)
    globals_, spatial = split_global_tokens(stream, 1)
    assert globals_.tolist() == [0.0]
    assert spatial.shape == (49,)
    assert float(spatial[0]) == 1.0


def test_split_global_tokens_with_zero_globals_is_a_no_op():
    stream = torch.arange(196, dtype=torch.float32)
    globals_, spatial = split_global_tokens(stream, 0)
    assert globals_.numel() == 0
    assert torch.equal(spatial, stream)


def test_split_global_tokens_then_project_recovers_the_grid():
    grid = STAGE1_GRIDS["pubmedclip"]
    stream = torch.zeros(1 + grid.num_tokens)
    stream[0] = 99.0  # CLS, must not reach the map
    stream[1] = 1.0  # first spatial patch -> top-left
    _, spatial = split_global_tokens(stream, 1)
    projected = project_to_image(spatial, grid, STAGE1_IMAGE_SIZE)
    rows, columns = CORNERS["top_left"]
    assert float(projected[rows, columns].sum()) > 0.0
    assert float(projected.max()) == pytest.approx(1.0)


def test_split_global_tokens_rejects_an_over_long_split():
    with pytest.raises(ValueError, match="cannot split"):
        split_global_tokens(torch.zeros(3), 4)


# --------------------------------------------------------------------------
# infer_square_grid
# --------------------------------------------------------------------------


def test_infer_square_grid_derives_the_side_length():
    grid = infer_square_grid(256, patch_px=14)
    assert (grid.height, grid.width) == (16, 16)
    assert grid.num_tokens == 256


def test_infer_square_grid_refuses_a_non_square_count():
    # 50 = 49 + CLS. The message must point at the real cause.
    with pytest.raises(ValueError, match="not a perfect square"):
        infer_square_grid(50)


def test_infer_square_grid_rejects_a_non_positive_count():
    with pytest.raises(ValueError, match="positive integer"):
        infer_square_grid(0)


# --------------------------------------------------------------------------
# The Q-Former refusal
# --------------------------------------------------------------------------


def test_projecting_q_former_soft_tokens_is_refused():
    with pytest.raises(SpatialProjectionUnsupported, match="carry no position"):
        assert_spatial_projection_supported("qformer_soft_token")


def test_project_to_image_refuses_a_soft_token_source_before_touching_shapes():
    # Refusal must not depend on the tensor being the right shape, or a caller
    # would get a shape error and "fix" it into a meaningless picture.
    with pytest.raises(SpatialProjectionUnsupported):
        project_to_image(torch.zeros(32), GridSpec(4, 8, 56), 448, source="qformer_soft_token")


def test_the_vision_patch_grid_source_is_permitted():
    assert_spatial_projection_supported("vision_patch_grid")


# --------------------------------------------------------------------------
# normalize_map, pinned against the Stage-1 implementation
# --------------------------------------------------------------------------


def test_normalize_map_matches_stage1_normalize_cam():
    """Do not let two normalisers drift apart.

    ``scripts/evaluate_explanation.py`` already defines this for Stage-1 CAMs.
    This module needs a torch version to stay import-free of numpy scripts, so
    the two are pinned to each other numerically instead.
    """
    from scripts.evaluate_explanation import _normalize_cam

    torch.manual_seed(0)
    batch = torch.randn(4, 7, 7)
    batch[2] = 3.5  # a constant map: both must return zeros, not NaN
    expected = _normalize_cam(batch.numpy())
    actual = normalize_map(batch).numpy()
    assert np.allclose(actual, expected, atol=1e-6)
    assert np.all(actual[2] == 0.0)


def test_normalize_map_sends_a_constant_map_to_zero():
    assert float(normalize_map(torch.full((7, 7), 2.0)).abs().max()) == 0.0


def test_normalize_map_puts_the_extremes_at_zero_and_one():
    values = torch.tensor([[1.0, 2.0], [3.0, 5.0]])
    normalized = normalize_map(values)
    assert float(normalized.min()) == pytest.approx(0.0)
    assert float(normalized.max()) == pytest.approx(1.0)
    assert float(normalized[0, 1]) == pytest.approx(0.25)


# --------------------------------------------------------------------------
# The common frame: original image coordinates
# --------------------------------------------------------------------------
#
# Stage 1 and MedGemma preprocess differently, so neither map means anything to
# the other until both are carried back to the original radiograph. The numbers
# below were taken from real torchvision on the training host 2026-08-30.

from training.explainability.projection import (  # noqa: E402
    MEDGEMMA_GRID,
    MEDGEMMA_IMAGE_SIZE,
    OriginalFrame,
    Stage1CropGeometry,
    medgemma_map_in_original,
    stage1_map_in_original,
)

# (original w, h) -> (resized w, h, crop_left, crop_top), from torchvision itself
TORCHVISION_CASES = [
    ((2544, 3056), (512, 615, 32, 84)),
    ((3056, 2544), (615, 512, 84, 32)),
    ((2022, 2022), (512, 512, 32, 32)),
    ((2500, 3000), (512, 614, 32, 83)),
    ((1935, 2544), (512, 673, 32, 112)),
]


@pytest.mark.parametrize(("original", "expected"), TORCHVISION_CASES)
def test_crop_geometry_matches_real_torchvision(original, expected):
    """Pinned against torchvision 0.24.1 output, not against the docs.

    Resize(int) truncates the long side with int(), CenterCrop offsets with
    int(round(...)). Getting either wrong shifts every map by a few pixels
    with nothing to show for it.
    """
    geometry = Stage1CropGeometry.from_original(OriginalFrame(*original))
    assert (
        geometry.resized_width,
        geometry.resized_height,
        geometry.crop_left,
        geometry.crop_top,
    ) == expected


def test_crop_box_lands_inside_the_original():
    geometry = Stage1CropGeometry.from_original(OriginalFrame(2544, 3056))
    x0, y0, x1, y1 = geometry.crop_box_in_original()
    assert 0 <= x0 < x1 <= 2544
    assert 0 <= y0 < y1 <= 3056


def test_a_square_original_discards_only_the_resize_margin():
    geometry = Stage1CropGeometry.from_original(OriginalFrame(2022, 2022))
    # 448/512 of each side survives, so 1 - (448/512)^2.
    assert geometry.discarded_fraction == pytest.approx(1 - (448 / 512) ** 2, abs=1e-3)


def test_a_tall_original_discards_much_more():
    """Stage 1 never sees a third of a portrait radiograph; MedGemma always does."""
    geometry = Stage1CropGeometry.from_original(OriginalFrame(2544, 3056))
    assert geometry.discarded_fraction == pytest.approx(0.363, abs=0.01)
    assert geometry.discarded_fraction > Stage1CropGeometry.from_original(
        OriginalFrame(2022, 2022)
    ).discarded_fraction


def test_an_image_smaller_than_the_crop_is_refused():
    with pytest.raises(ValueError, match="smaller than CenterCrop"):
        Stage1CropGeometry.from_original(OriginalFrame(100, 120), short_side=200, crop_size=448)


def test_canvas_preserves_the_original_aspect():
    frame = OriginalFrame(2544, 3056)
    height, width = frame.canvas(max_side=512)
    assert height == 512  # portrait -> the long side is capped
    assert width == pytest.approx(512 * 2544 / 3056, abs=1)
    assert width / height == pytest.approx(frame.aspect, abs=0.01)


def test_medgemma_grid_tiles_its_own_896_square():
    assert MEDGEMMA_GRID.covered_px == (MEDGEMMA_IMAGE_SIZE, MEDGEMMA_IMAGE_SIZE)
    assert MEDGEMMA_GRID.num_tokens == 256  # mm_tokens_per_image


def test_medgemma_map_covers_the_whole_canvas():
    """MedGemma squashed the entire image, so nothing is outside its view."""
    frame = OriginalFrame(2544, 3056)
    values = torch.ones(MEDGEMMA_GRID.num_tokens)
    projected = medgemma_map_in_original(values, frame, normalize=False)
    assert projected.shape == frame.canvas()
    assert bool((projected > 0).all())


def test_medgemma_corner_survives_the_anisotropic_undo():
    frame = OriginalFrame(2544, 3056)
    cells = torch.zeros(16, 16)
    cells[0, 15] = 1.0  # top-right
    projected = medgemma_map_in_original(cells.reshape(-1), frame)
    h, w = projected.shape
    assert float(projected[: h // 2, w // 2 :].sum()) > float(
        projected[h // 2 :, : w // 2].sum()
    )


def test_stage1_map_is_zero_where_the_crop_discarded_the_image():
    """0 outside the box means 'Stage 1 could not look here'."""
    frame = OriginalFrame(2544, 3056)
    geometry = Stage1CropGeometry.from_original(frame)
    projected = stage1_map_in_original(torch.ones(448, 448), geometry, normalize=False)
    assert projected.shape == frame.canvas()
    assert float(projected[0, 0]) == 0.0        # top edge is cropped away
    h, w = projected.shape
    assert float(projected[h // 2, w // 2]) == pytest.approx(1.0)
    assert float((projected > 0).float().mean()) == pytest.approx(
        1 - geometry.discarded_fraction, abs=0.02
    )


def test_stage1_and_medgemma_land_on_the_same_canvas():
    """The whole point: two maps that can finally be compared."""
    frame = OriginalFrame(2544, 3056)
    geometry = Stage1CropGeometry.from_original(frame)
    canvas = frame.canvas(max_side=256)
    a = medgemma_map_in_original(torch.rand(256), frame, canvas_hw=canvas)
    b = stage1_map_in_original(torch.rand(448, 448), geometry, canvas_hw=canvas)
    assert a.shape == b.shape == canvas


def test_a_centre_blob_agrees_between_the_two_frames():
    """A finding in the middle must project to the middle in BOTH pipelines.

    The centre is the one region both preprocessings keep, so it is where the
    two frames have to agree. If either transform were inverted, mirrored or
    left un-corrected, these two centroids would separate.
    """
    frame = OriginalFrame(2544, 3056)
    geometry = Stage1CropGeometry.from_original(frame)
    canvas = frame.canvas(max_side=256)

    medgemma_cells = torch.zeros(16, 16)
    medgemma_cells[7:9, 7:9] = 1.0
    stage1_cells = torch.zeros(448, 448)
    stage1_cells[200:248, 200:248] = 1.0

    a = medgemma_map_in_original(medgemma_cells.reshape(-1), frame, canvas_hw=canvas)
    b = stage1_map_in_original(stage1_cells, geometry, canvas_hw=canvas)

    def centroid(m):
        total = m.sum()
        rows = torch.arange(m.shape[0], dtype=torch.float32)[:, None]
        cols = torch.arange(m.shape[1], dtype=torch.float32)[None, :]
        return float((m * rows).sum() / total), float((m * cols).sum() / total)

    ar, ac = centroid(a)
    br, bc = centroid(b)
    assert abs(ar - br) < 0.05 * canvas[0]
    assert abs(ac - bc) < 0.05 * canvas[1]


def test_outside_value_is_configurable_and_distinguishable():
    frame = OriginalFrame(2544, 3056)
    geometry = Stage1CropGeometry.from_original(frame)
    projected = stage1_map_in_original(
        torch.ones(448, 448), geometry, normalize=False, outside_value=float("nan")
    )
    assert torch.isnan(projected[0, 0])
    assert not torch.isnan(projected[projected.shape[0] // 2, projected.shape[1] // 2])
