"""CPU-only tests for the two-tier explanation-mask data path."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from preporcessing import build_explanation_masks as mask_builder


def _encode_rle(mask: np.ndarray) -> str:
    """Synthetic inverse of CheXmask's one-based, row-major RLE decoder."""

    flat = (np.asarray(mask) > 0).astype(np.uint8).ravel(order="C")
    padded = np.concatenate(([0], flat, [0]))
    runs = np.flatnonzero(padded[1:] != padded[:-1]) + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(int(value)) for value in runs)


def _nonzero_bbox(array: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(array)
    assert len(xs) > 0
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def test_decode_rle_round_trip_matches_known_mask():
    expected = np.zeros((7, 11), dtype=np.uint8)
    expected[1:3, 2:8] = 1
    expected[5, [0, 4, 10]] = 1

    decoded = mask_builder.decode_rle(
        _encode_rle(expected), height=expected.shape[0], width=expected.shape[1]
    )

    assert decoded.dtype == np.uint8
    np.testing.assert_array_equal(decoded, expected)


def test_decode_lung_union_combines_both_lungs():
    left = np.zeros((6, 8), dtype=np.uint8)
    right = np.zeros_like(left)
    left[1:5, 1:3] = 1
    right[2:5, 5:7] = 1

    union = mask_builder.decode_lung_union(
        _encode_rle(left), _encode_rle(right), *left.shape
    )

    np.testing.assert_array_equal(union, np.logical_or(left, right).astype(np.uint8))


@pytest.fixture(scope="module")
def synthetic_cache(tmp_path_factory):
    root = tmp_path_factory.mktemp("explanation-mask-cache")
    ids = {
        "lung": "synthetic-lung-mask",
        "bbox": "synthetic-bbox-mask",
        "low_dice": "synthetic-low-dice-mask",
        "val": "synthetic-validation-only",
        "test": "synthetic-test-only",
    }

    manifest_paths = {}
    split_rows = {
        "train": [
            (1, 11, ids["lung"]),
            (2, 22, ids["bbox"]),
            (3, 33, ids["low_dice"]),
        ],
        "val": [(4, 44, ids["val"])],
        "test": [(5, 55, ids["test"])],
    }
    for split, rows in split_rows.items():
        path = root / f"project_{split}.csv"
        pd.DataFrame(
            rows,
            columns=["subject_id", "study_id", "dicom_id"],
        ).assign(ViewPosition="PA").to_csv(path, index=False)
        manifest_paths[split] = path

    height, width = 8, 12
    left = np.zeros((height, width), dtype=np.uint8)
    right = np.zeros_like(left)
    left[1:7, 1:4] = 1
    right[1:7, 8:11] = 1
    chexmask_path = root / "chexmask.csv"
    pd.DataFrame(
        [
            {
                "dicom_id": ids["lung"],
                "Dice RCA (Mean)": 0.91,
                "Left Lung": _encode_rle(left),
                "Right Lung": _encode_rle(right),
                "Height": height,
                "Width": width,
            },
            {
                "dicom_id": ids["bbox"],
                "Dice RCA (Mean)": 0.95,
                "Left Lung": _encode_rle(left),
                "Right Lung": _encode_rle(right),
                "Height": height,
                "Width": width,
            },
            {
                "dicom_id": ids["low_dice"],
                "Dice RCA (Mean)": 0.69,
                "Left Lung": _encode_rle(left),
                "Right Lung": _encode_rle(right),
                "Height": height,
                "Width": width,
            },
        ]
    ).to_csv(chexmask_path, index=False)

    ms_cxr_path = root / "ms_cxr.csv"
    bbox_row = {
        "dicom_id": ids["bbox"],
        "x": 5,
        "y": 2,
        "w": 3,
        "h": 3,
        "image_width": width,
        "image_height": height,
        # Deliberately differs from the project manifest.  The builder must
        # count this but still assign the box to the project's train split.
        "split": "test",
    }
    pd.DataFrame([bbox_row]).to_csv(ms_cxr_path, index=False)

    output_dir = root / "cache"
    stats = mask_builder.build_mask_caches(
        manifest_paths=manifest_paths,
        chexmask_csv=chexmask_path,
        ms_cxr_csv=ms_cxr_path,
        output_dir=output_dir,
        split="train",
        dice_threshold=0.7,
        chunk_size=2,
    )
    masks = np.load(output_dir / "masks_train.npy", mmap_mode="r")
    with (output_dir / "index_train.json").open(encoding="utf-8") as handle:
        index = json.load(handle)
    return {
        "ids": ids,
        "masks": masks,
        "index": index,
        "stats": stats,
        "bbox_row": bbox_row,
        "left": left,
        "right": right,
        "output_dir": output_dir,
    }


def test_ms_cxr_bbox_overrides_lung_mask_and_sets_source_one(synthetic_cache):
    ids = synthetic_cache["ids"]
    index = synthetic_cache["index"]
    masks = synthetic_cache["masks"]
    bbox_entry = index[ids["bbox"]]

    bbox_row = synthetic_cache["bbox_row"]
    expected_bbox_source = np.zeros(
        (bbox_row["image_height"], bbox_row["image_width"]), dtype=np.uint8
    )
    expected_bbox_source[2:5, 5:8] = 1
    expected_bbox = mask_builder.transform_mask_geometry(
        expected_bbox_source
    )
    lung_union = np.logical_or(
        synthetic_cache["left"], synthetic_cache["right"]
    ).astype(np.uint8)
    expected_lung = mask_builder.transform_mask_geometry(lung_union)

    assert bbox_entry["mask_source"] == 1
    np.testing.assert_array_equal(masks[bbox_entry["row"]], expected_bbox)
    assert not np.array_equal(masks[bbox_entry["row"]], expected_lung)


def test_low_dice_lung_mask_is_invalid_and_not_indexed(
    synthetic_cache, report_dataset_module
):
    ids = synthetic_cache["ids"]

    assert ids["low_dice"] not in synthetic_cache["index"]
    assert synthetic_cache["stats"]["train"]["no_mask"] == 1
    assert synthetic_cache["stats"]["train"]["lung_masks"] == 2

    module = report_dataset_module
    dataset = module.MIMIC_CXR_Dataset.__new__(module.MIMIC_CXR_Dataset)
    dataset.cur_split = "train"
    dataset.resize_size = 512
    dataset.img_size = 448
    dataset.explanation_mask_size = (112, 112)
    cfg = SimpleNamespace(
        model_cfg={
            "explanation": {"mask_cache_dir": str(synthetic_cache["output_dir"])}
        }
    )
    dataset._init_explanation_mask_cache(cfg)
    mask, valid, source = dataset._read_explanation_mask(ids["low_dice"])

    assert valid is False
    assert source == 0
    assert not mask.any()


def test_image_and_mask_receive_one_identical_affine_sample(report_dataset_module):
    module = report_dataset_module
    dataset = module.MIMIC_CXR_Dataset.__new__(module.MIMIC_CXR_Dataset)
    dataset.base_geometry_trans = module.Compose([])
    dataset.optical_trans = module.Compose([])
    dataset.augmentation_enabled = True
    dataset.affine_p = 1.0
    dataset.affine_degrees = (-5.0, 5.0)
    dataset.affine_translate = (0.1, 0.1)
    dataset.affine_scale = (0.95, 1.05)
    dataset.explanation_mask_size = (112, 112)

    affine_stub = module._test_random_affine
    affine_stub.fixed_params = (0.0, (40, 20), 1.0, (0.0, 0.0))
    affine_stub.get_params_calls = 0

    image_array = np.zeros((448, 448), dtype=np.uint8)
    image_array[100:132, 120:152] = 255
    mask_array = np.zeros((112, 112), dtype=np.uint8)
    mask_array[25:33, 30:38] = 255

    transformed_image, transformed_mask = dataset._apply_synced_image_mask_transforms(
        Image.fromarray(image_array), Image.fromarray(mask_array)
    )
    image_at_mask_resolution = transformed_image.resize(
        (112, 112), resample=Image.Resampling.NEAREST
    )

    assert affine_stub.get_params_calls == 1
    np.testing.assert_array_equal(
        np.asarray(image_at_mask_resolution) > 0,
        transformed_mask.numpy() > 0,
    )


def test_resize_512_center_crop_448_mask_geometry_matches_image_pipeline(
    report_dataset_module,
):
    source = np.zeros((300, 600), dtype=np.uint8)
    source[105:195, 250:350] = 255
    image = Image.fromarray(source)

    image_pipeline = report_dataset_module.Compose(
        [report_dataset_module.Resize(512), report_dataset_module.CenterCrop(448)]
    )
    transformed_image = image_pipeline(image)
    transformed_mask = mask_builder.apply_resize_center_crop(
        image,
        resize_size=512,
        crop_size=448,
        resample=Image.Resampling.NEAREST,
    )

    assert transformed_image.size == transformed_mask.size == (448, 448)
    image_bbox = _nonzero_bbox(np.asarray(transformed_image) >= 128)
    mask_bbox = _nonzero_bbox(np.asarray(transformed_mask) > 0)
    np.testing.assert_allclose(image_bbox, mask_bbox, atol=1)


def test_no_cache_configuration_keeps_new_keys_out_of_getitem(report_dataset_module):
    module = report_dataset_module
    dataset = module.MIMIC_CXR_Dataset.__new__(module.MIMIC_CXR_Dataset)
    label_columns = [f"synthetic_label_{index}" for index in range(14)]
    row = {
        "findings": "synthetic report target",
        "classification_valid": True,
        "target_valid": True,
        "dicom_id": "synthetic-image",
        **{column: 0 for column in label_columns},
    }
    dataset.studies = [{"anchor": 0, "aux": []}]
    dataset.annotation = pd.DataFrame([row])
    dataset.chexpert_cols = label_columns
    dataset.img_ids = {"synthetic-image": 0}
    dataset.multi_view = False
    dataset.explanation_mask_cache_dir = None
    dataset._row_visual = lambda _ann: {
        "image_path": "synthetic.jpg",
        "image": torch.zeros(3, 448, 448),
    }

    sample = dataset.__getitem__(0)

    assert {
        "explanation_mask",
        "explanation_mask_valid",
        "explanation_mask_source",
    }.isdisjoint(sample)


def test_cache_configuration_emits_expected_shapes_and_dtypes(report_dataset_module):
    module = report_dataset_module
    dataset = module.MIMIC_CXR_Dataset.__new__(module.MIMIC_CXR_Dataset)
    label_columns = [f"synthetic_label_{index}" for index in range(14)]
    row = {
        "findings": "synthetic report target",
        "classification_valid": True,
        "target_valid": True,
        "dicom_id": "synthetic-image",
        **{column: 0 for column in label_columns},
    }
    expected_mask = torch.zeros(112, 112, dtype=torch.float32)
    expected_mask[20:40, 30:50] = 1.0
    dataset.studies = [{"anchor": 0, "aux": []}]
    dataset.annotation = pd.DataFrame([row])
    dataset.chexpert_cols = label_columns
    dataset.img_ids = {"synthetic-image": 0}
    dataset.multi_view = False
    dataset.feature_cache = None
    dataset.explanation_mask_cache_dir = Path("configured")
    dataset._read_explanation_mask = lambda _identifier: (
        (expected_mask.numpy() * 255).astype(np.uint8),
        True,
        1,
    )
    dataset._row_visual = lambda _ann, explanation_mask=None: {
        "image_path": "synthetic.jpg",
        "image": torch.zeros(3, 448, 448),
        "explanation_mask": torch.from_numpy(
            (np.asarray(explanation_mask) > 0).astype(np.float32)
        ),
    }

    sample = dataset.__getitem__(0)

    assert sample["explanation_mask"].shape == (112, 112)
    assert sample["explanation_mask"].dtype == torch.float32
    assert set(torch.unique(sample["explanation_mask"]).tolist()) == {0.0, 1.0}
    assert sample["explanation_mask_valid"].dtype == torch.bool
    assert sample["explanation_mask_valid"].item() is True
    assert sample["explanation_mask_source"] == 1


def test_mask_memmap_is_opened_lazily_after_index_load(
    report_dataset_module, tmp_path, monkeypatch
):
    module = report_dataset_module
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    array = np.zeros((1, 112, 112), dtype=np.uint8)
    array[0, 20:40, 30:50] = 255
    np.save(cache_dir / "masks_train.npy", array)
    with (cache_dir / "index_train.json").open("w", encoding="utf-8") as handle:
        json.dump({"synthetic-image": {"row": 0, "mask_source": 0}}, handle)

    dataset = module.MIMIC_CXR_Dataset.__new__(module.MIMIC_CXR_Dataset)
    dataset.cur_split = "train"
    dataset.resize_size = 512
    dataset.img_size = 448
    dataset.explanation_mask_size = (112, 112)
    real_load = module.np.load
    load_calls = []

    def tracking_load(*args, **kwargs):
        load_calls.append((args, kwargs))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(module.np, "load", tracking_load)
    cfg = SimpleNamespace(
        model_cfg={"explanation": {"mask_cache_dir": str(cache_dir)}}
    )

    dataset._init_explanation_mask_cache(cfg)
    assert load_calls == []
    loaded, valid, source = dataset._read_explanation_mask("synthetic-image")

    assert len(load_calls) == 1
    assert valid is True
    assert source == 0
    assert loaded.shape == (112, 112)
    assert set(np.unique(loaded)) == {0, 255}


def test_missing_chexmask_column_fails_naming_the_column(tmp_path):
    """The real header is dicom_id, not the Image ID the data dictionary lists.

    Verified against the real file on 2026-08-14: the MIMIC CheXmask export ships
    dicom_id,Dice RCA (Mean),Dice RCA (Max),Landmarks,Left Lung,Right Lung,Heart,
    Height,Width.  A schema drift must name the missing column and must not echo
    any identifier value while doing it.
    """
    csv_path = Path(tmp_path) / "wrong_schema.csv"
    pd.DataFrame({"Image ID": ["some-identifier-shaped-value"]}).to_csv(
        csv_path, index=False
    )

    with pytest.raises(ValueError) as error:
        mask_builder._validate_columns(csv_path, mask_builder.CHEXMASK_COLUMNS, "CheXmask")

    message = str(error.value)
    assert "dicom_id" in message
    assert "some-identifier-shaped-value" not in message


def test_inspect_prints_schema_and_dice_range_without_identifiers(tmp_path, capsys):
    private_identifier = "feedface" * 5
    csv_path = Path(tmp_path) / "inspect.csv"
    pd.DataFrame(
        {
            "dicom_id": [private_identifier],
            "Dice RCA (Mean)": [0.812],
            "Left Lung": ["1 1"],
            "Right Lung": ["3 1"],
            "Height": [4],
            "Width": [4],
        }
    ).to_csv(csv_path, index=False)

    mask_builder.inspect_chexmask(csv_path, sample_rows=32)
    output = capsys.readouterr().out

    assert private_identifier not in output
    assert "dicom_id" in output
    assert "0.812" in output


def test_ms_cxr_rows_absent_from_the_manifest_are_dropped_not_fatal(tmp_path, capsys):
    """A box we cannot place in a split is dropped, never guessed into one.

    An image missing from every manifest is never trained on, so dropping its box
    cannot leak.  Aborting the whole 220k-study build over it would not buy any
    safety, so the count is reported instead.
    """
    known, unknown = "known-synthetic-id", "absent-synthetic-id"
    csv_path = Path(tmp_path) / "ms_cxr.csv"
    pd.DataFrame(
        [
            {"dicom_id": known, "x": 1, "y": 1, "w": 2, "h": 2,
             "image_width": 8, "image_height": 8, "split": "train"},
            {"dicom_id": unknown, "x": 1, "y": 1, "w": 2, "h": 2,
             "image_width": 8, "image_height": 8, "split": "train"},
        ]
    ).to_csv(csv_path, index=False)

    groups, mismatch, kept = mask_builder._load_ms_cxr(
        csv_path, {known: "train"}, {known, unknown}
    )

    assert set(groups) == {known}
    assert kept == 1
    assert mismatch == 0
    output = capsys.readouterr().out
    assert "1 / 2" in output
    assert unknown not in output


def test_bbox_cropped_out_of_frame_falls_back_to_lung(monkeypatch, tmp_path):
    """One unusable box must not abort a 228k-study build.

    Resize(512) crops the long axis, so a box against the top or bottom edge can
    disappear. Measured on MS-CXR v1.1.0: 3 of 1,448 boxes vanish and exactly one
    DICOM loses every box it has. That study should keep its lung mask.
    """
    tall = np.zeros((3000, 1500), dtype=np.uint8)
    tall[0:40, 700:800] = 1  # hugs the top edge, outside the centre crop

    assert not mask_builder.transform_mask_geometry(tall).any()

    centred = np.zeros((3000, 1500), dtype=np.uint8)
    centred[1400:1600, 700:800] = 1

    assert mask_builder.transform_mask_geometry(centred).any()
