"""Build privacy-sensitive explanation-mask caches for Stage-1 training.

CheXmask lung RLEs are decoded with the convention used by the dataset's
official utility: NumPy C-order flattening and one-based run starts.  Left and
right lungs are unioned; the heart is intentionally excluded.  MS-CXR boxes
override lung masks for the same DICOM.  Every selected mask is transformed at
its original image size through ``Resize(512) -> CenterCrop(448)`` and stored as
binary uint8 values ``{0, 255}`` at 112 x 112.

The generated ``.npy``/JSON files are derivatives of MIMIC-CXR and must remain
on private storage.  This script never logs identifiers, image paths, or report
text.  The 13 GB CheXmask CSV is always consumed with ``pandas.read_csv``
chunks; it is never loaded as one DataFrame.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

CHEXMASK_COLUMNS = (
    "dicom_id",
    "Dice RCA (Mean)",
    "Left Lung",
    "Right Lung",
    "Height",
    "Width",
)
MS_CXR_COLUMNS = (
    "dicom_id",
    "category_name",
    "x",
    "y",
    "w",
    "h",
    "image_width",
    "image_height",
    "split",
)
PROJECT_SPLITS = ("train", "val", "test")
# Must stay identical to `chexpert_cols` in blip2_qformer.py: the per-pathology
# cache is keyed by POSITION in this tuple, so a reordering here silently
# supervises the wrong finding.
CHEXPERT_LABELS = (
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
    "Support Devices",
)
_LABEL_INDEX = {name.lower(): index for index, name in enumerate(CHEXPERT_LABELS)}


def category_to_label_index(category_name: object) -> int | None:
    """Map an MS-CXR category to a CheXpert column, or None if it has none.

    MS-CXR v1.1.0 uses eight categories, all of which are CheXpert findings.
    Anything unmapped is skipped rather than guessed: a box attached to the
    wrong column would teach the model to look at the wrong place, which is
    worse than no supervision at all.
    """
    if category_name is None:
        return None
    key = str(category_name).strip().lower()
    return _LABEL_INDEX.get(key)
MASK_SIZE = (112, 112)
DEFAULT_CHEXMASK_CSV = Path(
    "/mnt/drive1tb/datasets/chexmask/MIMIC-CXR-JPG.csv"
)
DEFAULT_MS_CXR_CSV = Path(
    "/mnt/drive1tb/datasets/ms-cxr/MS_CXR_Local_Alignment_v1.1.0.csv"
)
DEFAULT_OUTPUT_DIR = Path("/mnt/drive1tb/datasets/explanation_masks")
DEFAULT_CHUNK_SIZE = 256

_VIEW_IDS = {"PA": 0, "AP": 1, "LATERAL": 2, "LL": 2}
_VIEW_PRIORITY = {0: 0, 1: 1, 2: 2}


def _missing(value: object) -> bool:
    return value is None or bool(pd.isna(value)) or not str(value).strip()


def _positive_int(value: object, field_name: str) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must contain finite positive integers") from exc
    if not math.isfinite(numeric) or numeric <= 0 or not numeric.is_integer():
        raise ValueError(f"{field_name} must contain finite positive integers")
    return int(numeric)


def decode_rle(rle: object, height: int, width: int) -> np.ndarray:
    """Decode a CheXmask RLE into a ``uint8`` mask with values ``{0, 1}``.

    CheXmask uses pairs ``(one_based_start, run_length)`` over a row-major
    (NumPy C-order) flattened mask.  Empty RLE values decode to an all-zero mask.
    Malformed or out-of-bounds runs fail without echoing the RLE content.
    """

    height = _positive_int(height, "Height")
    width = _positive_int(width, "Width")
    flat = np.zeros(height * width, dtype=np.uint8)
    if _missing(rle):
        return flat.reshape(height, width)

    try:
        runs = np.asarray(str(rle).split(), dtype=np.int64)
    except (TypeError, ValueError) as exc:
        raise ValueError("CheXmask RLE contains a non-integer token") from exc
    if runs.size % 2:
        raise ValueError("CheXmask RLE must contain start/length pairs")

    starts = runs[::2] - 1
    lengths = runs[1::2]
    ends = starts + lengths
    if (
        np.any(starts < 0)
        or np.any(lengths <= 0)
        or np.any(ends > flat.size)
    ):
        raise ValueError("CheXmask RLE contains a non-positive or out-of-bounds run")

    for start, end in zip(starts, ends, strict=True):
        flat[int(start) : int(end)] = 1
    return flat.reshape(height, width)


def decode_lung_union(
    left_lung_rle: object,
    right_lung_rle: object,
    height: int,
    width: int,
) -> np.ndarray:
    """Decode and union the left/right lungs; the heart is not used."""

    left = decode_rle(left_lung_rle, height, width)
    right = decode_rle(right_lung_rle, height, width)
    return np.logical_or(left, right).astype(np.uint8)


def resize_shorter_side(
    image: Image.Image,
    size: int,
    *,
    resample: Image.Resampling,
) -> Image.Image:
    """Reproduce torchvision ``Resize(int)`` using Pillow.

    The shorter side becomes ``size`` and the longer side is truncated to an
    integer after aspect-ratio scaling, matching torchvision's PIL path.
    """

    size = _positive_int(size, "resize size")
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if width <= height:
        output_size = (size, int(size * height / width))
    else:
        output_size = (int(size * width / height), size)
    return image.resize(output_size, resample=resample)


def center_crop(image: Image.Image, size: int) -> Image.Image:
    """Reproduce torchvision ``CenterCrop(int)`` for an image large enough."""

    size = _positive_int(size, "crop size")
    width, height = image.size
    if width < size or height < size:
        raise ValueError("center crop is larger than the resized image")
    left = int(round((width - size) / 2.0))
    top = int(round((height - size) / 2.0))
    return image.crop((left, top, left + size, top + size))


def apply_resize_center_crop(
    image: Image.Image,
    *,
    resize_size: int = 512,
    crop_size: int = 448,
    resample: Image.Resampling = Image.Resampling.NEAREST,
) -> Image.Image:
    """Apply the deterministic geometry shared by image and mask pipelines."""

    return center_crop(
        resize_shorter_side(image, resize_size, resample=resample), crop_size
    )


def transform_mask_geometry(
    mask: np.ndarray,
    *,
    resize_size: int = 512,
    crop_size: int = 448,
    output_size: tuple[int, int] = MASK_SIZE,
) -> np.ndarray:
    """Transform a source-size binary mask and return 112 x 112 ``uint8`` 0/255."""

    if mask.ndim != 2:
        raise ValueError("mask must have shape [height, width]")
    binary = (np.asarray(mask) > 0).astype(np.uint8) * 255
    image = Image.fromarray(binary)
    image = apply_resize_center_crop(
        image,
        resize_size=resize_size,
        crop_size=crop_size,
        resample=Image.Resampling.NEAREST,
    )
    image = image.resize(output_size, resample=Image.Resampling.NEAREST)
    return (np.asarray(image, dtype=np.uint8) > 0).astype(np.uint8) * 255


def rasterize_bbox_union(rows: pd.DataFrame) -> np.ndarray:
    """Rasterize the union of an MS-CXR DICOM's pixel-space bounding boxes."""

    if rows.empty:
        raise ValueError("cannot rasterize an empty MS-CXR bbox group")
    missing = sorted(set(MS_CXR_COLUMNS) - set(rows.columns))
    if missing:
        raise ValueError(f"MS-CXR CSV is missing columns: {missing}")

    widths = {_positive_int(value, "image_width") for value in rows["image_width"]}
    heights = {
        _positive_int(value, "image_height") for value in rows["image_height"]
    }
    if len(widths) != 1 or len(heights) != 1:
        raise ValueError("MS-CXR boxes for one image disagree on image dimensions")
    width = widths.pop()
    height = heights.pop()
    mask = np.zeros((height, width), dtype=np.uint8)

    for row in rows.itertuples(index=False):
        try:
            x = float(row.x)
            y = float(row.y)
            box_width = float(row.w)
            box_height = float(row.h)
        except (TypeError, ValueError) as exc:
            raise ValueError("MS-CXR bbox coordinates must be numeric") from exc
        if (
            not all(math.isfinite(value) for value in (x, y, box_width, box_height))
            or box_width <= 0
            or box_height <= 0
        ):
            raise ValueError("MS-CXR bbox coordinates must be finite with positive size")

        x0 = max(0, min(width, math.floor(x)))
        y0 = max(0, min(height, math.floor(y)))
        x1 = max(0, min(width, math.ceil(x + box_width)))
        y1 = max(0, min(height, math.ceil(y + box_height)))
        if x1 <= x0 or y1 <= y0:
            raise ValueError("MS-CXR bbox does not overlap its declared image dimensions")
        mask[y0:y1, x0:x1] = 1
    return mask


def choose_preferred_mask(
    lung_mask: np.ndarray | None,
    bbox_mask: np.ndarray | None,
) -> tuple[np.ndarray | None, int | None]:
    """Return bbox (source 1) when present, otherwise lung (source 0)."""

    if bbox_mask is not None:
        return bbox_mask, 1
    if lung_mask is not None:
        return lung_mask, 0
    return None, None


def inspect_chexmask(csv_path: str | Path, sample_rows: int = 48) -> None:
    """Print schema and Dice-score range from a small CheXmask prefix."""

    csv_path = Path(csv_path)
    header = pd.read_csv(csv_path, nrows=0)
    print("CheXmask columns: " + ", ".join(str(column) for column in header.columns))
    _validate_columns(csv_path, CHEXMASK_COLUMNS, "CheXmask")
    sample = pd.read_csv(
        csv_path,
        nrows=sample_rows,
        usecols=["Dice RCA (Mean)"],
    )
    print(f"CheXmask rows inspected: {len(sample)}")
    dice_values = pd.to_numeric(sample["Dice RCA (Mean)"], errors="coerce")
    dice_values = dice_values[np.isfinite(dice_values)]
    if dice_values.empty:
        print("Dice RCA (Mean) range: unavailable")
    else:
        print(
            "Dice RCA (Mean) range: "
            f"{float(dice_values.min()):.6g} to {float(dice_values.max()):.6g}"
        )


def _normalise_project_split(value: object) -> str:
    text = str(value).strip().lower()
    aliases = {"train": "train", "val": "val", "validate": "val", "test": "test"}
    if text not in aliases:
        raise ValueError("MS-CXR split contains a value outside train/val/test")
    return aliases[text]


def _view_id(value: object) -> int:
    if _missing(value):
        return 3
    return _VIEW_IDS.get(str(value).strip().upper(), 3)


def _read_project_manifest(path: str | Path, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"subject_id", "study_id", "dicom_id"}
    columns = set(pd.read_csv(path, nrows=0).columns)
    missing = sorted(required - columns)
    if missing:
        raise ValueError(f"Processed {split} CSV is missing columns: {missing}")
    usecols = ["subject_id", "study_id", "dicom_id"]
    if "ViewPosition" in columns:
        usecols.append("ViewPosition")
    frame = pd.read_csv(path, usecols=usecols, dtype={"dicom_id": "string"})
    if frame.empty:
        raise ValueError(f"Processed {split} CSV contains no rows")
    if frame[list(required)].isna().any().any():
        raise ValueError(f"Processed {split} CSV contains empty identity fields")

    frame = frame.copy()
    frame["dicom_id"] = frame["dicom_id"].astype(str)
    frame["_row_order"] = np.arange(len(frame), dtype=np.int64)
    frame["_study_order"] = frame.groupby(
        ["subject_id", "study_id"], sort=False
    ).ngroup()
    if "ViewPosition" in frame:
        view_ids = frame["ViewPosition"].map(_view_id)
    else:
        view_ids = pd.Series(3, index=frame.index)
    frame["_anchor_rank"] = view_ids.map(lambda value: _VIEW_PRIORITY.get(value, 3))
    anchors = (
        frame.sort_values(
            ["_study_order", "_anchor_rank", "_row_order"], kind="stable"
        )
        .drop_duplicates(["subject_id", "study_id"], keep="first")
        .sort_values("_study_order", kind="stable")
        .reset_index(drop=True)
    )
    if anchors["dicom_id"].duplicated().any():
        raise ValueError(f"Processed {split} CSV selects duplicate anchor identifiers")
    return frame, anchors


def _validate_columns(path: str | Path, required: Sequence[str], source_name: str) -> None:
    columns = set(pd.read_csv(path, nrows=0).columns)
    missing = sorted(set(required) - columns)
    if missing:
        raise ValueError(f"{source_name} CSV is missing columns: {missing}")


def _load_ms_cxr(
    csv_path: str | Path,
    project_split_by_dicom: Mapping[str, str],
    target_ids: set[str],
) -> tuple[dict[str, pd.DataFrame], int, int]:
    _validate_columns(csv_path, MS_CXR_COLUMNS, "MS-CXR")
    frame = pd.read_csv(
        csv_path,
        usecols=list(MS_CXR_COLUMNS),
        dtype={"dicom_id": "string", "split": "string"},
    )
    if frame["dicom_id"].isna().any():
        raise ValueError("MS-CXR CSV contains empty identifiers")
    frame["dicom_id"] = frame["dicom_id"].astype(str)
    project_split = frame["dicom_id"].map(project_split_by_dicom)
    unassigned_mask = project_split.isna()
    unassigned = int(unassigned_mask.sum())
    input_row_count = len(frame)
    print(
        "MS-CXR rows absent from project manifests and dropped: "
        f"{unassigned} / {input_row_count} "
        f"({_percentage(unassigned, input_row_count):.2f}%)"
    )
    frame = frame.loc[~unassigned_mask].copy()
    project_split = project_split.loc[~unassigned_mask]
    source_split = frame["split"].map(_normalise_project_split)
    mismatch_count = int((source_split != project_split).sum())

    groups = {
        str(dicom_id): group.copy()
        for dicom_id, group in frame[frame["dicom_id"].isin(target_ids)].groupby(
            "dicom_id", sort=False
        )
    }
    return groups, mismatch_count, len(frame)


def _safe_json_dump(value: object, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, separators=(",", ":"))


def _percentage(count: int, total: int) -> float:
    return 100.0 * count / total if total else 0.0


def build_mask_caches(
    *,
    manifest_paths: Mapping[str, str | Path],
    chexmask_csv: str | Path,
    ms_cxr_csv: str | Path,
    output_dir: str | Path,
    split: str = "all",
    limit: int | None = None,
    dice_threshold: float = 0.7,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> dict[str, dict[str, int | float]]:
    """Build one fixed-row memmapped mask cache for each requested project split.

    Study selection is driven by the project's manifests and uses the same PA ->
    AP -> lateral anchor priority as ``MIMIC_CXR_Dataset``.  Only valid masks are
    copied into the final compact array; an absent JSON entry is the dataset's
    ``valid=False`` signal.
    """

    if split not in {*PROJECT_SPLITS, "all"}:
        raise ValueError("split must be train, val, test, or all")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    if not math.isfinite(dice_threshold) or not 0.0 <= dice_threshold <= 1.0:
        raise ValueError("dice_threshold must be in [0, 1]")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    missing_paths = sorted(set(PROJECT_SPLITS) - set(manifest_paths))
    if missing_paths:
        raise ValueError(f"manifest_paths is missing splits: {missing_paths}")
    unavailable_inputs = sum(
        not Path(path).is_file()
        for path in (chexmask_csv, ms_cxr_csv, *manifest_paths.values())
    )
    if unavailable_inputs:
        raise FileNotFoundError(
            f"{unavailable_inputs} required input files are unavailable; verify that "
            "the private data volume is mounted and local_config paths are current"
        )

    selected_splits = PROJECT_SPLITS if split == "all" else (split,)
    full_manifests: dict[str, pd.DataFrame] = {}
    anchors_by_split: dict[str, pd.DataFrame] = {}
    for split_name in PROJECT_SPLITS:
        full_frame, anchors = _read_project_manifest(
            manifest_paths[split_name], split_name
        )
        full_manifests[split_name] = full_frame
        if split_name in selected_splits and limit is not None:
            anchors = anchors.iloc[:limit].copy()
        anchors_by_split[split_name] = anchors

    project_split_by_dicom: dict[str, str] = {}
    for split_name, frame in full_manifests.items():
        for dicom_id in frame["dicom_id"]:
            previous = project_split_by_dicom.setdefault(str(dicom_id), split_name)
            if previous != split_name:
                raise ValueError("Project manifests assign one image to multiple splits")

    target_locations: dict[str, tuple[str, int]] = {}
    for split_name in selected_splits:
        for row, dicom_id in enumerate(anchors_by_split[split_name]["dicom_id"]):
            dicom_id = str(dicom_id)
            if dicom_id in target_locations:
                raise ValueError("Selected study anchors contain duplicate identifiers")
            target_locations[dicom_id] = (split_name, row)
    if not target_locations:
        raise ValueError("No project studies were selected")

    ms_groups, mismatch_count, ms_row_count = _load_ms_cxr(
        ms_cxr_csv, project_split_by_dicom, set(target_locations)
    )
    print(
        "MS-CXR rows whose source split differs from the project split: "
        f"{mismatch_count} / {ms_row_count}"
    )
    _validate_columns(chexmask_csv, CHEXMASK_COLUMNS, "CheXmask")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    process_tag = str(os.getpid())
    temp_masks = {
        split_name: output_dir / f".masks_{split_name}.{process_tag}.tmp.npy"
        for split_name in selected_splits
    }
    compact_masks = {
        split_name: output_dir / f".masks_{split_name}.{process_tag}.compact.npy"
        for split_name in selected_splits
    }
    temp_indices = {
        split_name: output_dir / f".index_{split_name}.{process_tag}.tmp.json"
        for split_name in selected_splits
    }
    temp_bbox_masks = {
        split_name: output_dir / f".masks_bbox_{split_name}.{process_tag}.npy"
        for split_name in selected_splits
    }
    temp_bbox_indices = {
        split_name: output_dir / f".index_bbox_{split_name}.{process_tag}.tmp.json"
        for split_name in selected_splits
    }
    arrays: dict[str, np.memmap] = {}
    sources: dict[str, int] = {}
    lung_available: set[str] = set()
    bbox_available: set[str] = set()
    seen_target_chexmask: set[str] = set()
    matched_target_rows = 0
    remaining_smoke_ids = set(target_locations) if limit is not None else None

    try:
        for split_name in selected_splits:
            arrays[split_name] = np.lib.format.open_memmap(
                temp_masks[split_name],
                mode="w+",
                dtype=np.uint8,
                shape=(len(anchors_by_split[split_name]), *MASK_SIZE),
            )

        # Resize(512) crops the long axis, so a box hugging the top or bottom
        # edge can be cropped away entirely. Measured on MS-CXR v1.1.0: 3 of
        # 1,448 boxes vanish and exactly one DICOM loses every box it has.
        # Aborting the whole 228k-study build over one study is the wrong
        # trade; fall back to that study's lung mask and report the count.
        bbox_cropped_away = 0
        # Per-(study, finding) boxes, kept separate from the pooled union above.
        # The pooled mask answers "is the model inside any annotated region";
        # these answer "is the model on THIS finding", which is the only
        # question an expert box can actually settle.
        per_label_masks: dict[str, list[tuple[str, int, np.ndarray]]] = {
            split_name: [] for split_name in selected_splits
        }
        unmapped_categories: dict[str, int] = {}
        per_label_cropped_away = 0
        for dicom_id, rows in ms_groups.items():
            split_name, row = target_locations[dicom_id]

            for category, category_rows in rows.groupby("category_name", sort=False):
                label_index = category_to_label_index(category)
                if label_index is None:
                    key = str(category)
                    unmapped_categories[key] = unmapped_categories.get(key, 0) + 1
                    continue
                cached_label = transform_mask_geometry(
                    rasterize_bbox_union(category_rows)
                )
                if not cached_label.any():
                    per_label_cropped_away += 1
                    continue
                per_label_masks[split_name].append(
                    (dicom_id, label_index, cached_label)
                )

            bbox_mask = rasterize_bbox_union(rows)
            cached_bbox = transform_mask_geometry(bbox_mask)
            if not cached_bbox.any():
                bbox_cropped_away += 1
                continue
            arrays[split_name][row] = cached_bbox
            sources[dicom_id] = 1
            bbox_available.add(dicom_id)
        print(
            "MS-CXR studies whose boxes fell outside the centre crop "
            f"(fell back to the lung mask): {bbox_cropped_away} / {len(ms_groups)}"
        )
        print(
            "MS-CXR per-finding boxes cropped away: "
            f"{per_label_cropped_away}"
        )
        if unmapped_categories:
            print(
                "MS-CXR categories with no CheXpert column (skipped): "
                + ", ".join(
                    f"{name} x{count}"
                    for name, count in sorted(unmapped_categories.items())
                )
            )

        dtype = {
            "dicom_id": "string",
            "Left Lung": "string",
            "Right Lung": "string",
        }
        reader = pd.read_csv(
            chexmask_csv,
            usecols=list(CHEXMASK_COLUMNS),
            dtype=dtype,
            chunksize=chunk_size,
        )
        for chunk in reader:
            chunk = chunk.loc[:, list(CHEXMASK_COLUMNS)]
            dice_values = pd.to_numeric(chunk["Dice RCA (Mean)"], errors="coerce")
            for position, row in enumerate(chunk.itertuples(index=False, name=None)):
                dicom_id, _dice, left_rle, right_rle, height, width = row
                dicom_id = str(dicom_id)
                if dicom_id not in target_locations:
                    continue
                matched_target_rows += 1
                if dicom_id in seen_target_chexmask:
                    raise ValueError("CheXmask contains a duplicate selected identifier")
                seen_target_chexmask.add(dicom_id)
                if remaining_smoke_ids is not None:
                    remaining_smoke_ids.discard(dicom_id)

                dice = dice_values.iloc[position]
                if pd.isna(dice) or float(dice) < dice_threshold:
                    continue
                lung_mask = decode_lung_union(left_rle, right_rle, height, width)
                if not lung_mask.any():
                    continue
                cached_lung = transform_mask_geometry(lung_mask)
                if not cached_lung.any():
                    continue
                lung_available.add(dicom_id)
                if dicom_id in bbox_available:
                    continue
                split_name, cache_row = target_locations[dicom_id]
                arrays[split_name][cache_row] = cached_lung
                sources[dicom_id] = 0
            if remaining_smoke_ids is not None and not remaining_smoke_ids:
                break

        if matched_target_rows == 0:
            raise ValueError(
                "CheXmask dicom_id join produced zero selected-manifest matches; "
                "refusing to write an empty lung-mask join"
            )

        stats: dict[str, dict[str, int | float]] = {}
        for split_name in selected_splits:
            anchors = anchors_by_split[split_name]
            ids = [str(value) for value in anchors["dicom_id"]]
            valid_rows = [
                (manifest_row, dicom_id)
                for manifest_row, dicom_id in enumerate(ids)
                if dicom_id in sources
            ]
            arrays[split_name].flush()
            compact = np.lib.format.open_memmap(
                compact_masks[split_name],
                mode="w+",
                dtype=np.uint8,
                shape=(len(valid_rows), *MASK_SIZE),
            )
            index = {}
            for cache_row, (manifest_row, dicom_id) in enumerate(valid_rows):
                compact[cache_row] = arrays[split_name][manifest_row]
                index[dicom_id] = {
                    "row": cache_row,
                    "mask_source": sources[dicom_id],
                }
            compact.flush()
            del compact
            _safe_json_dump(index, temp_indices[split_name])

            total = len(ids)
            lung_count = sum(dicom_id in lung_available for dicom_id in ids)
            bbox_count = sum(dicom_id in bbox_available for dicom_id in ids)
            no_mask_count = sum(dicom_id not in sources for dicom_id in ids)
            stats[split_name] = {
                "studies": total,
                "lung_masks": lung_count,
                "bbox_masks": bbox_count,
                "no_mask": no_mask_count,
                "lung_percent": _percentage(lung_count, total),
                "bbox_percent": _percentage(bbox_count, total),
            }
            print(
                f"[{split_name}] studies={total}; "
                f"lung={lung_count} ({_percentage(lung_count, total):.2f}%); "
                f"bbox={bbox_count} ({_percentage(bbox_count, total):.2f}%); "
                f"no_mask={no_mask_count}"
            )

        # Per-pathology MS-CXR cache. Written as a SECOND pair of files rather
        # than folded into the index above, so a checkpoint trained before this
        # existed still loads the pooled cache unchanged, and so the strong term
        # can be disabled by simply not shipping these two files.
        for split_name in selected_splits:
            entries = per_label_masks[split_name]
            bbox_array = np.lib.format.open_memmap(
                temp_bbox_masks[split_name],
                mode="w+",
                dtype=np.uint8,
                shape=(len(entries), *MASK_SIZE),
            )
            bbox_index: dict[str, dict[str, int]] = {}
            for cache_row, (dicom_id, label_index, mask) in enumerate(entries):
                bbox_array[cache_row] = mask
                bbox_index[f"{dicom_id}:{label_index}"] = {
                    "row": cache_row,
                    "label_index": int(label_index),
                }
            bbox_array.flush()
            del bbox_array
            _safe_json_dump(bbox_index, temp_bbox_indices[split_name])
            stats[split_name]["bbox_label_pairs"] = len(entries)
            stats[split_name]["bbox_label_studies"] = len(
                {dicom_id for dicom_id, _, _ in entries}
            )
            print(
                f"[{split_name}] per-finding boxes: {len(entries)} pairs over "
                f"{stats[split_name]['bbox_label_studies']} studies"
            )

        arrays.clear()
        for split_name in selected_splits:
            os.replace(compact_masks[split_name], output_dir / f"masks_{split_name}.npy")
            os.replace(temp_indices[split_name], output_dir / f"index_{split_name}.json")
            os.replace(
                temp_bbox_masks[split_name],
                output_dir / f"masks_bbox_{split_name}.npy",
            )
            os.replace(
                temp_bbox_indices[split_name],
                output_dir / f"index_bbox_{split_name}.json",
            )
        return stats
    finally:
        arrays.clear()
        for path in (
            *temp_masks.values(),
            *compact_masks.values(),
            *temp_indices.values(),
            *temp_bbox_masks.values(),
            *temp_bbox_indices.values(),
        ):
            if path.exists():
                path.unlink()


def _project_manifest_paths() -> dict[str, str]:
    try:
        from local_config import (
            PROCESSED_TEST_CSV,
            PROCESSED_TRAIN_CSV,
            PROCESSED_VAL_CSV,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Project manifests require configs/env_config.yaml; copy the example "
            "and configure processed_train_csv/val/test before building masks"
        ) from exc
    return {
        "train": PROCESSED_TRAIN_CSV,
        "val": PROCESSED_VAL_CSV,
        "test": PROCESSED_TEST_CSV,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build two-tier CheXmask/MS-CXR explanation-mask caches."
    )
    parser.add_argument("--split", choices=(*PROJECT_SPLITS, "all"), default="all")
    parser.add_argument("--chexmask-csv", type=Path, default=DEFAULT_CHEXMASK_CSV)
    parser.add_argument("--ms-cxr-csv", type=Path, default=DEFAULT_MS_CXR_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None, help="Study limit per split")
    parser.add_argument("--dice-threshold", type=float, default=0.7)
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Inspect columns, prefix row count, and Dice-score range, then exit",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.inspect:
        inspect_chexmask(args.chexmask_csv)
        return 0
    build_mask_caches(
        manifest_paths=_project_manifest_paths(),
        chexmask_csv=args.chexmask_csv,
        ms_cxr_csv=args.ms_cxr_csv,
        output_dir=args.output_dir,
        split=args.split,
        limit=args.limit,
        dice_threshold=args.dice_threshold,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
