#!/usr/bin/env python3
"""Evaluate Stage-1 Grad-CAM explanations from a live checkpoint.

Unlike ``evaluate_stage1.py``, this command must load the model and keep
autograd enabled: Grad-CAM differentiates a class score with respect to live
MHCAC activations.  The model runs in ``eval`` mode and no optimizer is ever
created, so parameters are not updated.

The output is privacy-sensitive MIMIC-CXR derivative data.  The default lives
under the repository's ignored ``outputs/`` tree.  A non-ignored destination
inside the repository is rejected; identifiers are never printed or used in
figure filenames.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.evaluation.explanation_metrics import (  # noqa: E402
    BBOX_MASK_SOURCE,
    LUNG_MASK_SOURCE,
    summarize,
)

LOGGER = logging.getLogger("evaluate_explanation")
DEFAULT_MS_CXR_CSV = Path(
    "/mnt/drive1tb/datasets/ms-cxr/MS_CXR_Local_Alignment_v1.1.0.csv"
)
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "outputs" / "explanation_evaluation"
METRIC_GRID = (112, 112)
TOP_K = 0.5
COVERAGE_TAU = 0.01
MS_CXR_COLUMNS = (
    "dicom_id",
    "x",
    "y",
    "w",
    "h",
    "image_width",
    "image_height",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--cfg-path", required=True, type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--mask-cache-dir", required=True, type=Path)
    parser.add_argument("--ms-cxr-csv", type=Path, default=DEFAULT_MS_CXR_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--export-figures",
        type=int,
        default=0,
        metavar="N",
        help="Export N identifier-free PNG overlays (0 disables figures).",
    )
    parser.add_argument(
        "--save-cams",
        action="store_true",
        help="Write cams.npz. This artifact is patient-derived private data.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def _is_inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _assert_private_output_location(output_dir: Path) -> Path:
    """Reject a repository-local output unless Git confirms it is ignored."""

    resolved = output_dir.expanduser().resolve()
    repo = _REPO_ROOT.resolve()
    if not _is_inside(resolved, repo):
        return resolved

    relative = resolved.relative_to(repo)
    try:
        ignored = subprocess.run(
            ["git", "check-ignore", "--no-index", "-q", "--", str(relative)],
            cwd=repo,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode == 0
    except FileNotFoundError as exc:
        raise ValueError(
            "refusing a repository-local output because git is unavailable to "
            "verify its ignore status"
        ) from exc
    if not ignored:
        raise ValueError(
            "refusing to write patient-derived artifacts inside the repository: "
            "choose a private path outside the repo or a path covered by .gitignore"
        )

    LOGGER.warning(
        "PRIVACY WARNING: output is inside the repository, but Git confirms the "
        "destination is ignored. Never force-add these artifacts."
    )
    return resolved


def _validate_inputs(args: argparse.Namespace) -> Path:
    if not args.checkpoint.is_file():
        raise FileNotFoundError("Stage-1 checkpoint is unavailable")
    if not args.cfg_path.is_file():
        raise FileNotFoundError("Stage-1 config is unavailable")
    if not args.ms_cxr_csv.is_file():
        raise FileNotFoundError("MS-CXR CSV is unavailable")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.export_figures < 0:
        raise ValueError("--export-figures must be non-negative")
    for name in (f"masks_{args.split}.npy", f"index_{args.split}.json"):
        if not (args.mask_cache_dir / name).is_file():
            raise FileNotFoundError("the requested split mask cache is incomplete")
    return _assert_private_output_location(args.output_dir)


def _load_ms_cxr_groups(csv_path: Path):
    """Load bbox rows without ever consulting MS-CXR's incompatible split column."""

    import pandas as pd

    header = pd.read_csv(csv_path, nrows=0)
    missing = sorted(set(MS_CXR_COLUMNS) - set(header.columns))
    if missing:
        raise ValueError(f"MS-CXR CSV is missing required columns: {missing}")
    frame = pd.read_csv(
        csv_path,
        usecols=list(MS_CXR_COLUMNS),
        dtype={"dicom_id": "string"},
    )
    if frame["dicom_id"].isna().any():
        raise ValueError("MS-CXR CSV contains empty identifiers")
    frame["dicom_id"] = frame["dicom_id"].astype(str)
    return {
        str(key): group.reset_index(drop=True)
        for key, group in frame.groupby("dicom_id", sort=False)
    }


def _transform_individual_boxes(rows) -> tuple[np.ndarray | None, int]:
    """Map every original-pixel box through the verified cache geometry."""

    from preporcessing.build_explanation_masks import (
        rasterize_bbox_union,
        transform_mask_geometry,
    )

    # ``rasterize_bbox_union`` validates the mask-builder's full input schema,
    # which includes a split field even though geometry never reads it. Supply
    # a synthetic project-manifest marker; the incompatible MS-CXR source split
    # was intentionally not loaded by ``_load_ms_cxr_groups``.
    geometry_rows = rows.copy()
    geometry_rows["split"] = "project_manifest"

    masks = []
    cropped_out = 0
    for position in range(len(geometry_rows)):
        source_mask = rasterize_bbox_union(geometry_rows.iloc[[position]])
        transformed = transform_mask_geometry(source_mask, output_size=METRIC_GRID)
        if transformed.any():
            masks.append(transformed > 0)
        else:
            cropped_out += 1
    if not masks:
        return None, cropped_out
    return np.stack(masks, axis=0), cropped_out


def _build_runtime(args: argparse.Namespace):
    """Lazy-load the Stage-1 stack, checkpoint and project-manifest dataset."""

    import torch
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader

    from local_config import VIS_ROOT
    from model.lavis import tasks
    from model.lavis.common.config import Config
    from model.lavis.common.registry import registry
    from model.lavis.data.ReportDataset import MIMIC_CXR_Dataset

    registry.mapping["paths"]["cache_root"] = "."
    options = ["run.distributed=false", "run.world_size=1", "run.gpu=0"]
    cfg = Config(SimpleNamespace(cfg_path=str(args.cfg_path), options=options))

    # Apply path values as OmegaConf values rather than dot-list strings so
    # spaces and punctuation in private mount paths cannot be reinterpreted.
    OmegaConf.update(cfg.config, "model.load_finetuned", True, merge=False)
    OmegaConf.update(
        cfg.config,
        "model.finetuned",
        str(args.checkpoint.expanduser().resolve()),
        merge=False,
    )
    OmegaConf.update(
        cfg.config,
        "model.explanation.mask_cache_dir",
        str(args.mask_cache_dir.expanduser().resolve()),
        merge=False,
    )
    # Evaluation needs the transformed radiograph for optional thesis figures;
    # using images also avoids silently mixing a stale feature cache with a new
    # checkpoint projection.
    OmegaConf.update(cfg.config, "run.feature_cache_dir", None, merge=False)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("a CUDA device was requested but CUDA is unavailable")

    task = tasks.setup_task(cfg)
    model = task.build_model(cfg).to(device)
    model.eval()
    dataset = MIMIC_CXR_Dataset(
        vis_processor=None,
        text_processor=None,
        vis_root=VIS_ROOT,
        split=args.split,
        cfg=cfg,
        truncate=args.limit,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=dataset.collater,
    )
    return cfg, model, loader, device


def _autocast_context(cfg, device):
    import torch

    if device.type != "cuda" or not bool(cfg.run_cfg.get("amp", False)):
        return nullcontext()
    name = str(cfg.run_cfg.get("amp_dtype", "float16")).lower()
    dtype = torch.bfloat16 if name in {"bfloat16", "bf16"} else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _to_device(value, device):
    import torch

    if torch.is_tensor(value):
        return value.to(device, non_blocking=device.type == "cuda")
    return value


def _normalize_cam(cam: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Min-max normalize each CAM independently; constant maps become zero."""

    array = np.asarray(cam, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError("raw CAM must have shape [B,H,W]")
    flat = array.reshape(len(array), -1)
    minimum = flat.min(axis=1)[:, None, None]
    maximum = flat.max(axis=1)[:, None, None]
    scale = maximum - minimum
    normalized = np.divide(
        array - minimum,
        scale,
        out=np.zeros_like(array),
        where=scale > eps,
    )
    return np.clip(normalized, 0.0, 1.0)


def _compute_batch_cams(model, batch, device, amp_context) -> dict[str, np.ndarray]:
    """Run the image-only MHCAC path and reuse the training Grad-CAM helpers."""

    import torch
    import torch.nn.functional as F

    from mhcac.explanation import grad_cam, logit_difference_squared

    image = _to_device(batch.get("image"), device)
    cached = {
        name: _to_device(batch[f"{name}_feat"], device)
        for name in ("biovil", "pubmedclip", "swin", "raddino")
        if f"{name}_feat" in batch
    }
    aux_cached = {
        name: _to_device(batch[f"aux_{name}_feat"], device)
        for name in ("biovil", "pubmedclip", "swin", "raddino")
        if f"aux_{name}_feat" in batch
    }
    labels = _to_device(batch["classification_labels"], device)
    classification_mask = _to_device(batch["classification_mask"], device).bool()
    explanation_valid = _to_device(batch["explanation_mask_valid"], device).bool()
    valid = classification_mask & explanation_valid

    model.zero_grad(set_to_none=True)
    with torch.enable_grad(), amp_context:
        shared_visual = model._encode_image_streams(
            image,
            apply_aug=False,
            cached=cached,
            aux_image=_to_device(batch.get("aux_image"), device),
            aux_cached=aux_cached,
            aux_mask=_to_device(batch.get("aux_mask"), device),
            anchor_view_id=_to_device(batch.get("anchor_view_id"), device),
            aux_view_ids=_to_device(batch.get("aux_view_ids"), device),
        )

        streams = None
        model.mhcac.capture_streams = True
        try:
            logits, _, _, _, _ = model.mhcac(
                shared_visual,
                text_embeddings=None,
                labels=labels,
                sample_mask=classification_mask,
            )
            streams = model.mhcac._last_cam_streams
        finally:
            model.mhcac.capture_streams = False
            model.mhcac._last_cam_streams = None

        if not streams:
            raise RuntimeError("MHCAC did not expose any Grad-CAM streams")
        selected_names = getattr(model, "explanation_streams", None)
        if selected_names is not None:
            streams = {name: value for name, value in streams.items() if name in selected_names}
        if not streams:
            raise RuntimeError("no captured stream matches model.explanation.streams")

        score, positive = logit_difference_squared(logits, labels, sample_mask=valid)
        if not bool(positive.any().item()):
            raise RuntimeError("CAM computation received no eligible positive study")

        result = {}
        # ``grad_cam`` defaults to create_graph=True. Keeping that setting also
        # retains the shared graph while each configured stream is evaluated;
        # the detached NumPy output is the only object retained afterwards.
        for name, (activation, grid_hw) in streams.items():
            cam = grad_cam(score, activation, grid_hw)
            cam = F.interpolate(
                cam[:, None], size=METRIC_GRID, mode="bilinear", align_corners=False
            ).squeeze(1)
            result[name] = _normalize_cam(cam.detach().float().cpu().numpy())
    return result


def _figure_image(image_tensor) -> np.ndarray:
    array = image_tensor.detach().float().cpu().numpy()
    if array.ndim == 3:
        array = array[0]
    if array.ndim != 2:
        raise ValueError("figure export expects a single-channel radiograph")
    finite = np.isfinite(array)
    if not finite.all():
        raise ValueError("figure image contains non-finite pixels")
    minimum = float(array.min())
    maximum = float(array.max())
    if maximum > minimum:
        array = (array - minimum) / (maximum - minimum)
    else:
        array = np.zeros_like(array)
    return np.clip(array, 0.0, 1.0)


def _draw_annotation(axis, mask: np.ndarray, source: int, boxes: np.ndarray | None) -> None:
    height, width = axis.images[0].get_array().shape[:2]
    contour_kwargs = {
        "levels": [0.5],
        "origin": "upper",
        "extent": (0, width, height, 0),
        "linewidths": 1.25,
    }
    if source == BBOX_MASK_SOURCE and boxes is not None:
        for box in boxes:
            axis.contour(box.astype(float), colors="white", **contour_kwargs)
    else:
        axis.contour(mask.astype(float), colors="cyan", **contour_kwargs)


def _export_figure(
    path: Path,
    image: np.ndarray,
    cams: dict[str, np.ndarray],
    mask: np.ndarray,
    source: int,
    boxes: np.ndarray | None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "--export-figures requires matplotlib; install the eval-plots extra"
        ) from exc

    ordered = sorted(cams)
    figure, axes = plt.subplots(1, len(ordered) + 1, figsize=(4 * (len(ordered) + 1), 4))
    axes = np.atleast_1d(axes)
    axes[0].imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
    _draw_annotation(axes[0], mask, source, boxes)
    axes[0].set_title("Radiograph + annotation")

    for axis, name in zip(axes[1:], ordered, strict=True):
        axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        axis.imshow(
            cams[name][0],
            cmap="jet",
            alpha=0.45,
            vmin=0.0,
            vmax=1.0,
            extent=(0, image.shape[1], image.shape[0], 0),
        )
        _draw_annotation(axis, mask, source, boxes)
        axis.set_title(f"Grad-CAM: {name}")

    for axis in axes:
        axis.set_axis_off()
    annotation_name = "MS-CXR expert boxes" if source == BBOX_MASK_SOURCE else "lung prior"
    figure.suptitle(annotation_name)
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_cams(
    path: Path,
    cams: dict[str, np.ndarray],
    masks: np.ndarray,
    mask_sources: np.ndarray,
    split_positions: np.ndarray,
) -> None:
    arrays: dict[str, np.ndarray] = {
        "masks": masks.astype(np.uint8, copy=False),
        "mask_sources": mask_sources.astype(np.int8, copy=False),
        "split_positions": split_positions.astype(np.int64, copy=False),
    }
    for name, values in cams.items():
        safe_name = "".join(character if character.isalnum() else "_" for character in name)
        key = f"cam_{safe_name}"
        if key in arrays:
            raise ValueError("two stream names collapse to the same NPZ key")
        arrays[key] = values.astype(np.float32, copy=False)
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def _evaluate(args: argparse.Namespace, output_dir: Path) -> dict[str, object]:
    groups = _load_ms_cxr_groups(args.ms_cxr_csv)
    cfg, model, loader, device = _build_runtime(args)

    cam_lists: dict[str, list[np.ndarray]] = {}
    masks: list[np.ndarray] = []
    mask_sources: list[int] = []
    boxes_by_sample: list[np.ndarray | None] = []
    split_positions: list[int] = []
    counters = {
        "studies_seen": 0,
        "eligible_positive_with_mask": 0,
        "skipped_invalid_classification": 0,
        "skipped_without_mask": 0,
        "skipped_without_positive_label": 0,
        "boxes_cropped_out": 0,
    }
    figure_count = 0
    figure_dir = output_dir / "figures"
    if args.export_figures:
        figure_dir.mkdir(parents=True, exist_ok=True)

    expected_streams: tuple[str, ...] | None = None
    for split_position, batch in enumerate(loader):
        counters["studies_seen"] += 1
        if counters["studies_seen"] % 100 == 0:
            LOGGER.info(
                "processed %d studies; %d eligible CAMs",
                counters["studies_seen"],
                counters["eligible_positive_with_mask"],
            )
        if not bool(batch["classification_mask"].reshape(-1)[0].item()):
            counters["skipped_invalid_classification"] += 1
            continue
        if not bool(batch["explanation_mask_valid"].reshape(-1)[0].item()):
            counters["skipped_without_mask"] += 1
            continue
        if not bool((batch["classification_labels"] == 1).any().item()):
            counters["skipped_without_positive_label"] += 1
            continue

        source = int(batch["explanation_mask_source"].reshape(-1)[0].item())
        if source not in (LUNG_MASK_SOURCE, BBOX_MASK_SOURCE):
            raise ValueError("mask cache produced an unknown mask_source")

        box_masks = None
        if source == BBOX_MASK_SOURCE:
            # The project's dataset/manifest selected this sample and split.
            # MS-CXR's own `split` column was not loaded and cannot influence it.
            key = str(batch["dicom_id"][0])
            rows = groups.get(key)
            if rows is None:
                raise ValueError(
                    "a bbox-backed cache entry has no matching MS-CXR annotation"
                )
            box_masks, cropped_out = _transform_individual_boxes(rows)
            counters["boxes_cropped_out"] += cropped_out
            if box_masks is None:
                raise ValueError("all expert boxes for a bbox-backed sample fell outside the crop")

        cams = _compute_batch_cams(
            model,
            batch,
            device,
            _autocast_context(cfg, device),
        )
        stream_names = tuple(sorted(cams))
        if expected_streams is None:
            expected_streams = stream_names
            cam_lists = {name: [] for name in stream_names}
        elif stream_names != expected_streams:
            raise RuntimeError("captured Grad-CAM streams changed between samples")

        mask = batch["explanation_mask"][0].detach().cpu().numpy() > 0
        for name in expected_streams:
            cam_lists[name].append(cams[name][0])
        masks.append(mask)
        mask_sources.append(source)
        boxes_by_sample.append(box_masks)
        split_positions.append(split_position)
        counters["eligible_positive_with_mask"] += 1

        if figure_count < args.export_figures:
            image = _figure_image(batch["image"][0])
            _export_figure(
                figure_dir / f"sample_{figure_count + 1:04d}.png",
                image,
                cams,
                mask,
                source,
                box_masks,
            )
            figure_count += 1

    mask_array = (
        np.stack(masks, axis=0).astype(bool, copy=False)
        if masks
        else np.zeros((0, *METRIC_GRID), dtype=bool)
    )
    source_array = np.asarray(mask_sources, dtype=np.int8)
    cam_arrays = {
        name: np.stack(values, axis=0).astype(np.float32, copy=False)
        for name, values in cam_lists.items()
    }
    stream_reports = {
        name: summarize(
            values,
            mask_array,
            source_array,
            boxes_by_sample,
            k=TOP_K,
            tau=COVERAGE_TAU,
        ).to_dict()
        for name, values in cam_arrays.items()
    }

    cams_path = None
    if args.save_cams:
        cams_path = output_dir / "cams.npz"
        _write_cams(
            cams_path,
            cam_arrays,
            mask_array,
            source_array,
            np.asarray(split_positions, dtype=np.int64),
        )

    return {
        "schema_version": 1,
        "split": args.split,
        "settings": {
            "top_k": TOP_K,
            "annotation_coverage_tau": COVERAGE_TAU,
            "metric_grid": list(METRIC_GRID),
            "checkpoint_filename": args.checkpoint.name,
            "config_filename": args.cfg_path.name,
            "device": str(device),
            "limit": args.limit,
        },
        "counts": counters,
        "streams": stream_reports,
        "artifacts": {
            "cams_npz": cams_path.name if cams_path is not None else None,
            "figures_exported": figure_count,
        },
        "privacy": (
            "PNG/NPZ artifacts are patient-derived MIMIC-CXR data and must remain "
            "on access-controlled storage; no identifiers are included in this JSON."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        output_dir = _validate_inputs(args)
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = _evaluate(args, output_dir)
        _write_json(output_dir / "metrics.json", payload)
    except Exception as exc:  # noqa: BLE001 - suppress identifiers in nested errors
        LOGGER.error(
            "explanation evaluation failed (%s); patient identifiers are suppressed",
            type(exc).__name__,
        )
        return 1

    LOGGER.info(
        "wrote aggregate metrics for %d eligible studies to %s",
        payload["counts"]["eligible_positive_with_mask"],
        output_dir / "metrics.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
