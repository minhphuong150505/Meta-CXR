#!/usr/bin/env python3
"""Create Figure-8 style expert-token attention heatmaps for META-CXR.

The figure shows MHCAC expert-token attention maps for one MIMIC-CXR test
image, split by encoder stream: RN50/BioViL, ViT/PubMedCLIP, and Swin.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import patches

PROJECT_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "model"))

import model.lavis.tasks as tasks
from local_config import VIS_ROOT
from model.lavis.common.config import Config
from model.lavis.common.registry import registry
from model.lavis.data.ReportDataset import MIMIC_CXR_Dataset


registry.mapping["paths"]["cache_root"] = "."

RUN_NAME = "07_all_three"
ENCODER_COLUMNS = [("RN50", "biovil"), ("ViT", "pubmedclip"), ("Swin", "swin")]
EXPERT_ROWS = [0, 1, 2, 3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", default="/mnt/meta-cxr-checkpoint")
    parser.add_argument("--run-name", default=RUN_NAME)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--choose-positive", action="store_true")
    parser.add_argument("--sample-search-limit", type=int, default=200)
    parser.add_argument("--attention-layer", default="last", choices=["last", "mean"])
    parser.add_argument("--output-dir", default="/home/phuong/META-CXR/output/figure8_attention_heatmaps")
    parser.add_argument("--gcs-output", default="gs://meta-cxr-checkpoint/eval/MedGemma_QFormer/figure8_attention_heatmaps")
    parser.add_argument("--skip-upload", action="store_true")
    return parser.parse_args()


def set_seed(seed: int = 16) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_cfg(run_name: str) -> Config:
    cfg_path = PROJECT_DIR / "pretraining" / "configs" / "encoder_comparison" / f"{run_name}.yaml"
    return Config(SimpleNamespace(cfg_path=str(cfg_path), options=None))


def load_checkpoint(path: Path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def filter_state_dict_for_model(model, state_dict):
    model_state = model.state_dict()
    filtered = {}
    skipped = []
    for key, value in state_dict.items():
        if key in model_state and hasattr(value, "shape") and tuple(value.shape) != tuple(model_state[key].shape):
            skipped.append((key, tuple(value.shape), tuple(model_state[key].shape)))
            continue
        filtered[key] = value
    return filtered, skipped


def build_model(run_name: str, checkpoint_root: Path, device: torch.device):
    cfg = build_cfg(run_name)
    task = tasks.setup_task(cfg)
    model = task.build_model(cfg)
    checkpoint_path = checkpoint_root / run_name / "checkpoint_best.pth"
    ckpt = load_checkpoint(checkpoint_path)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    state_dict, skipped = filter_state_dict_for_model(model, state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(
        f"[model] loaded {checkpoint_path}; missing={len(missing)}, "
        f"unexpected={len(unexpected)}, mismatched_skipped={len(skipped)}"
    )
    model.to(device)
    model.eval()
    return cfg, model


def make_dataset(cfg, truncate: int | None):
    return MIMIC_CXR_Dataset(
        vis_processor=None,
        text_processor=None,
        vis_root=VIS_ROOT,
        split="test",
        cfg=cfg,
        truncate=truncate,
    )


def choose_positive_sample(dataset, search_limit: int) -> int:
    limit = min(len(dataset), search_limit)
    best_idx = 0
    best_score = -1
    for idx in range(limit):
        item = dataset[idx]
        labels = item["classification_labels"].numpy()
        # Prefer images with positive labels among common abnormalities so the
        # heatmaps are more likely to be clinically meaningful.
        common_indices = [8, 2, 6, 5, 10]  # Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion
        score = int((labels[common_indices] == 1).sum())
        if score > best_score:
            best_score = score
            best_idx = idx
        if score >= 3:
            break
    print(f"[sample] choose_positive selected index={best_idx} with common-positive-count={best_score}")
    return best_idx


@torch.no_grad()
def compute_attention(model, image: torch.Tensor, device: torch.device, attention_layer: str):
    image = image.unsqueeze(0).to(device)
    cnn_patches, vit_patches, swin_patches, _ = model._encode_image_streams(image, apply_aug=False)
    logits, attention_list, _, _, _ = model.mhcac(
        cnn_patches=cnn_patches,
        vit_patches=vit_patches,
        swin_patches=swin_patches,
        text_embeddings=None,
        labels=None,
    )
    if attention_layer == "mean":
        attention = torch.stack(attention_list, dim=0).mean(dim=0)
    else:
        attention = attention_list[-1]
    return logits.detach().cpu(), attention.detach().cpu()


def stream_slices(model) -> dict[str, slice]:
    offsets = {}
    start = 0
    if getattr(model, "use_biovil", False):
        offsets["biovil"] = slice(start, start + 49)
        start += 49
    if getattr(model, "use_pubmedclip", False):
        offsets["pubmedclip"] = slice(start, start + 49)
        start += 49
    if getattr(model, "use_swin", False):
        offsets["swin"] = slice(start, start + 49)
        start += 49
    return offsets


def normalize_heatmap(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo, hi = np.percentile(x, [2, 98])
    if hi <= lo:
        lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def upsample_heatmap(heatmap_7x7: np.ndarray, size: int) -> np.ndarray:
    tensor = torch.tensor(heatmap_7x7, dtype=torch.float32).view(1, 1, 7, 7)
    up = F.interpolate(tensor, size=(size, size), mode="bicubic", align_corners=False)
    return normalize_heatmap(up.squeeze().numpy())


def base_image_from_tensor(image: torch.Tensor) -> np.ndarray:
    arr = image.detach().cpu().float().numpy()
    if arr.ndim == 3:
        arr = arr[0]
    arr = normalize_heatmap(arr)
    return arr


def draw_grid(base_img: np.ndarray, attention: torch.Tensor, slices: dict[str, slice], output_dir: Path, meta: dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    image_size = int(base_img.shape[-1])
    attention = attention[0]  # [expert_token, patch]

    fig = plt.figure(figsize=(7.0, 7.2), dpi=220)
    gs = fig.add_gridspec(
        nrows=len(EXPERT_ROWS) + 1,
        ncols=len(ENCODER_COLUMNS) + 1,
        width_ratios=[0.62, 1.0, 1.0, 1.0],
        height_ratios=[0.38, 1.0, 1.0, 1.0, 1.0],
        wspace=0.08,
        hspace=0.08,
    )

    axes = {}
    for r in range(len(EXPERT_ROWS) + 1):
        for c in range(len(ENCODER_COLUMNS) + 1):
            ax = fig.add_subplot(gs[r, c])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor("#f1f2f3" if r == 0 or c == 0 else "white")
            for spine in ax.spines.values():
                spine.set_color("#7a7a7a")
                spine.set_linewidth(0.7)
            axes[(r, c)] = ax

    axes[(0, 0)].axis("off")
    for col_idx, (label, _stream_name) in enumerate(ENCODER_COLUMNS, start=1):
        axes[(0, col_idx)].text(0.5, 0.5, label, ha="center", va="center", fontsize=9, fontweight="bold")

    cmap = plt.get_cmap("jet")
    cell_paths = {}
    for row_idx, expert_idx in enumerate(EXPERT_ROWS, start=1):
        axes[(row_idx, 0)].text(
            0.5,
            0.5,
            f"ET{expert_idx + 1}",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
        )
        for col_idx, (_label, stream_name) in enumerate(ENCODER_COLUMNS, start=1):
            ax = axes[(row_idx, col_idx)]
            patch_slice = slices[stream_name]
            heat = attention[expert_idx, patch_slice].numpy().reshape(7, 7)
            heat = upsample_heatmap(heat, image_size)
            ax.imshow(base_img, cmap="gray", vmin=0, vmax=1)
            ax.imshow(heat, cmap=cmap, alpha=0.46, vmin=0, vmax=1)
            ax.set_aspect("equal")

            cell_name = f"ET{expert_idx + 1}_{stream_name}.png"
            cell_path = output_dir / cell_name
            cell_fig, cell_ax = plt.subplots(figsize=(2.4, 2.4), dpi=180)
            cell_ax.imshow(base_img, cmap="gray", vmin=0, vmax=1)
            cell_ax.imshow(heat, cmap=cmap, alpha=0.46, vmin=0, vmax=1)
            cell_ax.axis("off")
            cell_fig.savefig(cell_path, bbox_inches="tight", pad_inches=0.02)
            plt.close(cell_fig)
            cell_paths[f"ET{expert_idx + 1}_{stream_name}"] = str(cell_path)

    fig.patch.set_facecolor("white")
    fig.text(
        0.015,
        0.02,
        "Figure 8. Comparison of attention heatmaps for expert tokens (ET) across RN50, ViT, and Swin encoders.",
        fontsize=8.5,
        fontweight="bold",
        ha="left",
    )
    png_path = output_dir / "figure8_attention_heatmaps.png"
    pdf_path = output_dir / "figure8_attention_heatmaps.pdf"
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)

    meta["outputs"] = {
        "figure_png": str(png_path),
        "figure_pdf": str(pdf_path),
        "cells": cell_paths,
    }
    with (output_dir / "figure8_attention_metadata.json").open("w") as f:
        json.dump(meta, f, indent=2)
    return png_path, pdf_path


def upload_dir(output_dir: Path, gcs_output: str) -> None:
    cmd = f"gcloud storage cp -r {output_dir}/* {gcs_output}/ --quiet"
    print(f"[upload] {cmd}")
    rc = os.system(cmd)
    if rc != 0:
        raise RuntimeError(f"Upload failed with rc={rc}: {cmd}")


def main() -> None:
    args = parse_args()
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_root = Path(args.checkpoint_root)
    output_dir = Path(args.output_dir)

    cfg, model = build_model(args.run_name, checkpoint_root, device)
    dataset = make_dataset(cfg, truncate=max(args.sample_index + 1, args.sample_search_limit if args.choose_positive else 1))
    sample_index = choose_positive_sample(dataset, args.sample_search_limit) if args.choose_positive else args.sample_index
    item = dataset[sample_index]

    logits, attention = compute_attention(model, item["image"], device, args.attention_layer)
    slices = stream_slices(model)
    expected_streams = {name for _label, name in ENCODER_COLUMNS}
    missing_streams = expected_streams.difference(slices)
    if missing_streams:
        raise RuntimeError(f"Model does not include streams required by Figure 8: {sorted(missing_streams)}")
    if attention.shape[-1] != 147:
        print(f"[warn] expected 147 attention patches for 3 streams, got {attention.shape[-1]}")

    base_img = base_image_from_tensor(item["image"])
    meta = {
        "run_name": args.run_name,
        "checkpoint": str(checkpoint_root / args.run_name / "checkpoint_best.pth"),
        "data_root": str(VIS_ROOT),
        "sample_index": int(sample_index),
        "dicom_id": str(item.get("dicom_id", "")),
        "image_path": str(item.get("image_path", "")),
        "attention_layer": args.attention_layer,
        "attention_shape": list(attention.shape),
        "expert_tokens": [f"ET{i + 1}" for i in EXPERT_ROWS],
        "encoder_columns": [label for label, _stream in ENCODER_COLUMNS],
        "classification_logits_shape": list(logits.shape),
    }
    png_path, pdf_path = draw_grid(base_img, attention, slices, output_dir, meta)
    print(f"[saved] {png_path}")
    print(f"[saved] {pdf_path}")

    if not args.skip_upload:
        upload_dir(output_dir, args.gcs_output)
        os.system(
            f"gcloud storage cp {PROJECT_DIR / 'create_figure8_attention_heatmaps.py'} "
            f"{args.gcs_output}/scripts/create_figure8_attention_heatmaps.py --quiet"
        )
        print(f"[uploaded] {args.gcs_output}/")


if __name__ == "__main__":
    main()
