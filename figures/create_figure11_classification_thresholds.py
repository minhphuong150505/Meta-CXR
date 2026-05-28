#!/usr/bin/env python3
"""Create Figure 11: classification thresholds for META-CXR.

Computes one-vs-rest ROC thresholds by Youden's J for positive, negative, and
uncertain classes across the 14 CheXpert-style abnormalities, then saves the
paper figure and threshold tables.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import auc, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "model"))

import model.lavis.tasks as tasks  # noqa: E402
from local_config import VIS_ROOT  # noqa: E402
from model.lavis.common.config import Config  # noqa: E402
from model.lavis.common.registry import registry  # noqa: E402
from model.lavis.data.ReportDataset import MIMIC_CXR_Dataset  # noqa: E402

# Registration imports required by the LAVIS registry.
from model.lavis.common.optims import (  # noqa: E402,F401
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from model.lavis.datasets.builders import *  # noqa: E402,F403
from model.lavis.models import *  # noqa: E402,F403
from model.lavis.processors import *  # noqa: E402,F403
from model.lavis.tasks import *  # noqa: E402,F403

registry.mapping["paths"]["cache_root"] = "."

ABNORMALITIES = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

CLASS_MAP = {"negative": 0, "positive": 1, "uncertain": 2}
CLASS_PLOT_ORDER = [
    ("positive", "Positive Class"),
    ("negative", "Negative Class"),
    ("uncertain", "Uncertain Class"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="07_all_three")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--checkpoint-root", default="/mnt/meta-cxr-checkpoint")
    parser.add_argument("--output-dir", default="/home/phuong/META-CXR/output/figure11_classification_thresholds")
    parser.add_argument(
        "--gcs-output",
        default="gs://meta-cxr-checkpoint/eval/paper_figures/figure11_classification_thresholds",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument(
        "--image-only",
        action="store_true",
        help="Use image-only forward_image. Default uses report text embeddings, matching the original threshold notebook.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-upload", action="store_true")
    return parser.parse_args()


def run_cmd(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(args, check=check, text=True, capture_output=True)


def upload_dir(path: Path, gcs_dir: str) -> None:
    run_cmd(["gcloud", "storage", "cp", "-r", str(path), gcs_dir.rstrip("/") + "/"])


def build_cfg(run_name: str) -> Config:
    cfg_path = PROJECT_DIR / "pretraining" / "configs" / "encoder_comparison" / f"{run_name}.yaml"
    return Config(SimpleNamespace(cfg_path=str(cfg_path), options=None))


def load_torch_checkpoint(path: Path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def filter_state_dict_for_model(model, state_dict: dict) -> tuple[dict, list]:
    model_state = model.state_dict()
    filtered, mismatched = {}, []
    for key, value in state_dict.items():
        if key in model_state and hasattr(value, "shape"):
            if tuple(value.shape) != tuple(model_state[key].shape):
                mismatched.append((key, tuple(value.shape), tuple(model_state[key].shape)))
                continue
        filtered[key] = value
    return filtered, mismatched


def build_model(run_name: str, checkpoint_root: Path, device: torch.device):
    cfg = build_cfg(run_name)
    task = tasks.setup_task(cfg)
    model = task.build_model(cfg)
    ckpt_path = checkpoint_root / run_name / "checkpoint_best.pth"
    ckpt = load_torch_checkpoint(ckpt_path)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    state_dict, mismatched = filter_state_dict_for_model(model, state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(
        f"[load] {ckpt_path} missing={len(missing)} unexpected={len(unexpected)} "
        f"mismatched_skipped={len(mismatched)}"
    )
    model.to(device)
    model.eval()
    return cfg, model


def make_loader(cfg: Config, split: str, sample_limit: int | None, batch_size: int, num_workers: int) -> DataLoader:
    dataset = MIMIC_CXR_Dataset(
        vis_processor=None,
        text_processor=None,
        vis_root=VIS_ROOT,
        split=split,
        cfg=cfg,
        truncate=sample_limit,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def predict_logits_with_text(model, batch: dict, device: torch.device) -> torch.Tensor:
    image = batch["image"].to(device, non_blocking=True)
    text = batch["text_output"]
    cnn_patches, vit_patches, swin_patches, _ = model._encode_image_streams(image, apply_aug=False)
    text_tokens = model.tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=model.max_txt_len,
        return_tensors="pt",
    ).to(device)
    text_output = model.Qformer.bert(
        text_tokens.input_ids,
        attention_mask=text_tokens.attention_mask,
        return_dict=True,
    )
    logits, _, _, _, _ = model.mhcac(
        cnn_patches=cnn_patches,
        vit_patches=vit_patches,
        swin_patches=swin_patches,
        text_embeddings=text_output.last_hidden_state,
        labels=None,
    )
    return logits


@torch.no_grad()
def predict_logits_image_only(model, batch: dict, device: torch.device) -> torch.Tensor:
    image = batch["image"].to(device, non_blocking=True)
    logits, _ = model.forward_image(image)
    return logits


def collect_probs_labels(args: argparse.Namespace, out_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    mode = "image_only" if args.image_only else "with_text"
    limit = "full" if args.sample_limit is None else str(args.sample_limit)
    cache_path = out_dir / f"{args.run_name}_{args.split}_{mode}_{limit}_classification_probs.npz"
    if cache_path.exists() and not args.force:
        print(f"[cache] reusing {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return data["probs"], data["labels"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, model = build_model(args.run_name, Path(args.checkpoint_root), device)
    loader = make_loader(cfg, args.split, args.sample_limit, args.batch_size, args.num_workers)
    predict_fn = predict_logits_image_only if args.image_only else predict_logits_with_text
    all_probs, all_labels = [], []

    for batch in tqdm(loader, desc=f"{args.run_name}:{args.split}:{mode}"):
        logits = predict_fn(model, batch, device)
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        labels = batch["classification_labels"].detach().cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels)

    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    np.savez_compressed(
        cache_path,
        probs=probs,
        labels=labels,
        abnormalities=np.array(ABNORMALITIES),
        class_names=np.array(list(CLASS_MAP.keys())),
    )
    print(f"[cache] wrote {cache_path} probs={probs.shape} labels={labels.shape}")
    return probs, labels


def binary_roc_for_class(labels: np.ndarray, probs: np.ndarray, abnormality_idx: int, class_idx: int) -> dict | None:
    y_true = (labels[:, abnormality_idx] == class_idx).astype(int)
    scores = probs[:, abnormality_idx, class_idx]
    positive_count = int(y_true.sum())
    negative_count = int(len(y_true) - positive_count)
    if positive_count == 0 or negative_count == 0:
        return None
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    finite = np.isfinite(thresholds)
    if finite.any():
        fpr_f = fpr[finite]
        tpr_f = tpr[finite]
        thresholds_f = thresholds[finite]
        best_idx = int(np.argmax(tpr_f - fpr_f))
        best_threshold = float(thresholds_f[best_idx])
        youden_j = float(tpr_f[best_idx] - fpr_f[best_idx])
    else:
        best_threshold = float("nan")
        youden_j = float("nan")
    return {
        "auc": float(auc(fpr, tpr)),
        "optimal_threshold": best_threshold,
        "youden_j": youden_j,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "n": positive_count + negative_count,
    }


def compute_thresholds(probs: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    rows = []
    for abn_idx, abnormality in enumerate(ABNORMALITIES):
        for class_name, class_idx in CLASS_MAP.items():
            result = binary_roc_for_class(labels, probs, abn_idx, class_idx)
            if result is None:
                continue
            rows.append({"abnormality": abnormality, "class": class_name, **result})
    return pd.DataFrame(rows)


def save_tables(df: pd.DataFrame, args: argparse.Namespace, out_dir: Path) -> dict:
    stem = f"figure11_{args.run_name}_{args.split}_classification_thresholds"
    csv_path = out_dir / f"{stem}.csv"
    json_path = out_dir / f"{stem}.json"
    pivot_path = out_dir / f"{stem}_pivot.csv"
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)
    pivot = df.pivot(index="abnormality", columns="class", values="optimal_threshold")
    pivot = pivot.reindex(ABNORMALITIES)
    pivot.to_csv(pivot_path)
    return {"csv": str(csv_path), "json": str(json_path), "pivot_csv": str(pivot_path)}


def plot_thresholds(df: pd.DataFrame, args: argparse.Namespace, out_dir: Path) -> dict:
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 5.8), dpi=220, sharey=True)
    fig.suptitle("Optimal Thresholds for Each Class and Abnormality", fontsize=14, y=0.96)
    color = "#4C93C3"
    max_threshold = float(df["optimal_threshold"].max())
    y_top = min(1.0, max(0.62, np.ceil((max_threshold + 0.05) * 10) / 10))

    for ax, (class_name, title) in zip(axes, CLASS_PLOT_ORDER):
        part = df[df["class"] == class_name].copy()
        part["abnormality"] = pd.Categorical(part["abnormality"], categories=ABNORMALITIES, ordered=True)
        part = part.sort_values("abnormality")
        names = part["abnormality"].astype(str).tolist()
        values = part["optimal_threshold"].astype(float).tolist()
        ax.bar(names, values, color=color, edgecolor="white", linewidth=0.8)
        ax.set_title(f"Optimal Thresholds: {title}", fontsize=11)
        ax.set_xlabel("Abnormality", fontsize=9)
        ax.set_ylim(0, y_top)
        ax.grid(axis="y", alpha=0.35)
        ax.tick_params(axis="x", labelrotation=50, labelsize=7)
        ax.tick_params(axis="y", labelsize=8)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")

    axes[0].set_ylabel("Threshold", fontsize=9)
    fig.text(0.055, 0.035, "FIGURE 11.", fontsize=11, fontweight="bold", color="#0072BC")
    fig.text(0.135, 0.035, "Classification Thresholds.", fontsize=11, fontweight="bold", color="black")
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))

    png_path = out_dir / f"figure11_{args.run_name}_{args.split}_classification_thresholds.png"
    pdf_path = out_dir / f"figure11_{args.run_name}_{args.split}_classification_thresholds.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {"figure_png": str(png_path), "figure_pdf": str(pdf_path)}


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    probs, labels = collect_probs_labels(args, out_dir)
    df = compute_thresholds(probs, labels)
    table_outputs = save_tables(df, args, out_dir)
    figure_outputs = plot_thresholds(df, args, out_dir)
    summary = {
        "run_name": args.run_name,
        "split": args.split,
        "sample_limit": args.sample_limit,
        "mode": "image_only" if args.image_only else "with_report_text",
        "n_samples": int(labels.shape[0]),
        **table_outputs,
        **figure_outputs,
    }
    summary_path = out_dir / f"figure11_{args.run_name}_{args.split}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not args.no_upload:
        upload_dir(out_dir, args.gcs_output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
