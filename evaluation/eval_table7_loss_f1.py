#!/usr/bin/env python3
"""Compute Table 7-style Mean F1 from an existing 3-encoder checkpoint.

This follows the F1 convention used by META_CXR_encoder_f1_table_kaggle.ipynb:
weighted multiclass F1 over labels [negative, positive, uncertain] per
pathology, then the mean over the 5 common pathologies.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "model"))

import model.lavis.tasks as tasks  # noqa: E402
from local_config import VIS_ROOT  # noqa: E402
from model.lavis.common.config import Config  # noqa: E402
from model.lavis.common.registry import registry  # noqa: E402
from model.lavis.data.ReportDataset import MIMIC_CXR_Dataset  # noqa: E402

# Registration imports.
from model.lavis.common.optims import (  # noqa: E402,F401
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from model.lavis.datasets.builders import *  # noqa: E402,F403
from model.lavis.models import *  # noqa: E402,F403
from model.lavis.processors import *  # noqa: E402,F403
from model.lavis.tasks import *  # noqa: E402,F403

registry.mapping["paths"]["cache_root"] = "."

CHEXPERT_COLS = [
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

FIVE_COMMON_ABNORMALITIES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]
TASK_IDXS = [CHEXPERT_COLS.index(name) for name in FIVE_COMMON_ABNORMALITIES]
LABEL_NAMES = {0: "negative", 1: "positive", 2: "uncertain"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Table 7 loss-ablation Mean F1 eval")
    parser.add_argument("--run-name", default="07_all_three")
    parser.add_argument(
        "--cfg-path",
        type=Path,
        default=PROJECT_DIR / "pretraining/configs/encoder_comparison/07_all_three.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/mnt/meta-cxr-checkpoint/07_all_three/checkpoint_best.pth"),
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--f1-mode",
        choices=["weighted_multiclass", "binary_positive"],
        default="weighted_multiclass",
        help=(
            "weighted_multiclass matches the existing encoder F1 notebook; "
            "binary_positive is P-vs-rest for clinical efficacy checks."
        ),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=PROJECT_DIR / "outputs/tables/table_7_loss_ablation/07_all_three_mean_f1.json",
    )
    return parser.parse_args()


def build_cfg(cfg_path: Path) -> Config:
    return Config(SimpleNamespace(cfg_path=str(cfg_path), options=None))


def build_model(cfg: Config, checkpoint_path: Path) -> torch.nn.Module:
    task = tasks.setup_task(cfg)
    model = task.build_model(cfg)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(
        f"[load] {checkpoint_path} missing={len(missing)} unexpected={len(unexpected)}"
    )
    return model


def make_loader(cfg: Config, split: str, batch_size: int, num_workers: int) -> DataLoader:
    dataset = MIMIC_CXR_Dataset(
        vis_processor=None,
        text_processor=None,
        vis_root=VIS_ROOT,
        split=split,
        cfg=cfg,
        truncate=None,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def predict_logits_with_text(model: torch.nn.Module, batch: dict, device: str) -> torch.Tensor:
    image = batch["image"].to(device, non_blocking=True)
    text = batch["text_output"]

    cnn_patches, vit_patches, swin_patches, _ = model._encode_image_streams(
        image, apply_aug=False
    )
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


def task_f1(y_true: np.ndarray, y_pred: np.ndarray, mode: str) -> float:
    if mode == "weighted_multiclass":
        return float(
            f1_score(y_true, y_pred, average="weighted", zero_division=1)
        )
    y_true_bin = (y_true == 1).astype(np.int32)
    y_pred_bin = (y_pred == 1).astype(np.int32)
    return float(f1_score(y_true_bin, y_pred_bin, zero_division=0))


def task_pr(y_true: np.ndarray, y_pred: np.ndarray, mode: str) -> tuple[float, float]:
    if mode == "weighted_multiclass":
        precision = precision_score(y_true, y_pred, average="weighted", zero_division=1)
        recall = recall_score(y_true, y_pred, average="weighted", zero_division=1)
        return float(precision), float(recall)
    y_true_bin = (y_true == 1).astype(np.int32)
    y_pred_bin = (y_pred == 1).astype(np.int32)
    precision = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    recall = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    return float(precision), float(recall)


def label_counts(values: np.ndarray) -> dict[str, int]:
    return {
        LABEL_NAMES[label]: int(np.sum(values == label))
        for label in sorted(LABEL_NAMES)
    }


def main() -> None:
    args = parse_args()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")
    print(f"[cfg]    {args.cfg_path}")
    print(f"[ckpt]   {args.checkpoint}")
    print(f"[mode]   {args.f1_mode}")

    cfg = build_cfg(args.cfg_path)
    model = build_model(cfg, args.checkpoint).to(device).eval()
    loader = make_loader(cfg, args.split, args.batch_size, args.num_workers)

    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    for batch in tqdm(loader, desc=args.run_name):
        with torch.cuda.amp.autocast(enabled=device == "cuda"):
            logits = predict_logits_with_text(model, batch, device)
        preds = torch.argmax(torch.softmax(logits.float(), dim=-1), dim=-1)
        all_preds.append(preds[:, TASK_IDXS].cpu().numpy())
        all_labels.append(batch["classification_labels"][:, TASK_IDXS].cpu().numpy())

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_labels, axis=0)

    per_task = {}
    for local_idx, task_name in enumerate(FIVE_COMMON_ABNORMALITIES):
        yt = y_true[:, local_idx]
        yp = y_pred[:, local_idx]
        precision, recall = task_pr(yt, yp, args.f1_mode)
        per_task[task_name] = {
            "f1": round(task_f1(yt, yp, args.f1_mode), 6),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "label_counts": label_counts(yt),
            "pred_counts": label_counts(yp),
        }

    mean_f1 = float(np.mean([v["f1"] for v in per_task.values()]))
    payload = {
        "table": "Table 7",
        "run": args.run_name,
        "checkpoint": str(args.checkpoint),
        "cfg_path": str(args.cfg_path),
        "split": args.split,
        "num_samples": int(y_true.shape[0]),
        "f1_mode": args.f1_mode,
        "f1_definition": (
            "For each of the 5 common pathologies, compute weighted multiclass "
            "F1 over labels 0=negative, 1=positive, 2=uncertain; then average "
            "the 5 F1 values."
            if args.f1_mode == "weighted_multiclass"
            else "Positive-vs-rest F1 for each common pathology, then average the 5 F1 values."
        ),
        "loss_functions": {"CLS": True, "OL": True, "SL": True, "CL": True},
        "common_pathologies": FIVE_COMMON_ABNORMALITIES,
        "per_pathology": per_task,
        "mean_f1_score": round(mean_f1, 6),
        "table_row": {
            "CLS": "+",
            "OL": "+",
            "SL": "+",
            "CL": "+",
            "Mean F1 Score": round(mean_f1, 3),
        },
    }

    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["table_row"], indent=2))
    print(f"[saved] {args.out_json}")

    del model, loader
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
