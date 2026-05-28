#!/usr/bin/env python3
"""Evaluate encoder masking from the 07_all_three checkpoint.

This script keeps the same trained model and expert tokens from 07_all_three,
then toggles encoder streams at inference by passing None for disabled streams
inside MHCAC.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

import model.lavis.tasks as tasks
from model.lavis.common.config import Config
from model.lavis.common.registry import registry

# Registration imports required by LAVIS registry.
from model.lavis.common.optims import LinearWarmupCosineLRScheduler, LinearWarmupStepLRScheduler  # noqa: F401
from model.lavis.datasets.builders import *  # noqa: F401,F403
from model.lavis.models import *  # noqa: F401,F403
from model.lavis.processors import *  # noqa: F401,F403
from model.lavis.tasks import *  # noqa: F401,F403
from model.lavis.data.ReportDataset import MIMIC_CXR_Dataset
from local_config import VIS_ROOT


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

TOGGLE_RUNS = [
    {"run": "07mask_biovil_only", "RN50": True, "ViT": False, "Swin": False},
    {"run": "07mask_pubmedclip_only", "RN50": False, "ViT": True, "Swin": False},
    {"run": "07mask_swin_only", "RN50": False, "ViT": False, "Swin": True},
    {"run": "07mask_biovil_pubmedclip", "RN50": True, "ViT": True, "Swin": False},
    {"run": "07mask_biovil_swin", "RN50": True, "ViT": False, "Swin": True},
    {"run": "07mask_pubmedclip_swin", "RN50": False, "ViT": True, "Swin": True},
    {"run": "07mask_all_three", "RN50": True, "ViT": True, "Swin": True},
]

PAPER_F1 = {
    "07mask_biovil_only": 0.602,
    "07mask_pubmedclip_only": 0.473,
    "07mask_swin_only": 0.467,
    "07mask_biovil_pubmedclip": 0.631,
    "07mask_biovil_swin": 0.682,
    "07mask_pubmedclip_swin": None,
    "07mask_all_three": 0.701,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="META-CXR repo root.",
    )
    parser.add_argument(
        "--cfg",
        type=Path,
        default=Path("pretraining/configs/encoder_comparison/07_all_three.yaml"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/mnt/meta-cxr-checkpoint/07_all_three/checkpoint_best.pth"),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output/encoder_toggle_07"),
    )
    return parser.parse_args()


def load_torch_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def build_cfg(cfg_path: Path) -> Config:
    args = SimpleNamespace(cfg_path=str(cfg_path), options=None)
    return Config(args)


def build_model(cfg: Config, checkpoint_path: Path, device: str) -> torch.nn.Module:
    task = tasks.setup_task(cfg)
    model = task.build_model(cfg)
    ckpt = load_torch_checkpoint(checkpoint_path)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(
        f"Loaded {checkpoint_path}; missing={len(missing)}, unexpected={len(unexpected)}"
    )
    if missing:
        print("First missing keys:", missing[:10])
    if unexpected:
        print("First unexpected keys:", unexpected[:10])
    model.to(device)
    model.eval()
    return model


def make_test_loader(cfg: Config, batch_size: int, num_workers: int, limit: int | None):
    dataset = MIMIC_CXR_Dataset(
        vis_processor=None,
        text_processor=None,
        vis_root=VIS_ROOT,
        split="test",
        cfg=cfg,
        truncate=None,
    )
    if limit is not None:
        dataset = Subset(dataset, list(range(min(limit, len(dataset)))))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def selected_streams(
    cnn_patches: torch.Tensor | None,
    vit_patches: torch.Tensor | None,
    swin_patches: torch.Tensor | None,
    item: dict,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    return (
        cnn_patches if item["RN50"] else None,
        vit_patches if item["ViT"] else None,
        swin_patches if item["Swin"] else None,
    )


def weighted_multiclass_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="weighted", zero_division=1))


def positive_binary_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true_bin = (y_true == 1).astype(np.int32)
    y_pred_bin = (y_pred == 1).astype(np.int32)
    return float(f1_score(y_true_bin, y_pred_bin, zero_division=0))


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: str) -> tuple[pd.DataFrame, dict]:
    preds_by_run = {item["run"]: [] for item in TOGGLE_RUNS}
    labels_all = []

    for batch in tqdm(loader, desc="07_all_three toggle eval"):
        image = batch["image"].to(device, non_blocking=True)
        cnn_patches, vit_patches, swin_patches, _ = model._encode_image_streams(
            image, apply_aug=False
        )

        labels_all.append(batch["classification_labels"][:, TASK_IDXS].cpu().numpy())

        for item in TOGGLE_RUNS:
            selected_cnn, selected_vit, selected_swin = selected_streams(
                cnn_patches, vit_patches, swin_patches, item
            )
            logits, _, _, _, _ = model.mhcac(
                cnn_patches=selected_cnn,
                vit_patches=selected_vit,
                swin_patches=selected_swin,
                text_embeddings=None,
                labels=None,
            )
            pred = torch.argmax(torch.softmax(logits, dim=-1), dim=-1)
            preds_by_run[item["run"]].append(pred[:, TASK_IDXS].cpu().numpy())

    y_true = np.concatenate(labels_all, axis=0)
    rows = []
    details = {}

    for item in TOGGLE_RUNS:
        run_name = item["run"]
        y_pred = np.concatenate(preds_by_run[run_name], axis=0)
        per_task_weighted = {}
        per_task_positive = {}
        for col_idx, task_name in enumerate(FIVE_COMMON_ABNORMALITIES):
            per_task_weighted[task_name] = weighted_multiclass_f1(
                y_true[:, col_idx], y_pred[:, col_idx]
            )
            per_task_positive[task_name] = positive_binary_f1(
                y_true[:, col_idx], y_pred[:, col_idx]
            )

        mean_weighted = float(np.mean(list(per_task_weighted.values())))
        mean_positive = float(np.mean(list(per_task_positive.values())))
        paper = PAPER_F1[run_name]
        rows.append(
            {
                "RN50": "yes" if item["RN50"] else "no",
                "ViT": "yes" if item["ViT"] else "no",
                "Swin": "yes" if item["Swin"] else "no",
                "Mean F1 Score": round(mean_weighted, 4),
                "Mean F1 P-vs-rest": round(mean_positive, 4),
                "Paper F1": paper if paper is not None else None,
                "Delta vs Paper": round(mean_weighted - paper, 4)
                if paper is not None
                else None,
            }
        )
        details[run_name] = {
            "mean_f1_weighted_multiclass": mean_weighted,
            "mean_f1_positive_binary": mean_positive,
            "per_task_weighted_multiclass": per_task_weighted,
            "per_task_positive_binary": per_task_positive,
        }

    return pd.DataFrame(rows), details


def main() -> None:
    args = parse_args()
    project_dir = args.project_dir.resolve()
    cfg_path = args.cfg if args.cfg.is_absolute() else project_dir / args.cfg
    checkpoint_path = (
        args.checkpoint if args.checkpoint.is_absolute() else project_dir / args.checkpoint
    )
    out_dir = args.out_dir if args.out_dir.is_absolute() else project_dir / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    registry.mapping["paths"]["cache_root"] = "."
    print("project_dir =", project_dir)
    print("cfg_path    =", cfg_path)
    print("checkpoint  =", checkpoint_path)
    print("device      =", args.device)
    print("limit       =", args.limit)

    cfg = build_cfg(cfg_path)
    model = build_model(cfg, checkpoint_path, args.device)
    loader = make_test_loader(cfg, args.batch_size, args.num_workers, args.limit)
    print("test samples =", len(loader.dataset))

    table, details = evaluate(model, loader, args.device)
    table_path = out_dir / "encoder_toggle_07_table.csv"
    json_path = out_dir / "encoder_toggle_07_details.json"
    table.to_csv(table_path, index=False)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": str(checkpoint_path),
                "cfg": str(cfg_path),
                "limit": args.limit,
                "num_samples": len(loader.dataset),
                "metric_primary": "weighted multiclass F1, average='weighted', zero_division=1",
                "metric_secondary": "positive-vs-rest binary F1, zero_division=0",
                "details": details,
            },
            f,
            indent=2,
        )

    print("\nTable:")
    print(table.to_string(index=False))
    print("\nWrote:", table_path)
    print("Wrote:", json_path)

    del model, loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
