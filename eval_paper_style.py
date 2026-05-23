"""
Paper-style evaluation for META-CXR classification head.

Reproduces Table 5 methodology:
  - Binary P-vs-rest (P=1, N+U=0) per abnormality
  - Mean F1 over the 5 common abnormalities:
      Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion
  - macro-F1 with zero_division=0 (paper-faithful, NOT the inflated
    weighted+zero_division=1 used in mhcac/utils.py:181 during training)

Usage:
  1. Pull checkpoint from GCS first:
        gsutil cp gs://meta-cxr-checkpoint/01_biovil_only/checkpoint_best.pth \\
            pretraining/output/01_biovil_only/checkpoint_best.pth

  2. Run:
        python eval_paper_style.py \\
            --cfg-path pretraining/configs/encoder_comparison/01_biovil_only.yaml \\
            --checkpoint pretraining/output/01_biovil_only/checkpoint_best.pth \\
            --split test \\
            --batch-size 32 \\
            --out-json eval_results/01_biovil_only_test.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.lavis.common.config import Config  # noqa: E402
from model.lavis.data.ReportDataset import MIMIC_CXR_Dataset  # noqa: E402
from model.lavis.tasks import setup_task  # noqa: E402


# 14 abnormalities, in the exact order produced by the classification head
# (matches ReportDataset.chexpert_cols).
ABNORMALITIES_14: tuple[str, ...] = (
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
)

# The "5 common abnormalities" reported in paper Table 5.
COMMON_5: tuple[str, ...] = (
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
)

# Label encoding used during training:
#   0 = Negative, 1 = Positive, 2 = Uncertain  (see ReportDataset.py:254)
POSITIVE_CLASS = 1


@dataclass(frozen=True)
class EvalConfig:
    cfg_path: Path
    checkpoint: Path
    split: str
    batch_size: int
    num_workers: int
    out_json: Path
    vis_root: str | None


def parse_args() -> EvalConfig:
    p = argparse.ArgumentParser(description="Paper-style eval for META-CXR")
    p.add_argument("--cfg-path", type=Path, required=True,
                   help="YAML used during training (run_cfg + model_cfg)")
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Local path to checkpoint_*.pth (pull from GCS first)")
    p.add_argument("--split", choices=["test", "val"], default="test")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--vis-root", type=str, default=None,
                   help="Override images root if cfg's vis_root is wrong on this machine")
    a = p.parse_args()
    return EvalConfig(
        cfg_path=a.cfg_path,
        checkpoint=a.checkpoint,
        split=a.split,
        batch_size=a.batch_size,
        num_workers=a.num_workers,
        out_json=a.out_json,
        vis_root=a.vis_root,
    )


def build_cfg(cfg_path: Path) -> Any:
    """Load training YAML via LAVIS Config so model builder finds everything it needs."""
    return Config(argparse.Namespace(cfg_path=str(cfg_path), options=None))


def load_model(cfg: Any, checkpoint_path: Path) -> torch.nn.Module:
    task = setup_task(cfg)
    model = task.build_model(cfg)

    state = torch.load(checkpoint_path, map_location="cpu")
    sd = state.get("model", state) if isinstance(state, dict) else state
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[load] missing keys: {len(missing)} (first 5: {missing[:5]})")
    if unexpected:
        print(f"[load] unexpected keys: {len(unexpected)} (first 5: {unexpected[:5]})")

    model.cuda().eval()
    return model


def build_loader(cfg: Any, split: str, batch_size: int, num_workers: int,
                 vis_root_override: str | None) -> DataLoader:
    vis_root = vis_root_override or cfg.config.datasets.mimic_cxr.build_info.images.storage
    dataset = MIMIC_CXR_Dataset(
        vis_processor=None,
        text_processor=None,
        vis_root=vis_root,
        split=split,
        cfg=cfg.config,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


@torch.no_grad()
def run_inference(model: torch.nn.Module, loader: DataLoader
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        logits3:  (N, 14, 3)   raw logits over [N, P, U]
        probs_p:  (N, 14)      softmax probability of Positive class (for AUC)
        labels:  (N, 14)       ground-truth in {0,1,2}
    """
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for batch in tqdm(loader, desc="inference"):
        images = batch["image"].cuda(non_blocking=True)
        _, _, cls_logits, _ = model.forward_image(images, None)
        all_logits.append(cls_logits.float().cpu().numpy())
        all_labels.append(batch["classification_labels"].cpu().numpy())

    logits3 = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    probs = _softmax_last(logits3)
    probs_p = probs[..., POSITIVE_CLASS]
    return logits3, probs_p, labels


def _softmax_last(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=-1, keepdims=True)


def compute_metrics(logits3: np.ndarray, probs_p: np.ndarray,
                    labels: np.ndarray) -> dict[str, Any]:
    """
    Paper-faithful evaluation:
      - 3-way argmax -> predicted class.
      - Binarize: y_true = (label == P), y_pred = (pred == P).
      - F1, precision, recall per abnormality, then mean over (a) all 14,
        and (b) the 5 common abnormalities reported in Table 5.
      - AUROC uses Positive-class probability.
    """
    preds3 = logits3.argmax(axis=-1)

    per_class: dict[str, dict[str, float]] = {}
    for j, name in enumerate(ABNORMALITIES_14):
        y_true_bin = (labels[:, j] == POSITIVE_CLASS).astype(np.int32)
        y_pred_bin = (preds3[:, j] == POSITIVE_CLASS).astype(np.int32)

        f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
        prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
        rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)

        if 0 < y_true_bin.sum() < len(y_true_bin):
            auroc = float(roc_auc_score(y_true_bin, probs_p[:, j]))
        else:
            auroc = float("nan")

        per_class[name] = {
            "support_positive": int(y_true_bin.sum()),
            "support_total": int(len(y_true_bin)),
            "f1": float(f1),
            "precision": float(prec),
            "recall": float(rec),
            "auroc": auroc,
        }

    mean_f1_14 = float(np.mean([per_class[n]["f1"] for n in ABNORMALITIES_14]))
    mean_f1_5 = float(np.mean([per_class[n]["f1"] for n in COMMON_5]))
    mean_auroc_5 = float(np.nanmean([per_class[n]["auroc"] for n in COMMON_5]))

    return {
        "per_class": per_class,
        "mean_f1_over_14": mean_f1_14,
        "mean_f1_over_5_common": mean_f1_5,
        "mean_auroc_over_5_common": mean_auroc_5,
        "common_5": list(COMMON_5),
    }


def print_report(metrics: dict[str, Any], split: str, ckpt: Path) -> None:
    print()
    print("=" * 72)
    print(f"  Paper-style eval | split={split} | ckpt={ckpt.name}")
    print("=" * 72)
    print()
    print(f"{'Abnormality':<32} {'F1':>7} {'Prec':>7} {'Rec':>7} {'AUROC':>7} {'#Pos':>7}")
    print("-" * 72)
    for name in ABNORMALITIES_14:
        m = metrics["per_class"][name]
        marker = " *" if name in COMMON_5 else "  "
        print(f"{marker}{name:<30} {m['f1']:>7.3f} {m['precision']:>7.3f} "
              f"{m['recall']:>7.3f} {m['auroc']:>7.3f} {m['support_positive']:>7d}")
    print("-" * 72)
    print(f"  Mean F1 over all 14            : {metrics['mean_f1_over_14']:.3f}")
    print(f"  Mean F1 over 5 common (Table 5): {metrics['mean_f1_over_5_common']:.3f}")
    print(f"  Mean AUROC over 5 common       : {metrics['mean_auroc_over_5_common']:.3f}")
    print()
    print("Compare against paper Table 5 (MIMIC-CXR test set):")
    print("    RN50 only          : 0.602")
    print("    ViT only           : 0.473")
    print("    Swin only          : 0.467")
    print("    RN50 + ViT         : 0.631")
    print("    RN50 + Swin        : 0.682")
    print("    RN50 + ViT + Swin  : 0.701  (best)")
    print()


def main() -> None:
    args = parse_args()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    print(f"[cfg]     {args.cfg_path}")
    print(f"[ckpt]    {args.checkpoint}")
    print(f"[split]   {args.split}")

    cfg = build_cfg(args.cfg_path)
    model = load_model(cfg, args.checkpoint)
    loader = build_loader(cfg, args.split, args.batch_size,
                          args.num_workers, args.vis_root)

    logits3, probs_p, labels = run_inference(model, loader)
    print(f"[inference] logits shape: {logits3.shape}, labels shape: {labels.shape}")

    metrics = compute_metrics(logits3, probs_p, labels)

    payload = {
        "cfg_path": str(args.cfg_path),
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "num_samples": int(labels.shape[0]),
        "metrics": metrics,
    }
    args.out_json.write_text(json.dumps(payload, indent=2))
    print(f"[saved]   {args.out_json}")

    print_report(metrics, args.split, args.checkpoint)


if __name__ == "__main__":
    main()
