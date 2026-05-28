#!/usr/bin/env python3
"""Clinical Efficacy: Precision, Recall, Macro F1 for 07_all_three classification head.

Binary P-vs-rest per abnormality, following paper Table 5 methodology.
"""

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

from model.lavis.common.config import Config
from model.lavis.common.registry import registry
from model.lavis.tasks import setup_task
from local_config import VIS_ROOT
from model.lavis.data.ReportDataset import MIMIC_CXR_Dataset

registry.mapping["paths"]["cache_root"] = "."

ABNORMALITIES_14 = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices",
]
COMMON_5 = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_WORKERS = 4


def build_cfg_simple(run_name):
    cfg_path = PROJECT_DIR / "pretraining" / "configs" / "encoder_comparison" / f"{run_name}.yaml"
    return Config(SimpleNamespace(cfg_path=str(cfg_path), options=None))


def load_model(run_name, checkpoint_path):
    cfg = build_cfg_simple(run_name)
    task = setup_task(cfg)
    model = task.build_model(cfg)
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(sd, strict=False)
    model.to(DEVICE)
    model.eval()
    return cfg, model


def make_loader(cfg):
    dataset = MIMIC_CXR_Dataset(
        vis_processor=None, text_processor=None,
        vis_root=VIS_ROOT, split="test", cfg=cfg, truncate=None,
    )
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=NUM_WORKERS, pin_memory=True)


@torch.no_grad()
def run_inference(model, loader):
    all_logits, all_labels = [], []
    for batch in tqdm(loader, desc="Infer"):
        image = batch["image"].to(DEVICE, non_blocking=True)
        labels = batch["classification_labels"].float().cpu().numpy()
        cls_logits, _ = model.forward_image(image)
        all_logits.append(cls_logits.float().cpu().numpy())
        all_labels.append(labels)
    return np.concatenate(all_logits, axis=0), np.concatenate(all_labels, axis=0)


def compute_metrics(logits, labels):
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    preds = np.argmax(probs, axis=-1)

    per_abn = {}
    f1s, precs, recs = [], [], []

    for i, abn in enumerate(ABNORMALITIES_14):
        if abn == "No Finding":
            continue
        yt = (labels[:, i] == 1).astype(int)
        yp = (preds[:, i] == 1).astype(int)
        p = precision_score(yt, yp, zero_division=0)
        r = recall_score(yt, yp, zero_division=0)
        f = f1_score(yt, yp, zero_division=0)
        per_abn[abn] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}
        if abn in COMMON_5:
            f1s.append(f); precs.append(p); recs.append(r)

    return {
        "per_abnormality": per_abn,
        "mean_f1_over_5_common": round(float(np.mean(f1s)), 4),
        "mean_precision_over_5_common": round(float(np.mean(precs)), 4),
        "mean_recall_over_5_common": round(float(np.mean(recs)), 4),
    }, f1s, precs, recs


def main():
    run_name = "07_all_three"
    ckpt = Path("/mnt/meta-cxr-checkpoint") / run_name / "checkpoint_best.pth"
    print(f"[run] {run_name}  [ckpt] {ckpt}")

    cfg, model = load_model(run_name, ckpt)
    loader = make_loader(cfg)
    logits, labels = run_inference(model, loader)
    print(f"[data] {logits.shape[0]} samples")

    metrics, f1s, precs, recs = compute_metrics(logits, labels)

    print(f"\n{'='*62}")
    print(f"Clinical Efficacy — {run_name} (ResNet50 + ViT + Swin)")
    print(f"{'='*62}")
    print(f"{'Abnormality':<30} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 62)
    for abn in ABNORMALITIES_14:
        if abn == "No Finding":
            continue
        m = metrics["per_abnormality"][abn]
        star = " *" if abn in COMMON_5 else ""
        print(f"{abn + star:<30} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")
    print("-" * 62)
    print(f"{'Mean over 5 common':<30} {metrics['mean_precision_over_5_common']:>10.4f} {metrics['mean_recall_over_5_common']:>10.4f} {metrics['mean_f1_over_5_common']:>10.4f}")

    out = Path("/home/phuong/META-CXR/output/nlg_metrics/classification_07_all_three.json")
    with open(out, "w") as f:
        json.dump({"run": run_name, "num_samples": int(labels.shape[0]), **metrics}, f, indent=2)
    print(f"\n[Saved] {out}")

    os.system(f"gsutil -m cp {out} gs://meta-cxr-checkpoint/eval/classification_07_all_three.json")
    print("[Uploaded]")

    print(f"\n>>> Summary: Precision={metrics['mean_precision_over_5_common']:.4f}  "
          f"Recall={metrics['mean_recall_over_5_common']:.4f}  "
          f"Macro F1={metrics['mean_f1_over_5_common']:.4f}")


if __name__ == "__main__":
    main()
