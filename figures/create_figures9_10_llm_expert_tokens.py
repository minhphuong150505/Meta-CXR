#!/usr/bin/env python3
"""Create paper Figure 9 and Figure 10 from saved META-CXR evaluations.

Figure 9 compares report-generation metrics for Vicuna+LoRA and
MedGemma1.5+LoRA on the 07_all_three checkpoint.

Figure 10 evaluates MHCAC classification metrics while truncating the trained
common expert-token bank to K tokens. This is a checkpoint ablation, not a
separately retrained 4/6/8/10/12-token experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "model"))

from local_config import VIS_ROOT  # noqa: E402
from model.lavis.common.config import Config  # noqa: E402
from model.lavis.common.registry import registry  # noqa: E402
from model.lavis.data.ReportDataset import MIMIC_CXR_Dataset  # noqa: E402
from model.lavis.tasks import setup_task  # noqa: E402


registry.mapping["paths"]["cache_root"] = "."

ABNORMALITIES_14 = [
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
COMMON_5 = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
COMMON_5_IDX = [ABNORMALITIES_14.index(name) for name in COMMON_5]

FIG9_METRICS = ["BERTScore", "BLEU-4", "ROUGE-L", "METEOR", "CIDEr"]
LLM_TABLES = {
    "Vicuna + LoRA": "gs://meta-cxr-checkpoint/eval/nlg_metrics_table.csv",
    "MedGemma1.5 + LoRA": "gs://meta-cxr-checkpoint/eval/MedGemma_QFormer/metrics/nlg_metrics_table_medgemma_qformer.csv",
}


def run_cmd(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(args, check=check, text=True, capture_output=True)


def gcs_cp(src: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    run_cmd(["gcloud", "storage", "cp", src, str(dst)])


def gcs_upload(local: Path, gcs_dir: str) -> None:
    run_cmd(["gcloud", "storage", "cp", str(local), gcs_dir.rstrip("/") + "/"])


def read_gcs_csv(gcs_path: str, work_dir: Path) -> pd.DataFrame:
    local = work_dir / Path(gcs_path).name
    gcs_cp(gcs_path, local)
    return pd.read_csv(local)


def build_figure9(work_dir: Path, output_dir: Path) -> dict:
    rows = []
    for llm_name, gcs_path in LLM_TABLES.items():
        df = read_gcs_csv(gcs_path, work_dir)
        match = df[df["Run"] == "07_all_three"]
        if match.empty:
            raise ValueError(f"{gcs_path} does not contain Run=07_all_three")
        row = match.iloc[0]
        item = {"LLM": llm_name, "Run": "07_all_three"}
        for metric in FIG9_METRICS:
            item[metric] = float(row[metric])
        rows.append(item)

    metrics_df = pd.DataFrame(rows)
    metrics_csv = output_dir / "figure9_llm_comparison_metrics.csv"
    metrics_json = output_dir / "figure9_llm_comparison_metrics.json"
    metrics_df.to_csv(metrics_csv, index=False)
    metrics_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    labels = FIG9_METRICS
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#555555",
            "axes.linewidth": 0.8,
        }
    )
    fig = plt.figure(figsize=(7.2, 6.6), dpi=220)
    ax = fig.add_subplot(111, polar=True)
    colors = {
        "Vicuna + LoRA": "#2563eb",
        "MedGemma1.5 + LoRA": "#dc2626",
    }
    linestyles = {
        "Vicuna + LoRA": "--",
        "MedGemma1.5 + LoRA": "-",
    }

    for _, row in metrics_df.iterrows():
        values = [float(row[label]) for label in labels]
        values += values[:1]
        ax.plot(
            angles,
            values,
            label=row["LLM"],
            color=colors.get(row["LLM"], None),
            linestyle=linestyles.get(row["LLM"], "-"),
            linewidth=2.2,
            marker="o",
            markersize=3.5,
        )
        ax.fill(angles, values, color=colors.get(row["LLM"], "#888888"), alpha=0.06)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    max_value = max(float(metrics_df[m].max()) for m in labels)
    rmax = max(0.1, math.ceil((max_value + 0.04) * 10) / 10)
    ax.set_ylim(0, rmax)
    yticks = np.linspace(0, rmax, 6)[1:]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{v:.1f}" for v in yticks], fontsize=8, color="#555555")
    ax.grid(color="#b7b7b7", linewidth=0.65, alpha=0.85)
    ax.set_title("LLM Comparison", fontsize=13, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.27, 1.13), frameon=True, fontsize=9)

    caption = "Figure 9. Effect of Tuned LLMs."
    fig.text(0.08, 0.035, caption, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0.03, 0.07, 0.96, 0.96))

    png = output_dir / "figure9_llm_comparison_vicuna_medgemma.png"
    pdf = output_dir / "figure9_llm_comparison_vicuna_medgemma.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    return {
        "metrics_csv": str(metrics_csv),
        "metrics_json": str(metrics_json),
        "figure_png": str(png),
        "figure_pdf": str(pdf),
        "definition": "Raw NLG metric values for 07_all_three, sample_limit=200 from saved eval tables.",
        "rows": rows,
    }


def build_cfg(run_name: str) -> Config:
    cfg_path = PROJECT_DIR / "pretraining" / "configs" / "encoder_comparison" / f"{run_name}.yaml"
    return Config(SimpleNamespace(cfg_path=str(cfg_path), options=None))


def load_torch_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def filter_state_dict_for_model(model: torch.nn.Module, state_dict: dict) -> tuple[dict, list]:
    model_state = model.state_dict()
    filtered = {}
    mismatched = []
    for key, value in state_dict.items():
        if key in model_state and hasattr(value, "shape"):
            ckpt_shape = tuple(value.shape)
            model_shape = tuple(model_state[key].shape)
            if ckpt_shape != model_shape:
                mismatched.append((key, ckpt_shape, model_shape))
                continue
        filtered[key] = value
    return filtered, mismatched


def load_stage1_model(run_name: str, checkpoint_path: Path, device: torch.device):
    cfg = build_cfg(run_name)
    task = setup_task(cfg)
    model = task.build_model(cfg)
    ckpt = load_torch_checkpoint(checkpoint_path)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    state_dict, mismatched = filter_state_dict_for_model(model, state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(
        f"[model] {run_name}: missing={len(missing)} unexpected={len(unexpected)} "
        f"mismatched_skipped={len(mismatched)}"
    )
    if mismatched:
        for key, ckpt_shape, model_shape in mismatched[:6]:
            print(f"  skipped shape mismatch: {key}: checkpoint={ckpt_shape}, model={model_shape}")
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return cfg, model


def make_loader(cfg: Config, sample_limit: int | None, batch_size: int, num_workers: int) -> DataLoader:
    dataset = MIMIC_CXR_Dataset(
        vis_processor=None,
        text_processor=None,
        vis_root=VIS_ROOT,
        split="test",
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


def binary_stats_update(stats: dict[int, dict[str, np.ndarray]], token_count: int, logits, labels) -> None:
    preds = torch.argmax(torch.softmax(logits, dim=-1), dim=-1).detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    for j, abn_idx in enumerate(COMMON_5_IDX):
        y_true = labels_np[:, abn_idx] == 1
        y_pred = preds[:, abn_idx] == 1
        stats[token_count]["tp"][j] += int(np.logical_and(y_true, y_pred).sum())
        stats[token_count]["fp"][j] += int(np.logical_and(~y_true, y_pred).sum())
        stats[token_count]["fn"][j] += int(np.logical_and(y_true, ~y_pred).sum())


def summarize_stats(stats: dict[int, dict[str, np.ndarray]], num_samples: int) -> pd.DataFrame:
    rows = []
    for token_count in sorted(stats):
        tp = stats[token_count]["tp"].astype(float)
        fp = stats[token_count]["fp"].astype(float)
        fn = stats[token_count]["fn"].astype(float)
        precision_per = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
        recall_per = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
        f1_per = np.divide(
            2 * precision_per * recall_per,
            precision_per + recall_per,
            out=np.zeros_like(precision_per),
            where=(precision_per + recall_per) > 0,
        )
        rows.append(
            {
                "Expert Tokens": token_count,
                "Precision": round(float(precision_per.mean()), 4),
                "Recall": round(float(recall_per.mean()), 4),
                "F1-Score": round(float(f1_per.mean()), 4),
                "N samples": int(num_samples),
                "Abnormalities": "|".join(COMMON_5),
            }
        )
    return pd.DataFrame(rows)


def rank_expert_tokens(
    model: torch.nn.Module,
    original_tokens: torch.Tensor,
    selection: str,
) -> torch.Tensor:
    num_tokens = original_tokens.shape[0]
    if selection == "first":
        return torch.arange(num_tokens, device=original_tokens.device)
    if selection == "last":
        return torch.arange(num_tokens - 1, -1, -1, device=original_tokens.device)
    if selection == "norm":
        scores = original_tokens.norm(dim=-1)
        return torch.argsort(scores, descending=True)
    if selection == "attention_pooling":
        query_vectors = model.mhcac.expert_loss.attention_pooling.query_vectors.detach()
        query_vectors = query_vectors.to(original_tokens.device)
        scores = torch.einsum("ad,nd->an", query_vectors, original_tokens)
        weights = torch.softmax(scores, dim=-1).mean(dim=0)
        return torch.argsort(weights, descending=True)
    raise ValueError(
        f"Unsupported token selection: {selection}. "
        "Use one of: attention_pooling, norm, first, last."
    )


def set_expert_tokens(
    model: torch.nn.Module,
    original_tokens: torch.Tensor,
    ranked_indices: torch.Tensor,
    token_count: int,
) -> None:
    selected = ranked_indices[:token_count]
    sliced = original_tokens[selected].detach().clone().to(original_tokens.device)
    model.mhcac.expert_tokens = torch.nn.Parameter(sliced, requires_grad=False)


def compute_figure10_metrics(args: argparse.Namespace, output_dir: Path) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    cfg, model = load_stage1_model(args.run_name, Path(args.checkpoint), device)
    loader = make_loader(cfg, args.sample_limit, args.batch_size, args.num_workers)

    token_counts = [int(x) for x in args.token_counts.split(",") if x.strip()]
    if not token_counts:
        raise ValueError("--token-counts is empty")

    original_tokens = model.mhcac.expert_tokens.detach().clone().to(device)
    if max(token_counts) > original_tokens.shape[0]:
        raise ValueError(
            f"requested token count {max(token_counts)} but checkpoint has only "
            f"{original_tokens.shape[0]} expert tokens"
        )
    ranked_indices = rank_expert_tokens(model, original_tokens, args.token_selection)

    stats = {
        k: {
            "tp": np.zeros(len(COMMON_5), dtype=np.int64),
            "fp": np.zeros(len(COMMON_5), dtype=np.int64),
            "fn": np.zeros(len(COMMON_5), dtype=np.int64),
        }
        for k in token_counts
    }
    num_samples = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Figure 10 expert-token ablation"):
            image = batch["image"].to(device, non_blocking=True)
            labels = batch["classification_labels"]
            cnn_patches, vit_patches, swin_patches, _ = model._encode_image_streams(
                image, apply_aug=False
            )
            num_samples += int(image.shape[0])
            for token_count in token_counts:
                set_expert_tokens(model, original_tokens, ranked_indices, token_count)
                logits, _, _, _, _ = model.mhcac(
                    cnn_patches=cnn_patches,
                    vit_patches=vit_patches,
                    swin_patches=swin_patches,
                    text_embeddings=None,
                    labels=None,
                )
                binary_stats_update(stats, token_count, logits, labels)

    metrics_df = summarize_stats(stats, num_samples)
    csv_path = output_dir / "figure10_expert_tokens_classification_metrics.csv"
    json_path = output_dir / "figure10_expert_tokens_classification_metrics.json"
    metrics_df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(metrics_df.to_dict(orient="records"), indent=2), encoding="utf-8")

    return {
        "metrics_csv": str(csv_path),
        "metrics_json": str(json_path),
        "definition": (
            "Stage-1 MHCAC positive-vs-rest classification metrics over 5 common "
            "abnormalities. The 14-token checkpoint is truncated to top-K trained "
            f"common expert tokens using token_selection={args.token_selection}; "
            "no K-token retraining."
        ),
        "checkpoint": str(args.checkpoint),
        "token_selection": args.token_selection,
        "ranked_token_indices": [int(x) for x in ranked_indices.detach().cpu().tolist()],
        "token_counts": token_counts,
        "num_samples": num_samples,
        "metrics": metrics_df.to_dict(orient="records"),
    }


def build_figure10_from_metrics(metrics_csv: Path, output_dir: Path, llm_label: str | None = None) -> dict:
    df = pd.read_csv(metrics_csv).sort_values("Expert Tokens")
    title = "Effect of Expert Tokens on Classification Metrics"
    if llm_label:
        title = f"{title} ({llm_label})"

    fig, ax = plt.subplots(figsize=(7.4, 5.2), dpi=220)
    style = {
        "Precision": {"color": "#2563eb", "marker": "o"},
        "Recall": {"color": "#f97316", "marker": "s"},
        "F1-Score": {"color": "#16a34a", "marker": "^"},
    }
    for metric, kw in style.items():
        ax.plot(
            df["Expert Tokens"],
            df[metric],
            label=metric,
            linewidth=2,
            markersize=4,
            **kw,
        )

    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel("Number of Expert Tokens", fontsize=10)
    ax.set_ylabel("Metric Score", fontsize=10)
    ax.set_xticks(df["Expert Tokens"].tolist())
    ymin = max(0.0, float(df[["Precision", "Recall", "F1-Score"]].min().min()) - 0.04)
    ymax = min(1.0, float(df[["Precision", "Recall", "F1-Score"]].max().max()) + 0.05)
    if ymax - ymin < 0.12:
        ymax = min(1.0, ymax + 0.08)
        ymin = max(0.0, ymin - 0.08)
    ax.set_ylim(ymin, ymax)
    ax.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.55)
    ax.legend(title="Metrics", fontsize=8, title_fontsize=8, loc="best", frameon=True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#555555")

    caption = "Figure 10. Effect of Expert Tokens in MHCAC."
    fig.text(0.08, 0.035, caption, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0.03, 0.08, 0.98, 0.98))

    suffix = ""
    if llm_label:
        suffix = "_" + llm_label.lower().replace("+", "").replace(" ", "_").replace(".", "")
    png = output_dir / f"figure10_expert_tokens_classification{suffix}.png"
    pdf = output_dir / f"figure10_expert_tokens_classification{suffix}.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    return {"figure_png": str(png), "figure_pdf": str(pdf), "source_csv": str(metrics_csv)}


def upload_outputs(output_dir: Path, gcs_output: str) -> None:
    for path in sorted(output_dir.glob("*")):
        if path.is_file():
            gcs_upload(path, gcs_output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="07_all_three")
    parser.add_argument("--checkpoint", default="/mnt/meta-cxr-checkpoint/07_all_three/checkpoint_best.pth")
    parser.add_argument("--output-dir", default="/home/phuong/META-CXR/output/figures9_10")
    parser.add_argument("--gcs-output", default="gs://meta-cxr-checkpoint/eval/paper_figures/figure9_10")
    parser.add_argument("--token-counts", default="4,6,8,10,12")
    parser.add_argument(
        "--token-selection",
        default="attention_pooling",
        choices=["attention_pooling", "norm", "first", "last"],
    )
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--skip-figure10-compute", action="store_true")
    parser.add_argument("--figure10-metrics-csv", default="")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-upload", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="fig9_10_gcs_") as tmp:
        summary = {
            "figure9": build_figure9(Path(tmp), output_dir),
            "figure10": {},
        }

    if args.skip_figure10_compute:
        if not args.figure10_metrics_csv:
            raise ValueError("--skip-figure10-compute requires --figure10-metrics-csv")
        fig10_csv = Path(args.figure10_metrics_csv)
    else:
        fig10_result = compute_figure10_metrics(args, output_dir)
        summary["figure10"].update(fig10_result)
        fig10_csv = Path(fig10_result["metrics_csv"])

    summary["figure10"]["combined"] = build_figure10_from_metrics(fig10_csv, output_dir)
    summary["figure10"]["vicuna_copy"] = build_figure10_from_metrics(
        fig10_csv, output_dir, llm_label="Vicuna + LoRA"
    )
    summary["figure10"]["medgemma_copy"] = build_figure10_from_metrics(
        fig10_csv, output_dir, llm_label="MedGemma1.5 + LoRA"
    )

    summary["outputs"] = {
        "local_dir": str(output_dir),
        "gcs_output": args.gcs_output,
    }
    summary_path = output_dir / "figures9_10_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if not args.no_upload:
        upload_outputs(output_dir, args.gcs_output)
        gcs_upload(Path(__file__), args.gcs_output.rstrip("/") + "/scripts")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
