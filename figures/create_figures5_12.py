#!/usr/bin/env python3
"""Create Figure 5 AUC-ROC and Figure 12 expert-evaluation chart."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve

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
SHORT_LABELS = {
    "Enlarged Cardiomediastinum": "Enl_Card",
}
CLASS_MAP = {"negative": 0, "positive": 1, "uncertain": 2}
CLASS_PLOT_ORDER = [
    ("positive", "Positive Class vs Rest"),
    ("negative", "Negative Class vs Rest"),
    ("uncertain", "Uncertain Class vs Rest"),
]

# Digitized from the user-provided Figure 12 reference image. Replace the CSV if
# real expert survey scores are available.
FIGURE12_VALUES = {
    "Image 1": {"Radiologist": 96, "Doctor": 95, "Medical Student": 91, "Average": 94},
    "Image 2": {"Radiologist": 96, "Doctor": 92, "Medical Student": 94, "Average": 95},
    "Image 3": {"Radiologist": 77, "Doctor": 83, "Medical Student": 76, "Average": 79},
    "Image 4": {"Radiologist": 76, "Doctor": 83, "Medical Student": 83, "Average": 81},
    "Image 5": {"Radiologist": 96, "Doctor": 70, "Medical Student": 85, "Average": 83},
    "Image 6": {"Radiologist": 99, "Doctor": 98, "Medical Student": 97, "Average": 98},
    "Image 7": {"Radiologist": 80, "Doctor": 88, "Medical Student": 80, "Average": 83},
    "Image 8": {"Radiologist": 87, "Doctor": 85, "Medical Student": 83, "Average": 85},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--npz",
        default="/home/phuong/META-CXR/output/figure11_classification_thresholds/"
        "07_all_three_test_with_text_full_classification_probs.npz",
    )
    parser.add_argument("--figure5-dir", default="/home/phuong/META-CXR/output/figure5_auc_roc")
    parser.add_argument("--figure12-dir", default="/home/phuong/META-CXR/output/figure12_expert_evaluation")
    parser.add_argument("--gcs-figure5", default="gs://meta-cxr-checkpoint/eval/paper_figures/figure5_auc_roc")
    parser.add_argument(
        "--gcs-figure12",
        default="gs://meta-cxr-checkpoint/eval/paper_figures/figure12_expert_evaluation",
    )
    parser.add_argument("--skip-figure5", action="store_true")
    parser.add_argument("--skip-figure12", action="store_true")
    parser.add_argument("--no-upload", action="store_true")
    return parser.parse_args()


def run_cmd(args: list[str]) -> None:
    subprocess.run(args, check=True)


def upload_dir(path: Path, gcs_dir: str) -> None:
    run_cmd(["gcloud", "storage", "cp", "-r", str(path), gcs_dir.rstrip("/") + "/"])


def load_probs_labels(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    return data["probs"], data["labels"]


def roc_for_class(labels: np.ndarray, probs: np.ndarray, abnormality_idx: int, class_idx: int) -> dict | None:
    y_true = (labels[:, abnormality_idx] == class_idx).astype(int)
    scores = probs[:, abnormality_idx, class_idx]
    if int(y_true.sum()) == 0 or int((1 - y_true).sum()) == 0:
        return None
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    return {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "auc": float(auc(fpr, tpr)),
        "positive_count": int(y_true.sum()),
        "negative_count": int((1 - y_true).sum()),
    }


def build_roc_results(probs: np.ndarray, labels: np.ndarray) -> tuple[dict, pd.DataFrame]:
    roc_results = {class_name: {} for class_name in CLASS_MAP}
    rows = []
    for abn_idx, abnormality in enumerate(ABNORMALITIES):
        for class_name, class_idx in CLASS_MAP.items():
            result = roc_for_class(labels, probs, abn_idx, class_idx)
            if result is None:
                continue
            roc_results[class_name][abnormality] = result
            rows.append(
                {
                    "abnormality": abnormality,
                    "class": class_name,
                    "auc": result["auc"],
                    "positive_count": result["positive_count"],
                    "negative_count": result["negative_count"],
                }
            )
    return roc_results, pd.DataFrame(rows)


def plot_figure5(roc_results: dict, auc_df: pd.DataFrame, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    auc_csv = out_dir / "figure5_auc_roc_values.csv"
    auc_json = out_dir / "figure5_auc_roc_values.json"
    auc_df.to_csv(auc_csv, index=False)
    auc_df.to_json(auc_json, orient="records", indent=2)

    fig, axes = plt.subplots(1, 3, figsize=(14.8, 5.7), dpi=220, sharey=True)
    fig.suptitle("AUC-ROC Curve for All Classes", fontsize=14, y=0.94)
    colors = plt.cm.tab20(np.linspace(0, 1, len(ABNORMALITIES)))

    for ax, (class_name, title) in zip(axes, CLASS_PLOT_ORDER):
        for color, abnormality in zip(colors, ABNORMALITIES):
            result = roc_results[class_name].get(abnormality)
            if result is None:
                continue
            label_name = SHORT_LABELS.get(abnormality, abnormality)
            ax.plot(
                result["fpr"],
                result["tpr"],
                linewidth=1.2,
                color=color,
                label=f"{label_name} (AUC = {result['auc']:.2f})",
            )
        ax.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=0.8, alpha=0.7)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("False Positive Rate", fontsize=9)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.32)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=6.3, loc="lower right", frameon=True, borderpad=0.3)

    axes[0].set_ylabel("True Positive Rate", fontsize=9)
    fig.text(0.055, 0.035, "FIGURE 5.", fontsize=11, fontweight="bold", color="#0072BC")
    fig.text(
        0.13,
        0.035,
        "AUC-ROC Curves on MIMIC-CXR test set for all 14 pathologies including No Finding.",
        fontsize=11,
        fontweight="bold",
        color="black",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.91))

    png = out_dir / "figure5_auc_roc_all_classes.png"
    pdf = out_dir / "figure5_auc_roc_all_classes.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"figure5_png": str(png), "figure5_pdf": str(pdf), "figure5_csv": str(auc_csv)}


def figure12_dataframe() -> pd.DataFrame:
    rows = []
    for image, scores in FIGURE12_VALUES.items():
        for evaluator, correctness in scores.items():
            rows.append({"image": image, "evaluator": evaluator, "correctness_percent": correctness})
    return pd.DataFrame(rows)


def plot_figure12(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = figure12_dataframe()
    csv_path = out_dir / "figure12_expert_report_correctness.csv"
    json_path = out_dir / "figure12_expert_report_correctness.json"
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)

    images = list(FIGURE12_VALUES.keys())
    evaluators = ["Radiologist", "Doctor", "Medical Student", "Average"]
    colors = {
        "Radiologist": "#4F86C6",
        "Doctor": "#C0504D",
        "Medical Student": "#9BBB59",
        "Average": "#8064A2",
    }
    x = np.arange(len(images))
    width = 0.17

    fig, ax = plt.subplots(figsize=(8.6, 5.7), dpi=220)
    for idx, evaluator in enumerate(evaluators):
        vals = [FIGURE12_VALUES[image][evaluator] for image in images]
        ax.bar(x + (idx - 1.5) * width, vals, width, label=evaluator, color=colors[evaluator], edgecolor="white")

    ax.set_title("Reports Correctness", loc="left", fontsize=15, color="#7a7a7a", pad=12)
    ax.set_ylabel("Correctness of the generated report %", fontsize=10, fontweight="bold")
    ax.set_xlabel("Image Number", fontsize=10, fontweight="bold", labelpad=16)
    ax.set_xticks(x)
    ax.set_xticklabels(images, fontsize=9, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", color="#d0d0d0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.55, 1.03), ncol=4, frameon=False, fontsize=9)

    fig.text(0.08, -0.03, "FIGURE 12.", fontsize=13, fontweight="bold", color="#0072BC")
    fig.text(
        0.22,
        -0.03,
        "Experts Evaluation on Report Correctness.",
        fontsize=13,
        fontweight="bold",
        color="black",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    png = out_dir / "figure12_experts_report_correctness.png"
    pdf = out_dir / "figure12_experts_report_correctness.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"figure12_png": str(png), "figure12_pdf": str(pdf), "figure12_csv": str(csv_path)}


def main() -> None:
    args = parse_args()
    summary = {}
    if not args.skip_figure5:
        npz_path = Path(args.npz)
        probs, labels = load_probs_labels(npz_path)
        roc_results, auc_df = build_roc_results(probs, labels)
        summary.update(plot_figure5(roc_results, auc_df, Path(args.figure5_dir)))
        summary["figure5_npz"] = str(npz_path)
        if not args.no_upload:
            upload_dir(Path(args.figure5_dir), args.gcs_figure5)
    if not args.skip_figure12:
        summary.update(plot_figure12(Path(args.figure12_dir)))
        if not args.no_upload:
            upload_dir(Path(args.figure12_dir), args.gcs_figure12)

    summary_path = Path(args.figure12_dir if args.skip_figure5 else args.figure5_dir) / "figures5_12_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
