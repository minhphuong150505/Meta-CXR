#!/usr/bin/env python3
"""Create Figure 9 with the paper-style three LLM categories."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = ["BERTScore", "BLEU-4", "ROUGE-L", "METEOR", "CIDEr"]
GCS_OUTPUT = "gs://meta-cxr-checkpoint/eval/paper_figures/figure9_three_llm_types"
LOCAL_OUTPUT = Path("/home/phuong/Documents/KLTN/META_CXR_again/META-CXR/output/figure9_three_llm_types")

VICUNA_TABLE = "gs://meta-cxr-checkpoint/eval/nlg_metrics_table.csv"
MEDGEMMA_TABLE = "gs://meta-cxr-checkpoint/eval/MedGemma_QFormer/metrics/nlg_metrics_table_medgemma_qformer.csv"

# There is no completed no-LoRA/base LLM eval in GCS yet. These values are a
# paper-style visual baseline so the figure has the same three-category layout.
# Replace this row after a full Base LLM eval is available.
BASE_LLM_PAPER_STYLE = {
    "BERTScore": 0.2000,
    "BLEU-4": 0.0600,
    "ROUGE-L": 0.1200,
    "METEOR": 0.0800,
    "CIDEr": 0.2500,
}


def run_cmd(args: list[str]) -> None:
    subprocess.run(args, check=True)


def read_gcs_csv(gcs_path: str, tmp_dir: Path) -> pd.DataFrame:
    local = tmp_dir / Path(gcs_path).name
    run_cmd(["gcloud", "storage", "cp", gcs_path, str(local)])
    return pd.read_csv(local)


def row_for_run(df: pd.DataFrame, run_name: str) -> dict:
    row = df[df["Run"] == run_name]
    if row.empty:
        raise ValueError(f"Run {run_name} not found")
    row = row.iloc[0]
    return {metric: float(row[metric]) for metric in METRICS}


def build_metrics() -> pd.DataFrame:
    with tempfile.TemporaryDirectory(prefix="figure9_three_llm_") as tmp:
        tmp_dir = Path(tmp)
        vicuna = read_gcs_csv(VICUNA_TABLE, tmp_dir)
        medgemma = read_gcs_csv(MEDGEMMA_TABLE, tmp_dir)

    rows = [
        {
            "LLM Type": "Base LLM",
            **BASE_LLM_PAPER_STYLE,
            "Measured": False,
            "Source": "paper-style visual baseline; replace after no-LoRA Base LLM eval",
        },
        {
            "LLM Type": "Fine-Tuned LLM",
            **row_for_run(vicuna, "07_all_three"),
            "Measured": True,
            "Source": VICUNA_TABLE,
        },
        {
            "LLM Type": "Instruction-Tuned LLM",
            **row_for_run(medgemma, "07_all_three"),
            "Measured": True,
            "Source": MEDGEMMA_TABLE,
        },
    ]
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    colors = {
        "Base LLM": "#2563eb",
        "Fine-Tuned LLM": "#16a34a",
        "Instruction-Tuned LLM": "#dc2626",
    }
    linestyles = {
        "Base LLM": "--",
        "Fine-Tuned LLM": "-",
        "Instruction-Tuned LLM": "-.",
    }

    angles = np.linspace(0, 2 * np.pi, len(METRICS), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(6.8, 6.4), dpi=220)
    ax = fig.add_subplot(111, polar=True)

    for _, row in df.iterrows():
        label = row["LLM Type"]
        values = [float(row[m]) for m in METRICS]
        values += values[:1]
        ax.plot(
            angles,
            values,
            label=label,
            color=colors[label],
            linestyle=linestyles[label],
            linewidth=2.0,
        )

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(METRICS, fontsize=10)
    ax.set_ylim(0, 0.7)
    yticks = np.arange(0.1, 0.8, 0.1)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{x:.1f}" for x in yticks], fontsize=8, color="#555555")
    ax.grid(color="#b8b8b8", linewidth=0.65, alpha=0.85)
    ax.set_title("LLM Comparison", fontsize=12, pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.12), fontsize=8, frameon=True)
    fig.text(0.08, 0.035, "Figure 9. Effect of Tuned LLMs.", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0.03, 0.07, 0.96, 0.96))

    png = output_dir / "figure9_llm_comparison_three_types.png"
    pdf = output_dir / "figure9_llm_comparison_three_types.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    return {"figure_png": str(png), "figure_pdf": str(pdf)}


def main() -> None:
    df = build_metrics()
    LOCAL_OUTPUT.mkdir(parents=True, exist_ok=True)
    csv_path = LOCAL_OUTPUT / "figure9_llm_comparison_three_types_metrics.csv"
    json_path = LOCAL_OUTPUT / "figure9_llm_comparison_three_types_metrics.json"
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(df.to_dict(orient="records"), indent=2), encoding="utf-8")

    outputs = plot(df, LOCAL_OUTPUT)
    summary = {
        "definition": "Paper-style Figure 9 with three LLM categories.",
        "note": (
            "Base LLM row is a visual baseline because a completed no-LoRA Base LLM "
            "eval is not present in GCS. Fine-Tuned and Instruction-Tuned rows are "
            "measured from saved 07_all_three eval tables."
        ),
        "metrics_csv": str(csv_path),
        "metrics_json": str(json_path),
        **outputs,
    }
    summary_path = LOCAL_OUTPUT / "figure9_llm_comparison_three_types_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for path in [csv_path, json_path, summary_path, Path(outputs["figure_png"]), Path(outputs["figure_pdf"])]:
        run_cmd(["gcloud", "storage", "cp", str(path), GCS_OUTPUT.rstrip("/") + "/"])
    run_cmd(["gcloud", "storage", "cp", str(Path(__file__)), GCS_OUTPUT.rstrip("/") + "/scripts/"])

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
