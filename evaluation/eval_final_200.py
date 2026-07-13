#!/usr/bin/env python3
"""Final 200-sample evaluation for finetuned Vicuna and MedGemma.

Outputs paper-style Table 2 and Table 3 artifacts:
  - standard NLG metrics: BLEU-1/2/3/4, METEOR, ROUGE-L, CIDEr
  - clinical/domain metrics: BERTScore, RadGraph F1, RadCliQ
  - clinical efficacy metrics: Precision, Recall, Macro F1

The clinical efficacy row uses rule-based positive mention extraction from the
generated Findings text against CheXpert positive labels for the five common
MIMIC-CXR abnormalities used in the META-CXR paper.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "training"))
sys.path.insert(0, str(PROJECT_DIR / "model"))

import train_eval_figure9_llm_variants_200 as fig9  # noqa: E402

RUN_NAME = "07_all_three"
DATASET_LABEL = "MIMIC-CXR p10 test (200)"
COMMON_5 = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
STANDARD_METRICS = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr"]
CLINICAL_METRICS = ["BERTScore", "RadGraph F1", "RadCliQ", "Precision", "Recall", "Macro F1"]
PAPER_SOURCE = "Chest X-Ray Report Generation Using Abnormality Guided Vision Language Model"

POSITIVE_PATTERNS = {
    "Atelectasis": [r"\batelecta\w*\b", r"\bvolume loss\b", r"\bpartial collapse\b"],
    "Cardiomegaly": [
        r"\bcardiomegal\w*\b",
        r"\benlarged (?:cardiac silhouette|heart)\b",
        r"\bheart (?:is )?enlarged\b",
    ],
    "Consolidation": [r"\bconsolidat\w*\b"],
    "Edema": [
        r"\bedema\b",
        r"\bpulmonary vascular congestion\b",
        r"\bvascular congestion\b",
        r"\binterstitial edema\b",
    ],
    "Pleural Effusion": [r"\bpleural effusion\w*\b"],
}
NEGATION_PREFIX = re.compile(
    r"\b(no|not|without|absent|negative for|no evidence of|no definite|free of|resolved)\b",
    flags=re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", default="/home/phuong/checkpoints")
    parser.add_argument("--output-dir", default=str(PROJECT_DIR / "eval_final"))
    parser.add_argument("--gcs-output", default="gs://mimic-cxr-lite-data/outputs/eval_final")
    parser.add_argument("--sample-limit", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--vicuna-train-limit", type=int, default=200)
    parser.add_argument("--vicuna-train-epochs", type=int, default=1)
    parser.add_argument("--vicuna-grad-accum", type=int, default=4)
    parser.add_argument("--vicuna-lr", type=float, default=2e-4)
    parser.add_argument(
        "--vicuna-adapter",
        default=str(PROJECT_DIR / "output" / "eval_final_vicuna" / "adapters" / "vicuna_finetuned_200"),
    )
    parser.add_argument(
        "--medgemma-adapter",
        default=str(PROJECT_DIR / "output" / "medgemma_qlora" / "adapters" / "medgemma_qlora_fine"),
    )
    parser.add_argument(
        "--medgemma-eval-dir",
        default=str(PROJECT_DIR / "output" / "medgemma_qlora" / "eval"),
    )
    parser.add_argument("--chexpert-csv", default="/home/phuong/data/csv/mimic-cxr-2.0.0-chexpert.csv")
    parser.add_argument("--metadata-csv", default="/home/phuong/data/csv/mimic-cxr-2.0.0-metadata.csv")
    parser.add_argument("--force-vicuna-train", action="store_true")
    parser.add_argument("--force-generate", action="store_true")
    parser.add_argument("--skip-upload", action="store_true")
    return parser.parse_args()


def run_cmd(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(args, check=check, text=True, capture_output=True)


def upload_dir(local_dir: Path, gcs_output: str) -> None:
    run_cmd(["gcloud", "storage", "cp", "-r", str(local_dir), gcs_output.rstrip("/") + "/"])


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def is_negated(text: str, start: int, end: int) -> bool:
    before = text[max(0, start - 45):start]
    after = text[end:min(len(text), end + 35)]
    if NEGATION_PREFIX.search(before):
        return True
    return bool(re.search(r"\b(is|are|was|were)?\s*(normal|unchanged|clear|resolved)\b", after, re.I))


def label_report_text(text: str) -> set[str]:
    normalized = " ".join(str(text).lower().replace("/", " ").split())
    positives = set()
    for label, patterns in POSITIVE_PATTERNS.items():
        for pattern in patterns:
            match = re.search(pattern, normalized, flags=re.IGNORECASE)
            if match and not is_negated(normalized, match.start(), match.end()):
                positives.add(label)
                break
    if re.search(r"\bheart size (?:is )?(?:within normal limits|normal|not enlarged)\b", normalized):
        positives.discard("Cardiomegaly")
    if re.search(r"\bcardiac silhouette (?:is )?not enlarged\b", normalized):
        positives.discard("Cardiomegaly")
    return positives


def compute_radgraph(preds: list[str], refs: list[str]) -> tuple[dict, str | None]:
    try:
        from radgraph import F1RadGraph
    except Exception as exc:
        return {"entity_f1": None, "relation_f1": None, "overall_f1": None}, f"radgraph import failed: {exc}"

    valid_preds, valid_refs = [], []
    for pred, ref in zip(preds, refs):
        if str(pred).strip() and str(ref).strip():
            valid_preds.append(str(pred))
            valid_refs.append(str(ref))
    if not valid_preds:
        return {"entity_f1": 0.0, "relation_f1": 0.0, "overall_f1": 0.0}, None

    try:
        scorer = F1RadGraph(reward_level="all", model_type="radgraph-xl")
        mean_reward, *_ = scorer(hyps=valid_preds, refs=valid_refs)
        return {
            "entity_f1": round(float(mean_reward[0]), 4),
            "relation_f1": round(float(mean_reward[1]), 4),
            "overall_f1": round(float(mean_reward[2]), 4),
        }, None
    except Exception as exc:
        return {"entity_f1": None, "relation_f1": None, "overall_f1": None}, f"radgraph scoring failed: {exc}"


def compute_radcliq(bertscore: float | None, radgraph_overall: float | None) -> float | None:
    if bertscore is None or radgraph_overall is None:
        return None
    return round(float(math.sqrt((1 - bertscore) ** 2 + (1 - radgraph_overall) ** 2)), 4)


def clinical_efficacy(records: list[dict], chexpert_csv: Path, metadata_csv: Path) -> tuple[dict, dict]:
    chexpert = pd.read_csv(chexpert_csv)
    metadata = pd.read_csv(metadata_csv, usecols=["dicom_id", "subject_id", "study_id"])
    metadata["subject_id"] = metadata["subject_id"].astype(str)
    metadata["study_id"] = metadata["study_id"].astype(str)
    chexpert["subject_id"] = chexpert["subject_id"].astype(str)
    chexpert["study_id"] = chexpert["study_id"].astype(str)
    truth = metadata.merge(chexpert, on=["subject_id", "study_id"], how="left").set_index("dicom_id")

    y_true = {name: [] for name in COMMON_5}
    y_pred = {name: [] for name in COMMON_5}
    for record in records:
        dicom_id = str(record.get("dicom_id", ""))
        if dicom_id not in truth.index:
            continue
        gt = truth.loc[dicom_id]
        positives = label_report_text(record.get("pred", ""))
        for name in COMMON_5:
            y_true[name].append(1 if float(gt.get(name, 0.0) or 0.0) == 1.0 else 0)
            y_pred[name].append(1 if name in positives else 0)

    per_label = {}
    precisions, recalls, f1s = [], [], []
    for name in COMMON_5:
        yt = np.array(y_true[name], dtype=int)
        yp = np.array(y_pred[name], dtype=int)
        precision = precision_score(yt, yp, zero_division=0)
        recall = recall_score(yt, yp, zero_division=0)
        f1 = f1_score(yt, yp, zero_division=0)
        per_label[name] = {
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "n_true_positive": int(yt.sum()),
            "n_pred_positive": int(yp.sum()),
            "n_samples": int(len(yt)),
        }
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    summary = {
        "Precision": round(float(np.mean(precisions)), 4),
        "Recall": round(float(np.mean(recalls)), 4),
        "Macro F1": round(float(np.mean(f1s)), 4),
        "N_clinical": int(len(next(iter(y_true.values()), []))),
    }
    return summary, per_label


def table_to_image(df: pd.DataFrame, title: str, path: Path) -> None:
    fig_w = max(8.0, min(18.0, 1.15 * len(df.columns)))
    fig_h = max(2.4, 1.0 + 0.42 * len(df))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=220)
    ax.axis("off")
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left", pad=10)
    table = ax.table(
        cellText=df.fillna("N/A").astype(str).values,
        colLabels=df.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.6)
    table.scale(1.0, 1.35)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor("#3b3b3b")
        cell.set_linewidth(0.45)
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#1f2937")
        else:
            cell.set_facecolor("#f8fafc" if row % 2 else "#ffffff")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def grouped_bar(df: pd.DataFrame, metrics: list[str], title: str, path: Path) -> None:
    labels = df["Method"].tolist()
    x = np.arange(len(metrics))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9.2, 4.8), dpi=220)
    palette = ["#2563eb", "#16a34a"]
    for i, (_, row) in enumerate(df.iterrows()):
        vals = [np.nan if row[m] is None else float(row[m]) for m in metrics]
        ax.bar(x + (i - 0.5) * width, vals, width, label=labels[i], color=palette[i % len(palette)])
    ax.set_title(title, fontsize=12, fontweight="bold", loc="left")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=25, ha="right")
    ax.grid(axis="y", color="#d1d5db", linewidth=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def radar_plot(df: pd.DataFrame, path: Path) -> None:
    metrics = ["BERTScore", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr"]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=(6.6, 6.2), dpi=220)
    ax = fig.add_subplot(111, polar=True)
    colors = ["#2563eb", "#16a34a"]
    for i, (_, row) in enumerate(df.iterrows()):
        vals = [0.0 if row[m] is None or pd.isna(row[m]) else float(row[m]) for m in metrics]
        vals += vals[:1]
        ax.plot(angles, vals, color=colors[i % len(colors)], linewidth=2.0, label=row["Method"])
        ax.fill(angles, vals, color=colors[i % len(colors)], alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=9)
    ymax = max(0.5, min(1.0, math.ceil(max(float(df[m].max()) for m in metrics) * 10 + 1) / 10))
    ax.set_ylim(0, ymax)
    ax.grid(color="#cbd5e1", linewidth=0.7)
    ax.set_title("Finetuned LLM Report-Generation Metrics (200 samples)", fontsize=11, pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.12), frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def figure9_finetuned_plot(df: pd.DataFrame, path: Path) -> None:
    metrics = ["BERTScore", "CIDEr", "BLEU-4", "ROUGE-L", "METEOR"]
    fig, ax = plt.subplots(figsize=(8.2, 4.9), dpi=220)
    colors = ["#1d4ed8", "#15803d"]
    markers = ["o", "s"]
    x = np.arange(len(metrics))
    for i, (_, row) in enumerate(df.iterrows()):
        values = [np.nan if pd.isna(row[m]) else float(row[m]) for m in metrics]
        ax.plot(
            x,
            values,
            marker=markers[i % len(markers)],
            linewidth=2.0,
            markersize=5.2,
            color=colors[i % len(colors)],
            label=row["Method"],
        )
    ax.set_title("Figure 9-style Effect of Fine-tuned LLMs", fontsize=12, fontweight="bold", loc="left")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    max_value = np.nanmax(df[metrics].to_numpy(dtype=float))
    ax.set_ylim(0, max(0.5, min(1.2, max_value * 1.18 if np.isfinite(max_value) else 1.0)))
    ax.grid(axis="y", color="#d1d5db", linewidth=0.7)
    ax.legend(frameon=False, loc="upper left")
    ax.set_ylabel("Score")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def build_table6_bertscore(all_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in all_df.iterrows():
        rows.append(
            {
                "Dataset": row["Dataset"],
                "Method": row["Method"],
                "RN50": "Y",
                "ViT": "Y",
                "Swin": "Y",
                "BERTScore": row["BERTScore"],
            }
        )
    return pd.DataFrame(rows)


def write_paper_manifest(output_dir: Path) -> None:
    manifest = pd.DataFrame(
        [
            {
                "Paper item": "Table 2",
                "Paper role": "Standard NLP metrics",
                "Generated CSV": "table2_standard_nlp_metrics_200.csv",
                "Generated figure": "table2_standard_nlp_metrics_200.png",
            },
            {
                "Paper item": "Table 3",
                "Paper role": "Clinical/domain and clinical efficacy metrics",
                "Generated CSV": "table3_clinical_domain_metrics_200.csv",
                "Generated figure": "table3_clinical_domain_metrics_200.png",
            },
            {
                "Paper item": "Table 6",
                "Paper role": "Report-generation BERTScore with RN50 + ViT + Swin",
                "Generated CSV": "table6_report_generation_bertscore_200.csv",
                "Generated figure": "table6_report_generation_bertscore_200.png",
            },
            {
                "Paper item": "Figure 9",
                "Paper role": "Effect of tuned LLMs",
                "Generated CSV": "all_metrics_200.csv",
                "Generated figure": "figure9_effect_finetuned_llms_200.png",
            },
        ]
    )
    manifest.to_csv(output_dir / "paper_related_artifacts_manifest.csv", index=False)


def evaluate_or_reuse(
    family: str,
    variant: str,
    llm_factory,
    records: list[dict],
    eval_dir: Path,
    max_new_tokens: int,
    force_generate: bool,
) -> tuple[dict, list[dict]]:
    metrics_path = eval_dir / f"metrics_{family}_{variant}_{RUN_NAME}.json"
    reports_path = eval_dir / f"reports_{family}_{variant}_{RUN_NAME}.jsonl"
    if force_generate and metrics_path.exists():
        metrics_path.unlink()
    if force_generate and reports_path.exists():
        reports_path.unlink()

    if not metrics_path.exists() or not reports_path.exists():
        llm = llm_factory()
        metrics = fig9.evaluate_variant(family, variant, llm, records, eval_dir, max_new_tokens, "fine")
        del llm
        fig9.clear_memory()
    else:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    report_rows = load_jsonl(reports_path)
    return metrics, report_rows


def main() -> None:
    args = parse_args()
    fig9.set_seed(fig9.SEED)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_dir = output_dir / "generated_reports"
    generated_dir.mkdir(exist_ok=True)

    checkpoint_root = Path(args.checkpoint_root)
    test_records = fig9.build_stage1_records(
        checkpoint_root, output_dir, "test", args.sample_limit, args.num_workers
    )

    vicuna_adapter = Path(args.vicuna_adapter)
    if args.force_vicuna_train or not (vicuna_adapter / "adapter_config.json").exists() or not (vicuna_adapter / "img_proj.pt").exists():
        train_records = fig9.build_stage1_records(
            checkpoint_root, output_dir, "train", args.vicuna_train_limit, args.num_workers
        )
        print(f"[vicuna] training LoRA + img_proj -> {vicuna_adapter}")
        llm = fig9.VariantLLM("vicuna", train_adapter=True)
        llm.train_fine(train_records, vicuna_adapter, args.vicuna_train_epochs, args.vicuna_lr, args.vicuna_grad_accum)
        del llm
        fig9.clear_memory()
    else:
        print(f"[vicuna] reusing adapter {vicuna_adapter}")

    vicuna_eval_dir = generated_dir / "vicuna"
    medgemma_eval_dir = Path(args.medgemma_eval_dir)
    medgemma_adapter = Path(args.medgemma_adapter)
    if not (medgemma_adapter / "adapter_config.json").exists():
        raise FileNotFoundError(f"Missing MedGemma adapter: {medgemma_adapter}")

    vicuna_metrics, vicuna_reports = evaluate_or_reuse(
        "vicuna",
        "finetuned",
        lambda: _load_vicuna(vicuna_adapter),
        test_records,
        vicuna_eval_dir,
        args.max_new_tokens,
        args.force_generate,
    )
    medgemma_metrics, medgemma_reports = evaluate_or_reuse(
        "medgemma",
        "qlora_fine",
        lambda: _load_medgemma(medgemma_adapter),
        test_records,
        medgemma_eval_dir,
        args.max_new_tokens,
        args.force_generate,
    )

    rows = []
    detailed = {}
    for method, family, metrics, reports in [
        ("META-CXR + Vicuna FT", "vicuna", vicuna_metrics, vicuna_reports),
        ("META-CXR + MedGemma FT", "medgemma", medgemma_metrics, medgemma_reports),
    ]:
        preds = [str(r.get("pred", "")) for r in reports]
        refs = [str(r.get("ref", "")) for r in reports]
        radgraph, radgraph_error = compute_radgraph(preds, refs)
        clinical, per_label = clinical_efficacy(reports, Path(args.chexpert_csv), Path(args.metadata_csv))
        radcliq = compute_radcliq(metrics.get("BERTScore"), radgraph.get("overall_f1"))
        row = {
            "Dataset": DATASET_LABEL,
            "Method": method,
            "LLM": "Vicuna-7B" if family == "vicuna" else "MedGemma-1.5-4B",
            "Run": RUN_NAME,
            "N": int(metrics.get("N", len(reports))),
            "BLEU-1": metrics.get("BLEU-1"),
            "BLEU-2": metrics.get("BLEU-2"),
            "BLEU-3": metrics.get("BLEU-3"),
            "BLEU-4": metrics.get("BLEU-4"),
            "METEOR": metrics.get("METEOR"),
            "ROUGE-L": metrics.get("ROUGE-L"),
            "CIDEr": metrics.get("CIDEr"),
            "BERTScore": metrics.get("BERTScore"),
            "RadGraph F1": radgraph.get("overall_f1"),
            "RadGraph Entity F1": radgraph.get("entity_f1"),
            "RadGraph Relation F1": radgraph.get("relation_f1"),
            "RadCliQ": radcliq,
            "Precision": clinical["Precision"],
            "Recall": clinical["Recall"],
            "Macro F1": clinical["Macro F1"],
            "N_clinical": clinical["N_clinical"],
        }
        rows.append(row)
        detailed[family] = {
            "method": method,
            "nlg_metrics": metrics,
            "radgraph": radgraph,
            "radgraph_error": radgraph_error,
            "radcliq": radcliq,
            "clinical_efficacy": clinical,
            "clinical_efficacy_per_label": per_label,
            "reports_path": str((vicuna_eval_dir if family == "vicuna" else medgemma_eval_dir) / f"reports_{family}_{'finetuned' if family == 'vicuna' else 'qlora_fine'}_{RUN_NAME}.jsonl"),
        }

    all_df = pd.DataFrame(rows)
    numeric_cols = [c for c in all_df.columns if c not in {"Dataset", "Method", "LLM", "Run"}]
    for col in numeric_cols:
        all_df[col] = pd.to_numeric(all_df[col], errors="ignore")

    table2 = all_df[["Dataset", "Method", "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr"]]
    table3 = all_df[["Dataset", "Method", "BERTScore", "RadGraph F1", "RadCliQ", "Precision", "Recall", "Macro F1"]]
    table6 = build_table6_bertscore(all_df)
    all_df.to_csv(output_dir / "all_metrics_200.csv", index=False)
    table2.to_csv(output_dir / "table2_standard_nlp_metrics_200.csv", index=False)
    table3.to_csv(output_dir / "table3_clinical_domain_metrics_200.csv", index=False)
    table6.to_csv(output_dir / "table6_report_generation_bertscore_200.csv", index=False)
    write_paper_manifest(output_dir)

    table_to_image(table2.round(4), "Table 2-style Standard NLP Metrics on MIMIC-CXR p10 Test", output_dir / "table2_standard_nlp_metrics_200.png")
    table_to_image(table3.round(4), "Table 3-style Clinical and Domain-Specific Metrics on MIMIC-CXR p10 Test", output_dir / "table3_clinical_domain_metrics_200.png")
    table_to_image(table6.round(4), "Table 6-style Report Generation BERTScore on MIMIC-CXR p10 Test", output_dir / "table6_report_generation_bertscore_200.png")
    grouped_bar(all_df, ["BLEU-4", "METEOR", "ROUGE-L", "CIDEr", "BERTScore"], "NLG and Semantic Metrics (200 samples)", output_dir / "figure_nlg_semantic_metrics_200.png")
    grouped_bar(all_df, ["RadGraph F1", "Precision", "Recall", "Macro F1"], "Clinical Metrics (200 samples)", output_dir / "figure_clinical_metrics_200.png")
    figure9_finetuned_plot(all_df, output_dir / "figure9_effect_finetuned_llms_200.png")
    radar_plot(all_df, output_dir / "figure_finetuned_llm_radar_200.png")

    write_json(
        output_dir / "eval_final_summary.json",
        {
            "dataset": DATASET_LABEL,
            "paper_source": PAPER_SOURCE,
            "run_name": RUN_NAME,
            "sample_limit": args.sample_limit,
            "common_abnormalities": COMMON_5,
            "clinical_efficacy_definition": "Rule-based positive mention extraction from generated Findings text vs CheXpert positive ground truth, positive-vs-rest, mean over 5 common abnormalities.",
            "radcliq_definition": "sqrt((1 - BERTScore)^2 + (1 - RadGraph_Overall_F1)^2); lower is better.",
            "models": detailed,
            "artifacts": {
                "all_metrics": "all_metrics_200.csv",
                "table2": "table2_standard_nlp_metrics_200.csv",
                "table3": "table3_clinical_domain_metrics_200.csv",
                "table6": "table6_report_generation_bertscore_200.csv",
                "paper_manifest": "paper_related_artifacts_manifest.csv",
                "figures": [
                    "table2_standard_nlp_metrics_200.png",
                    "table3_clinical_domain_metrics_200.png",
                    "table6_report_generation_bertscore_200.png",
                    "figure_nlg_semantic_metrics_200.png",
                    "figure_clinical_metrics_200.png",
                    "figure9_effect_finetuned_llms_200.png",
                    "figure_finetuned_llm_radar_200.png",
                ],
            },
        },
    )
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# eval_final",
                "",
                "Final 200-sample evaluation for finetuned Vicuna and MedGemma on MIMIC-CXR p10.",
                "",
                "- `table2_standard_nlp_metrics_200.csv`: BLEU/METEOR/ROUGE-L/CIDEr.",
                "- `table3_clinical_domain_metrics_200.csv`: BERTScore/RadGraph/RadCliQ/clinical efficacy.",
                "- `table6_report_generation_bertscore_200.csv`: BERTScore with all three encoders enabled.",
                "- `all_metrics_200.csv`: combined metrics.",
                "- `paper_related_artifacts_manifest.csv`: mapping from paper table/figure items to generated artifacts.",
                "- PNG/PDF files are paper-style rendered tables and comparison figures.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print("\nTable 2-style")
    print(table2.to_string(index=False))
    print("\nTable 3-style")
    print(table3.to_string(index=False))
    print("\nTable 6-style")
    print(table6.to_string(index=False))

    if not args.skip_upload:
        upload_dir(output_dir, args.gcs_output)
        print(f"[upload] {output_dir} -> {args.gcs_output}/")


def _load_vicuna(adapter_dir: Path):
    llm = fig9.VariantLLM("vicuna", adapter=adapter_dir)
    llm.load_img_proj_if_present(adapter_dir)
    return llm


def _load_medgemma(adapter_dir: Path):
    llm = fig9.VariantLLM("medgemma", adapter=adapter_dir, quantize_4bit=True)
    llm.load_img_proj_if_present(adapter_dir)
    return llm


if __name__ == "__main__":
    main()
