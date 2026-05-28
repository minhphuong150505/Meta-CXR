#!/usr/bin/env python3
"""Build paper-style NLP and clinical efficacy tables for MedGemma Q-Former eval."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


RUNS = [
    "01_biovil_only",
    "02_pubmedclip_only",
    "03_swin_only",
    "04_biovil_pubmedclip",
    "05_biovil_swin",
    "06_pubmedclip_swin",
    "07_all_three",
]

COMMON_5 = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]

NLP_ORDER = [
    ("BLEU-1", "BLEU-1"),
    ("BLEU-2", "BLEU-2"),
    ("BLEU-3", "BLEU-3"),
    ("BLEU-4", "BLEU-4"),
    ("METEOR", "METEOR"),
    ("ROUGE-L", "ROUGE-L"),
    ("CIDEr", "CIDEr"),
    ("BERTScore", "BERTScore"),
    ("RadGraph Entity F1", "RadGraph_Entity_F1"),
    ("RadGraph Relation F1", "RadGraph_Relation_F1"),
    ("RadGraph Overall F1", "RadGraph_Overall_F1"),
    ("RadCliQ ↓", "RadCliQ"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="/home/phuong/META-CXR/output/medgemma_qformer")
    parser.add_argument("--metrics-dir", default="/home/phuong/META-CXR/output/medgemma_qformer/metrics")
    parser.add_argument("--output-dir", default="/home/phuong/META-CXR/output/medgemma_qformer/paper_tables")
    parser.add_argument("--chexpert-csv", default="/home/phuong/META-CXR/data/images/mimic-cxr-2.0.0-chexpert.csv")
    parser.add_argument("--metadata-csv", default="/home/phuong/META-CXR/data/images/mimic-cxr-2.0.0-metadata.csv")
    parser.add_argument("--run-name", default="07_all_three", choices=RUNS)
    parser.add_argument("--gcs-output", default="gs://meta-cxr-checkpoint/eval/MedGemma_QFormer/paper_tables")
    parser.add_argument("--skip-upload", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def load_radgraph(metrics_dir: Path) -> dict[str, dict]:
    path = metrics_dir / "RadGraph" / "summary.json"
    with path.open() as f:
        data = json.load(f)
    return {row["run"]: row for row in data["runs"]}


def load_nlg_table(metrics_dir: Path) -> pd.DataFrame:
    return pd.read_csv(metrics_dir / "nlg_metrics_table_medgemma_qformer.csv")


def build_nlp_table(metrics_dir: Path, run_name: str) -> pd.DataFrame:
    nlg = load_nlg_table(metrics_dir)
    row = nlg[nlg["Run"] == run_name]
    if row.empty:
        raise ValueError(f"Run {run_name!r} not found in NLG table")
    row = row.iloc[0].to_dict()

    radgraph = load_radgraph(metrics_dir).get(run_name)
    if not radgraph:
        raise ValueError(f"Run {run_name!r} not found in RadGraph summary")

    values = {
        "BLEU-1": row["BLEU-1"],
        "BLEU-2": row["BLEU-2"],
        "BLEU-3": row["BLEU-3"],
        "BLEU-4": row["BLEU-4"],
        "METEOR": row["METEOR"],
        "ROUGE-L": row["ROUGE-L"],
        "CIDEr": row["CIDEr"],
        "BERTScore": row["BERTScore"],
        "RadGraph_Entity_F1": radgraph["entity_f1"],
        "RadGraph_Relation_F1": radgraph["relation_f1"],
        "RadGraph_Overall_F1": radgraph["overall_f1"],
        "RadCliQ": row["RadCliQ"],
    }
    return pd.DataFrame(
        [{"Metric": label, "Value": round(float(values[key]), 4)} for label, key in NLP_ORDER]
    )


def build_clinical_tables(input_dir: Path, chexpert_csv: Path, metadata_csv: Path) -> tuple[pd.DataFrame, dict]:
    chexpert = pd.read_csv(chexpert_csv)
    metadata = pd.read_csv(metadata_csv, usecols=["dicom_id", "subject_id", "study_id"])
    metadata["subject_id"] = metadata["subject_id"].astype(str)
    metadata["study_id"] = metadata["study_id"].astype(str)
    chexpert["subject_id"] = chexpert["subject_id"].astype(str)
    chexpert["study_id"] = chexpert["study_id"].astype(str)

    truth = metadata.merge(chexpert, on=["subject_id", "study_id"], how="left")
    truth_by_dicom = truth.set_index("dicom_id")

    run_rows = []
    per_run = {}
    for run_name in RUNS:
        path = input_dir / f"reports_medgemma_qformer_{run_name}.jsonl"
        if not path.exists():
            continue

        records = load_jsonl(path)
        y_true = {name: [] for name in COMMON_5}
        y_pred = {name: [] for name in COMMON_5}

        for record in records:
            dicom_id = record.get("dicom_id")
            if dicom_id not in truth_by_dicom.index:
                continue
            gt = truth_by_dicom.loc[dicom_id]
            positives = set(record.get("pred_groups", {}).get("positive", []))
            for name in COMMON_5:
                y_true[name].append(1 if float(gt.get(name, 0.0) or 0.0) == 1.0 else 0)
                y_pred[name].append(1 if name in positives else 0)

        per_abnormality = {}
        precisions, recalls, f1s = [], [], []
        for name in COMMON_5:
            yt = np.array(y_true[name], dtype=int)
            yp = np.array(y_pred[name], dtype=int)
            precision = precision_score(yt, yp, zero_division=0)
            recall = recall_score(yt, yp, zero_division=0)
            f1 = f1_score(yt, yp, zero_division=0)
            per_abnormality[name] = {
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
            "run": run_name,
            "n_samples": int(len(next(iter(y_true.values()), []))),
            "mean_precision_over_5_common": round(float(np.mean(precisions)), 4),
            "mean_recall_over_5_common": round(float(np.mean(recalls)), 4),
            "macro_f1_over_5_common": round(float(np.mean(f1s)), 4),
            "per_abnormality": per_abnormality,
        }
        per_run[run_name] = summary
        run_rows.append(
            {
                "Run": run_name,
                "Precision": summary["mean_precision_over_5_common"],
                "Recall": summary["mean_recall_over_5_common"],
                "Macro F1": summary["macro_f1_over_5_common"],
                "N_samples": summary["n_samples"],
            }
        )

    return pd.DataFrame(run_rows), per_run


POSITIVE_PATTERNS = {
    "Atelectasis": [
        r"\batelecta\w*\b",
        r"\bvolume loss\b",
        r"\bpartial collapse\b",
    ],
    "Cardiomegaly": [
        r"\bcardiomegal\w*\b",
        r"\benlarged (?:cardiac silhouette|heart)\b",
        r"\bheart (?:is )?enlarged\b",
    ],
    "Consolidation": [
        r"\bconsolidat\w*\b",
    ],
    "Edema": [
        r"\bedema\b",
        r"\bpulmonary vascular congestion\b",
        r"\bvascular congestion\b",
        r"\binterstitial edema\b",
    ],
    "Pleural Effusion": [
        r"\bpleural effusion\w*\b",
    ],
}

NEGATION_PREFIX = re.compile(
    r"\b(no|not|without|absent|negative for|no evidence of|no definite|free of|resolved)\b",
    flags=re.IGNORECASE,
)


def is_negated(text: str, start: int, end: int) -> bool:
    before = text[max(0, start - 45):start]
    after = text[end:min(len(text), end + 35)]
    if NEGATION_PREFIX.search(before):
        return True
    if re.search(r"\b(is|are|was|were)?\s*(normal|unchanged|clear|resolved)\b", after, re.I):
        return True
    return False


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


def build_text_clinical_tables(input_dir: Path, chexpert_csv: Path, metadata_csv: Path) -> tuple[pd.DataFrame, dict]:
    chexpert = pd.read_csv(chexpert_csv)
    metadata = pd.read_csv(metadata_csv, usecols=["dicom_id", "subject_id", "study_id"])
    metadata["subject_id"] = metadata["subject_id"].astype(str)
    metadata["study_id"] = metadata["study_id"].astype(str)
    chexpert["subject_id"] = chexpert["subject_id"].astype(str)
    chexpert["study_id"] = chexpert["study_id"].astype(str)

    truth = metadata.merge(chexpert, on=["subject_id", "study_id"], how="left")
    truth_by_dicom = truth.set_index("dicom_id")

    run_rows = []
    per_run = {}
    for run_name in RUNS:
        path = input_dir / f"reports_medgemma_qformer_{run_name}.jsonl"
        if not path.exists():
            continue

        records = load_jsonl(path)
        y_true = {name: [] for name in COMMON_5}
        y_pred = {name: [] for name in COMMON_5}

        for record in records:
            dicom_id = record.get("dicom_id")
            if dicom_id not in truth_by_dicom.index:
                continue
            gt = truth_by_dicom.loc[dicom_id]
            positives = label_report_text(record.get("pred", ""))
            for name in COMMON_5:
                y_true[name].append(1 if float(gt.get(name, 0.0) or 0.0) == 1.0 else 0)
                y_pred[name].append(1 if name in positives else 0)

        per_abnormality = {}
        precisions, recalls, f1s = [], [], []
        for name in COMMON_5:
            yt = np.array(y_true[name], dtype=int)
            yp = np.array(y_pred[name], dtype=int)
            precision = precision_score(yt, yp, zero_division=0)
            recall = recall_score(yt, yp, zero_division=0)
            f1 = f1_score(yt, yp, zero_division=0)
            per_abnormality[name] = {
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
            "run": run_name,
            "n_samples": int(len(next(iter(y_true.values()), []))),
            "mean_precision_over_5_common": round(float(np.mean(precisions)), 4),
            "mean_recall_over_5_common": round(float(np.mean(recalls)), 4),
            "macro_f1_over_5_common": round(float(np.mean(f1s)), 4),
            "per_abnormality": per_abnormality,
        }
        per_run[run_name] = summary
        run_rows.append(
            {
                "Run": run_name,
                "Precision": summary["mean_precision_over_5_common"],
                "Recall": summary["mean_recall_over_5_common"],
                "Macro F1": summary["macro_f1_over_5_common"],
                "N_samples": summary["n_samples"],
            }
        )

    return pd.DataFrame(run_rows), per_run


def upload_dir(local_dir: Path, gcs_output: str) -> None:
    cmd = f"gcloud storage cp -r {local_dir}/* {gcs_output}/ --quiet"
    print(f">>> Uploading: {cmd}")
    rc = os.system(cmd)
    if rc != 0:
        raise RuntimeError(f"Upload failed with rc={rc}: {cmd}")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    metrics_dir = Path(args.metrics_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nlp_table = build_nlp_table(metrics_dir, args.run_name)
    clinical_all, clinical_summary = build_clinical_tables(
        input_dir,
        Path(args.chexpert_csv),
        Path(args.metadata_csv),
    )
    text_clinical_all, text_clinical_summary = build_text_clinical_tables(
        input_dir,
        Path(args.chexpert_csv),
        Path(args.metadata_csv),
    )
    clinical_row = clinical_all[clinical_all["Run"] == args.run_name].iloc[0]
    text_clinical_row = text_clinical_all[text_clinical_all["Run"] == args.run_name].iloc[0]
    clinical_table = pd.DataFrame(
        [
            {"Metric": "Precision", "Value": clinical_row["Precision"]},
            {"Metric": "Recall", "Value": clinical_row["Recall"]},
            {"Metric": "Macro F1", "Value": clinical_row["Macro F1"]},
        ]
    )
    text_clinical_table = pd.DataFrame(
        [
            {"Metric": "Precision", "Value": text_clinical_row["Precision"]},
            {"Metric": "Recall", "Value": text_clinical_row["Recall"]},
            {"Metric": "Macro F1", "Value": text_clinical_row["Macro F1"]},
        ]
    )

    nlp_table.to_csv(output_dir / f"nlp_derived_metrics_{args.run_name}.csv", index=False)
    clinical_table.to_csv(output_dir / f"clinical_efficacy_stage1_{args.run_name}.csv", index=False)
    clinical_all.to_csv(output_dir / "clinical_efficacy_stage1_all_runs.csv", index=False)
    text_clinical_table.to_csv(output_dir / f"clinical_efficacy_report_text_{args.run_name}.csv", index=False)
    text_clinical_all.to_csv(output_dir / "clinical_efficacy_report_text_all_runs.csv", index=False)

    with (output_dir / "paper_tables_summary.json").open("w") as f:
        json.dump(
            {
                "run_name": args.run_name,
                "sample_limit": 200,
                "clinical_stage1_definition": "Stage-1 predicted positive labels vs CheXpert positive ground truth, positive-vs-rest, mean over 5 common abnormalities.",
                "clinical_report_text_definition": "Rule-based positive mention extraction from generated reports vs CheXpert positive ground truth, positive-vs-rest, mean over 5 common abnormalities.",
                "common_abnormalities": COMMON_5,
                "nlp_derived_metrics": nlp_table.to_dict(orient="records"),
                "clinical_efficacy_stage1": clinical_table.to_dict(orient="records"),
                "clinical_efficacy_stage1_all_runs": clinical_summary,
                "clinical_efficacy_report_text": text_clinical_table.to_dict(orient="records"),
                "clinical_efficacy_report_text_all_runs": text_clinical_summary,
            },
            f,
            indent=2,
        )

    print("\nNLP-derived Metrics")
    print(nlp_table.to_string(index=False))
    print("\nClinical Efficacy from stage-1 labels (mean over 5 common abnormalities)")
    print(clinical_table.to_string(index=False))
    print("\nClinical Efficacy from generated report text (mean over 5 common abnormalities)")
    print(text_clinical_table.to_string(index=False))
    print("\nClinical Efficacy from stage-1 labels - all runs")
    print(clinical_all.to_string(index=False))
    print("\nClinical Efficacy from generated report text - all runs")
    print(text_clinical_all.to_string(index=False))

    if not args.skip_upload:
        upload_dir(output_dir, args.gcs_output)
        print(f">>> Uploaded to {args.gcs_output}/")


if __name__ == "__main__":
    main()
