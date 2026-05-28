#!/usr/bin/env python3
"""Compute RadGraph F1 and RadCliQ from existing pred/ref JSONL files.

RadGraph F1: entity/relation/overall F1 via F1RadGraph
RadCliQ (lower=better): sqrt((1-BERTScore)^2 + (1-RadGraph_overall)^2)

Reads reports_vicuna_*.jsonl + summary.json (for BERTScore).
Uploads to gs://meta-cxr-checkpoint/eval/RadGraph/ and /RadCliQ/
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

INPUT_DIR = PROJECT_DIR / "output" / "bertscore_vicuna"
OUTPUT_BASE = PROJECT_DIR / "output" / "nlg_metrics"
GCS_BASE = "gs://meta-cxr-checkpoint/eval"

RUNS = [
    "01_biovil_only",
    "02_pubmedclip_only",
    "03_swin_only",
    "04_biovil_pubmedclip",
    "05_biovil_swin",
    "06_pubmedclip_swin",
    "07_all_three",
]

ENCODER_INFO = {
    "01_biovil_only":       {"RN50": "+", "ViT": "-", "Swin": "-"},
    "02_pubmedclip_only":   {"RN50": "-", "ViT": "+", "Swin": "-"},
    "03_swin_only":         {"RN50": "-", "ViT": "-", "Swin": "+"},
    "04_biovil_pubmedclip": {"RN50": "+", "ViT": "+", "Swin": "-"},
    "05_biovil_swin":       {"RN50": "+", "ViT": "-", "Swin": "+"},
    "06_pubmedclip_swin":   {"RN50": "-", "ViT": "+", "Swin": "+"},
    "07_all_three":         {"RN50": "+", "ViT": "+", "Swin": "+"},
}


def load_jsonl(path):
    preds, refs = [], []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            preds.append(obj["pred"])
            refs.append(obj["ref"])
    return preds, refs


def load_bertscore():
    summary_path = INPUT_DIR / "summary.json"
    with open(summary_path) as f:
        data = json.load(f)
    return {r["Run"]: r["BERTScore"] for r in data["runs"] if r["BERTScore"] != "MISSING"}


def compute_radgraph(preds, refs):
    from radgraph import F1RadGraph

    f1radgraph = F1RadGraph(reward_level="all", model_type="radgraph-xl")
    valid_preds, valid_refs = [], []
    for p, r in zip(preds, refs):
        if p.strip() and r.strip():
            valid_preds.append(p)
            valid_refs.append(r)

    if not valid_preds:
        return {"entity_f1": 0.0, "relation_f1": 0.0, "overall_f1": 0.0}

    mean_reward, _, _, _ = f1radgraph(hyps=valid_preds, refs=valid_refs)
    return {
        "entity_f1": round(float(mean_reward[0]), 4),
        "relation_f1": round(float(mean_reward[1]), 4),
        "overall_f1": round(float(mean_reward[2]), 4),
    }


def compute_radcliq(bertscore, radgraph_overall):
    return round(np.sqrt((1 - bertscore) ** 2 + (1 - radgraph_overall) ** 2), 4)


def save_metric_dir(metric_name, run_results, output_base):
    metric_dir = output_base / metric_name
    metric_dir.mkdir(parents=True, exist_ok=True)
    summary = {"metric": metric_name, "llm": "lmsys/vicuna-7b-v1.3", "sample_limit": 200, "runs": []}
    for run_name in RUNS:
        if run_name not in run_results:
            continue
        val = run_results[run_name]
        entry = {"run": run_name, **ENCODER_INFO[run_name]}
        entry.update(val if isinstance(val, dict) else {metric_name: val})
        summary["runs"].append(entry)
    with open(metric_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved {metric_dir / 'summary.json'}")


def main():
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    bertscores = load_bertscore()
    print(f"Loaded BERTScore for {len(bertscores)} runs")

    radgraph_results = {}
    radcliq_results = {}
    all_metrics = {}

    for run_name in RUNS:
        jsonl_path = INPUT_DIR / f"reports_vicuna_{run_name}.jsonl"
        if not jsonl_path.exists():
            print(f"WARNING: {jsonl_path} not found, skipping {run_name}")
            continue

        preds, refs = load_jsonl(jsonl_path)
        print(f"\n{'='*50}")
        print(f"RadGraph for {run_name} ({len(preds)} samples)")
        print(f"{'='*50}")

        rg = compute_radgraph(preds, refs)
        radgraph_results[run_name] = rg
        print(f"  Entity F1:   {rg['entity_f1']}")
        print(f"  Relation F1: {rg['relation_f1']}")
        print(f"  Overall F1:  {rg['overall_f1']}")

        bs = bertscores.get(run_name, 0.0)
        rq = compute_radcliq(bs, rg["overall_f1"])
        radcliq_results[run_name] = rq
        print(f"  BERTScore:   {bs}")
        print(f"  RadCliQ:     {rq}")

        all_metrics[run_name] = {**rg, "BERTScore": bs, "RadCliQ": rq}

    # Save per-metric directories
    save_metric_dir("RadGraph", radgraph_results, OUTPUT_BASE)
    save_metric_dir("RadCliQ", radcliq_results, OUTPUT_BASE)

    # Update per-run all_metrics.json with RadGraph + RadCliQ
    for run_name in all_metrics:
        run_dir = OUTPUT_BASE / run_name
        metrics_file = run_dir / "all_metrics.json"
        with open(metrics_file) as f:
            existing = json.load(f)
        existing["metrics"]["RadGraph_Entity"] = all_metrics[run_name]["entity_f1"]
        existing["metrics"]["RadGraph_Relation"] = all_metrics[run_name]["relation_f1"]
        existing["metrics"]["RadGraph_Overall"] = all_metrics[run_name]["overall_f1"]
        existing["metrics"]["RadCliQ"] = all_metrics[run_name]["RadCliQ"]
        with open(metrics_file, "w") as f:
            json.dump(existing, f, indent=2)

    # Build table
    rows = []
    for run_name in RUNS:
        if run_name not in all_metrics:
            continue
        m = all_metrics[run_name]
        rows.append({
            "Run": run_name,
            **ENCODER_INFO[run_name],
            "BERTScore": m["BERTScore"],
            "RadGraph_F1": m["overall_f1"],
            "RadGraph_Entity": m["entity_f1"],
            "RadGraph_Relation": m["relation_f1"],
            "RadCliQ": m["RadCliQ"],
        })

    table = pd.DataFrame(rows)

    # Merge with existing NLG metrics table
    nlg_csv = OUTPUT_BASE / "nlg_metrics_table.csv"
    if nlg_csv.exists():
        nlg_table = pd.read_csv(nlg_csv)
        merged = nlg_table
        for col in ["BERTScore", "RadGraph_F1", "RadGraph_Entity", "RadGraph_Relation", "RadCliQ"]:
            if col in table.columns:
                merged[col] = table[col].values
        merged.to_csv(nlg_csv, index=False)

    rg_csv = OUTPUT_BASE / "radgraph_radcliq_table.csv"
    table.to_csv(rg_csv, index=False)

    print("\n" + "=" * 100)
    print("NLP Clinical Metrics — META-CXR (Vicuna-7B + LoRA) — 200 test samples")
    print("=" * 100)
    print(table.to_string(index=False))

    # Upload to GCS
    print(f"\nUploading to {GCS_BASE}/ ...")
    for metric_name in ["RadGraph", "RadCliQ"]:
        os.system(f"gsutil -m cp -r {OUTPUT_BASE / metric_name} {GCS_BASE}/")
    for run_name in RUNS:
        run_dir = OUTPUT_BASE / run_name
        if run_dir.exists():
            os.system(f"gsutil -m cp -r {run_dir} {GCS_BASE}/")
    os.system(f"gsutil -m cp {rg_csv} {GCS_BASE}/radgraph_radcliq_table.csv")
    os.system(f"gsutil -m cp {nlg_csv} {GCS_BASE}/nlg_metrics_table.csv")
    print("Upload complete.")

    return table


if __name__ == "__main__":
    main()
