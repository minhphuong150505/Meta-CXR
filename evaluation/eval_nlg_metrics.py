#!/usr/bin/env python3
"""Compute BLEU-1/2/3/4, METEOR, ROUGE-L, CIDEr from existing pred/ref JSONL files.

Reads each reports_vicuna_*.jsonl → computes all NLG metrics → saves per-metric
directories → uploads to gs://meta-cxr-checkpoint/eval/
"""

import json
import os
import re
from collections import defaultdict
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

# Ensure NLTK wordnet available for METEOR
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)

INPUT_DIR = Path("/home/phuong/META-CXR/output/bertscore_vicuna")
OUTPUT_BASE = Path("/home/phuong/META-CXR/output/nlg_metrics")
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

smooth = SmoothingFunction().method1


def tokenize(text: str):
    return re.findall(r"\w+", str(text).lower())


def load_jsonl(path: Path):
    preds, refs = [], []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            preds.append(obj["pred"])
            refs.append(obj["ref"])
    return preds, refs


def compute_bleu(predictions, references):
    scores = {1: [], 2: [], 3: [], 4: []}
    for pred, ref in zip(predictions, references):
        pred_tok = tokenize(pred)
        ref_tok = [tokenize(ref)]
        if not pred_tok:
            for k in scores:
                scores[k].append(0.0)
            continue
        for n in range(1, 5):
            weights = tuple(1.0 / n if i < n else 0.0 for i in range(4))
            try:
                s = sentence_bleu(ref_tok, pred_tok, weights=weights, smoothing_function=smooth)
            except Exception:
                s = 0.0
            scores[n].append(s)
    return {k: float(np.mean(v)) for k, v in scores.items()}


def compute_meteor(predictions, references):
    vals = []
    for pred, ref in zip(predictions, references):
        pred_tok = tokenize(pred)
        ref_tok = tokenize(ref)
        if not pred_tok or not ref_tok:
            vals.append(0.0)
            continue
        try:
            vals.append(meteor_score([ref_tok], pred_tok))
        except Exception:
            vals.append(0.0)
    return float(np.mean(vals))


def compute_rougel(predictions, references):
    gts = {i: [ref] for i, ref in enumerate(references)}
    res = {i: [pred] for i, pred in enumerate(predictions)}
    try:
        _, scores = Rouge().compute_score(gts, res)
        return float(np.mean(scores)) if len(scores) > 0 else float("nan")
    except Exception:
        return float("nan")


def compute_cider(predictions, references):
    gts = {i: [ref] for i, ref in enumerate(references)}
    res = {i: [pred] for i, pred in enumerate(predictions)}
    try:
        score, _ = Cider().compute_score(gts, res)
        return float(score) if score else 0.0
    except Exception:
        return float("nan")


def process_run(run_name, preds, refs):
    print(f"\n{'='*50}")
    print(f"Computing metrics for {run_name} ({len(preds)} samples)")
    print(f"{'='*50}")

    results = {}
    results["bleu"] = compute_bleu(preds, refs)
    for n in range(1, 5):
        print(f"  BLEU-{n}: {results['bleu'][n]:.4f}")
    results["meteor"] = compute_meteor(preds, refs)
    print(f"  METEOR: {results['meteor']:.4f}")
    results["rougel"] = compute_rougel(preds, refs)
    print(f"  ROUGE-L: {results['rougel']:.4f}")
    results["cider"] = compute_cider(preds, refs)
    print(f"  CIDEr: {results['cider']:.4f}")
    return results


def save_metric_dir(metric_name, run_results, output_base):
    metric_dir = output_base / metric_name
    metric_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "metric": metric_name,
        "llm": "lmsys/vicuna-7b-v1.3",
        "sample_limit": 200,
        "runs": [],
    }
    for run_name in RUNS:
        val = run_results.get(run_name)
        if val is None:
            continue
        entry = {
            "run": run_name,
            **ENCODER_INFO[run_name],
            metric_name: round(val, 4),
        }
        summary["runs"].append(entry)

    with open(metric_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved {metric_dir / 'summary.json'}")


def build_table(all_results):
    rows = []
    metric_keys = [
        ("BLEU-1", "bleu1"),
        ("BLEU-2", "bleu2"),
        ("BLEU-3", "bleu3"),
        ("BLEU-4", "bleu4"),
        ("METEOR", "meteor"),
        ("ROUGE-L", "rougel"),
        ("CIDEr", "cider"),
    ]

    for run_name in RUNS:
        if run_name not in all_results:
            continue
        row = {"Run": run_name, **ENCODER_INFO[run_name]}
        for display_name, key in metric_keys:
            val = all_results[run_name].get(key, float("nan"))
            row[display_name] = round(val, 4) if not np.isnan(val) else "N/A"
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    all_results = defaultdict(dict)

    for run_name in RUNS:
        jsonl_path = INPUT_DIR / f"reports_vicuna_{run_name}.jsonl"
        if not jsonl_path.exists():
            print(f"WARNING: {jsonl_path} not found, skipping {run_name}")
            continue
        preds, refs = load_jsonl(jsonl_path)
        result = process_run(run_name, preds, refs)

        all_results[run_name]["bleu1"] = result["bleu"][1]
        all_results[run_name]["bleu2"] = result["bleu"][2]
        all_results[run_name]["bleu3"] = result["bleu"][3]
        all_results[run_name]["bleu4"] = result["bleu"][4]
        all_results[run_name]["meteor"] = result["meteor"]
        all_results[run_name]["rougel"] = result["rougel"]
        all_results[run_name]["cider"] = result["cider"]

        # Save per-run all-metrics file
        run_dir = OUTPUT_BASE / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "all_metrics.json", "w") as f:
            json.dump({
                "run": run_name,
                **ENCODER_INFO[run_name],
                "metrics": {
                    "BLEU-1": round(all_results[run_name]["bleu1"], 4),
                    "BLEU-2": round(all_results[run_name]["bleu2"], 4),
                    "BLEU-3": round(all_results[run_name]["bleu3"], 4),
                    "BLEU-4": round(all_results[run_name]["bleu4"], 4),
                    "METEOR": round(all_results[run_name]["meteor"], 4),
                    "ROUGE-L": round(all_results[run_name]["rougel"], 4),
                    "CIDEr": round(all_results[run_name]["cider"], 4),
                },
            }, f, indent=2)

    # Per-metric directories
    metric_sets = {
        "BLEU-1": {rn: all_results[rn]["bleu1"] for rn in all_results},
        "BLEU-2": {rn: all_results[rn]["bleu2"] for rn in all_results},
        "BLEU-3": {rn: all_results[rn]["bleu3"] for rn in all_results},
        "BLEU-4": {rn: all_results[rn]["bleu4"] for rn in all_results},
        "METEOR": {rn: all_results[rn]["meteor"] for rn in all_results},
        "ROUGE-L": {rn: all_results[rn]["rougel"] for rn in all_results},
        "CIDEr": {rn: all_results[rn]["cider"] for rn in all_results},
    }
    for metric_name, run_results in metric_sets.items():
        save_metric_dir(metric_name, run_results, OUTPUT_BASE)

    # Build & save table
    table = build_table(all_results)
    csv_path = OUTPUT_BASE / "nlg_metrics_table.csv"
    table.to_csv(csv_path, index=False)
    print(f"\nSaved CSV: {csv_path}")

    # Overall summary
    summary_path = OUTPUT_BASE / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "llm": "lmsys/vicuna-7b-v1.3",
            "lora": "lora-vicuna-7b-report-20250621",
            "sample_limit": 200,
            "runs": {
                rn: {
                    **ENCODER_INFO[rn],
                    "BLEU-1": round(all_results[rn]["bleu1"], 4),
                    "BLEU-2": round(all_results[rn]["bleu2"], 4),
                    "BLEU-3": round(all_results[rn]["bleu3"], 4),
                    "BLEU-4": round(all_results[rn]["bleu4"], 4),
                    "METEOR": round(all_results[rn]["meteor"], 4),
                    "ROUGE-L": round(all_results[rn]["rougel"], 4),
                    "CIDEr": round(all_results[rn]["cider"], 4),
                }
                for rn in all_results
            },
        }, f, indent=2)

    print("\n" + "=" * 110)
    print("NLG Metrics Results — META-CXR (Vicuna-7B + LoRA) — 200 test samples")
    print("=" * 110)
    print(table.to_string(index=False))

    # Upload to GCS
    print(f"\nUploading to {GCS_BASE}/ ...")
    for metric_name in metric_sets:
        local_dir = OUTPUT_BASE / metric_name
        os.system(f"gsutil -m cp -r {local_dir} {GCS_BASE}/")
    for run_name in RUNS:
        run_dir = OUTPUT_BASE / run_name
        if run_dir.exists():
            os.system(f"gsutil -m cp -r {run_dir} {GCS_BASE}/")
    os.system(f"gsutil -m cp {csv_path} {GCS_BASE}/nlg_metrics_table.csv")
    os.system(f"gsutil -m cp {summary_path} {GCS_BASE}/nlg_summary.json")
    print("Upload complete.")

    return table


if __name__ == "__main__":
    main()
