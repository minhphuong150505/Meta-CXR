#!/usr/bin/env python3
"""Compute NLG and clinical metrics for MedGemma Q-Former report JSONL files."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)

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
    "01_biovil_only": {"RN50": "+", "ViT": "-", "Swin": "-"},
    "02_pubmedclip_only": {"RN50": "-", "ViT": "+", "Swin": "-"},
    "03_swin_only": {"RN50": "-", "ViT": "-", "Swin": "+"},
    "04_biovil_pubmedclip": {"RN50": "+", "ViT": "+", "Swin": "-"},
    "05_biovil_swin": {"RN50": "+", "ViT": "-", "Swin": "+"},
    "06_pubmedclip_swin": {"RN50": "-", "ViT": "+", "Swin": "+"},
    "07_all_three": {"RN50": "+", "ViT": "+", "Swin": "+"},
}

SMOOTH = SmoothingFunction().method1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="/home/phuong/META-CXR/output/medgemma_qformer")
    parser.add_argument("--output-base", default="/home/phuong/META-CXR/output/medgemma_qformer/metrics")
    parser.add_argument("--gcs-output", default="gs://meta-cxr-checkpoint/eval/MedGemma_QFormer/metrics")
    parser.add_argument("--skip-radgraph", action="store_true")
    parser.add_argument("--skip-upload", action="store_true")
    return parser.parse_args()


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


def load_bertscore(input_dir: Path):
    summary_path = input_dir / "summary.json"
    if not summary_path.exists():
        return {}
    with open(summary_path) as f:
        data = json.load(f)
    return {r["Run"]: r["BERTScore"] for r in data.get("runs", []) if r.get("BERTScore") != "MISSING"}


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
                score = sentence_bleu(ref_tok, pred_tok, weights=weights, smoothing_function=SMOOTH)
            except Exception:
                score = 0.0
            scores[n].append(score)
    return {k: float(np.mean(v)) for k, v in scores.items()}


def compute_meteor(predictions, references):
    values = []
    for pred, ref in zip(predictions, references):
        pred_tok = tokenize(pred)
        ref_tok = tokenize(ref)
        if not pred_tok or not ref_tok:
            values.append(0.0)
            continue
        try:
            values.append(meteor_score([ref_tok], pred_tok))
        except Exception:
            values.append(0.0)
    return float(np.mean(values))


def compute_rougel(predictions, references):
    gts = {i: [ref] for i, ref in enumerate(references)}
    res = {i: [pred] for i, pred in enumerate(predictions)}
    try:
        _score, scores = Rouge().compute_score(gts, res)
        return float(np.mean(scores)) if len(scores) > 0 else float("nan")
    except Exception:
        return float("nan")


def compute_cider(predictions, references):
    gts = {i: [ref] for i, ref in enumerate(references)}
    res = {i: [pred] for i, pred in enumerate(predictions)}
    try:
        score, _scores = Cider().compute_score(gts, res)
        return float(score) if score else 0.0
    except Exception:
        return float("nan")


def compute_radgraph(predictions, references):
    from radgraph import F1RadGraph

    f1radgraph = F1RadGraph(reward_level="all", model_type="radgraph-xl")
    valid_preds, valid_refs = [], []
    for pred, ref in zip(predictions, references):
        if pred.strip() and ref.strip():
            valid_preds.append(pred)
            valid_refs.append(ref)
    if not valid_preds:
        return {"entity_f1": 0.0, "relation_f1": 0.0, "overall_f1": 0.0}
    mean_reward, _, _, _ = f1radgraph(hyps=valid_preds, refs=valid_refs)
    return {
        "entity_f1": round(float(mean_reward[0]), 4),
        "relation_f1": round(float(mean_reward[1]), 4),
        "overall_f1": round(float(mean_reward[2]), 4),
    }


def compute_radcliq(bertscore, radgraph_overall):
    if isinstance(bertscore, str):
        return "N/A"
    return round(float(np.sqrt((1 - bertscore) ** 2 + (1 - radgraph_overall) ** 2)), 4)


def process_nlg(preds, refs):
    result = {}
    bleu = compute_bleu(preds, refs)
    result["bleu1"] = bleu[1]
    result["bleu2"] = bleu[2]
    result["bleu3"] = bleu[3]
    result["bleu4"] = bleu[4]
    result["meteor"] = compute_meteor(preds, refs)
    result["rougel"] = compute_rougel(preds, refs)
    result["cider"] = compute_cider(preds, refs)
    return result


def save_metric_dir(metric_name, run_results, output_base: Path):
    metric_dir = output_base / metric_name
    metric_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "metric": metric_name,
        "llm": "google/medgemma-1.5-4b-it",
        "lora": "DeepRadiology/medgemma1.5-CXR",
        "image_conditioning": "Q-Former embeddings injected at 32 <IMG> token positions",
        "runs": [],
    }
    for run_name in RUNS:
        if run_name not in run_results:
            continue
        val = run_results[run_name]
        entry = {"run": run_name, **ENCODER_INFO[run_name]}
        entry.update(val if isinstance(val, dict) else {metric_name: val})
        summary["runs"].append(entry)
    with open(metric_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def build_nlg_table(all_results):
    rows = []
    for run_name in RUNS:
        if run_name not in all_results:
            continue
        metrics = all_results[run_name]
        row = {"Run": run_name, **ENCODER_INFO[run_name]}
        for display, key in [
            ("BLEU-1", "bleu1"),
            ("BLEU-2", "bleu2"),
            ("BLEU-3", "bleu3"),
            ("BLEU-4", "bleu4"),
            ("METEOR", "meteor"),
            ("ROUGE-L", "rougel"),
            ("CIDEr", "cider"),
            ("BERTScore", "BERTScore"),
            ("RadGraph_F1", "RadGraph_Overall"),
            ("RadCliQ", "RadCliQ"),
        ]:
            val = metrics.get(key)
            if val is None:
                row[display] = "N/A"
            elif isinstance(val, str):
                row[display] = val
            else:
                row[display] = round(float(val), 4) if not np.isnan(float(val)) else "N/A"
        rows.append(row)
    return pd.DataFrame(rows)


def upload_dir(local_dir: Path, gcs_output: str):
    cmd = f"gcloud storage cp -r {local_dir}/* {gcs_output}/ --quiet"
    print(f">>> Uploading: {cmd}")
    rc = os.system(cmd)
    if rc != 0:
        raise RuntimeError(f"Upload failed with rc={rc}: {cmd}")


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_base = Path(args.output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    bertscores = load_bertscore(input_dir)
    all_results = defaultdict(dict)
    radgraph_results = {}
    radcliq_results = {}

    for run_name in RUNS:
        jsonl_path = input_dir / f"reports_medgemma_qformer_{run_name}.jsonl"
        if not jsonl_path.exists():
            print(f"WARNING: {jsonl_path} not found, skipping {run_name}")
            continue
        preds, refs = load_jsonl(jsonl_path)
        print(f"\n{'=' * 70}")
        print(f"Metrics for {run_name} ({len(preds)} samples)")
        print(f"{'=' * 70}")

        nlg = process_nlg(preds, refs)
        all_results[run_name].update(nlg)
        all_results[run_name]["BERTScore"] = bertscores.get(run_name, "N/A")
        print(
            "  "
            + ", ".join(
                [
                    f"BLEU-1={nlg['bleu1']:.4f}",
                    f"BLEU-4={nlg['bleu4']:.4f}",
                    f"METEOR={nlg['meteor']:.4f}",
                    f"ROUGE-L={nlg['rougel']:.4f}",
                    f"CIDEr={nlg['cider']:.4f}",
                ]
            )
        )

        if not args.skip_radgraph:
            rg = compute_radgraph(preds, refs)
            radgraph_results[run_name] = rg
            all_results[run_name]["RadGraph_Entity"] = rg["entity_f1"]
            all_results[run_name]["RadGraph_Relation"] = rg["relation_f1"]
            all_results[run_name]["RadGraph_Overall"] = rg["overall_f1"]
            rq = compute_radcliq(all_results[run_name]["BERTScore"], rg["overall_f1"])
            radcliq_results[run_name] = rq
            all_results[run_name]["RadCliQ"] = rq
            print(f"  RadGraph overall={rg['overall_f1']:.4f}, RadCliQ={rq}")

        run_dir = output_base / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "all_metrics.json", "w") as f:
            json.dump(
                {
                    "run": run_name,
                    **ENCODER_INFO[run_name],
                    "metrics": build_nlg_table({run_name: all_results[run_name]}).iloc[0].to_dict(),
                },
                f,
                indent=2,
            )

    metric_sets = {
        "BLEU-1": {rn: all_results[rn]["bleu1"] for rn in all_results},
        "BLEU-2": {rn: all_results[rn]["bleu2"] for rn in all_results},
        "BLEU-3": {rn: all_results[rn]["bleu3"] for rn in all_results},
        "BLEU-4": {rn: all_results[rn]["bleu4"] for rn in all_results},
        "METEOR": {rn: all_results[rn]["meteor"] for rn in all_results},
        "ROUGE-L": {rn: all_results[rn]["rougel"] for rn in all_results},
        "CIDEr": {rn: all_results[rn]["cider"] for rn in all_results},
    }
    if not args.skip_radgraph:
        metric_sets["RadGraph"] = radgraph_results
        metric_sets["RadCliQ"] = radcliq_results

    for metric_name, values in metric_sets.items():
        save_metric_dir(metric_name, values, output_base)

    table = build_nlg_table(all_results)
    nlg_csv = output_base / "nlg_metrics_table_medgemma_qformer.csv"
    table.to_csv(nlg_csv, index=False)

    summary_path = output_base / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "llm": "google/medgemma-1.5-4b-it",
                "lora": "DeepRadiology/medgemma1.5-CXR",
                "image_conditioning": "Q-Former embeddings injected at 32 <IMG> token positions",
                "runs": {rn: dict(all_results[rn]) for rn in all_results},
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 110)
    print("NLG/Clinical Metrics - MedGemma + LoRA + Q-Former injection")
    print("=" * 110)
    print(table.to_string(index=False))

    if not args.skip_upload:
        upload_dir(output_base, args.gcs_output)
        print(f">>> Uploaded to {args.gcs_output}/")


if __name__ == "__main__":
    main()
