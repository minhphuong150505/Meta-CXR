#!/usr/bin/env python3
"""Finetune MedGemma-1.5-4b with QLoRA (4-bit NF4) on the MIMIC-CXR train split,
using checkpoint 07 (07_all_three) Q-Former embeddings injected as soft image
tokens. Evaluates the finetuned model on the test split and uploads reports +
metrics to GCS.

Reuses the building blocks (stage-1 extraction, VariantLLM, evaluate_variant)
from train_eval_figure9_llm_variants_200.py. Only the MedGemma "fine" variant is
trained/evaluated here (no base / instruction variants).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # training/
import train_eval_figure9_llm_variants_200 as fig9  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-root", default="/mnt/meta-cxr-checkpoint")
    p.add_argument("--output-dir", default="/home/phuong/META-CXR/output/medgemma_qlora")
    p.add_argument("--gcs-output", default="gs://meta-cxr-checkpoint/eval/MedGemma_QLoRA")
    p.add_argument("--train-limit", type=int, default=100000)  # >= full train -> all train rows
    p.add_argument("--eval-limit", type=int, default=300)
    p.add_argument("--train-epochs", type=int, default=1)
    p.add_argument("--train-lr", type=float, default=2e-4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=160)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--no-upload", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    fig9.set_seed(fig9.SEED)
    root = Path(args.output_dir)
    eval_dir = root / "eval"
    adapter_dir = root / "adapters" / "medgemma_qlora_fine"
    root.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root = Path(args.checkpoint_root)

    # Stage 1: Q-Former embeddings (checkpoint 07) for train + test.
    print(f"[stage1] extracting train (limit={args.train_limit})", flush=True)
    train_records = fig9.build_stage1_records(
        checkpoint_root, root, "train", args.train_limit, args.num_workers
    )
    print(f"[stage1] extracting test (limit={args.eval_limit})", flush=True)
    test_records = fig9.build_stage1_records(
        checkpoint_root, root, "test", args.eval_limit, args.num_workers
    )
    print(f"[data] train={len(train_records)} test={len(test_records)}", flush=True)

    # Stage 2: QLoRA finetune MedGemma on train.
    if not (adapter_dir / "adapter_config.json").exists():
        print(f"[train] QLoRA finetuning MedGemma -> {adapter_dir}", flush=True)
        llm = fig9.VariantLLM("medgemma", train_adapter=True, quantize_4bit=True)
        llm.train_fine(train_records, adapter_dir, args.train_epochs, args.train_lr, args.grad_accum)
        del llm
        fig9.clear_memory()
    else:
        print(f"[train] reusing existing adapter at {adapter_dir}", flush=True)

    # Eval the finetuned model on the test split.
    print("[eval] generating + scoring on test", flush=True)
    llm = fig9.VariantLLM("medgemma", adapter=adapter_dir, quantize_4bit=True)
    llm.load_img_proj_if_present(adapter_dir)
    metrics = fig9.evaluate_variant(
        "medgemma", "qlora_fine", llm, test_records, eval_dir, args.max_new_tokens, "fine"
    )
    del llm
    fig9.clear_memory()

    summary = {
        "model": fig9.MEDGEMMA_MODEL_ID,
        "method": "QLoRA 4-bit NF4 + trainable img_proj, Q-Former soft tokens",
        "checkpoint": "07_all_three",
        "train_samples": len(train_records),
        "test_samples": len(test_records),
        "train_epochs": args.train_epochs,
        "adapter_dir": str(adapter_dir),
        "metrics": metrics,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[done] metrics:", json.dumps(metrics, indent=2), flush=True)

    if not args.no_upload:
        print(f"[upload] -> {args.gcs_output}", flush=True)
        fig9.upload_dir(root, args.gcs_output)
    print("PIPELINE_DONE", flush=True)


if __name__ == "__main__":
    main()
