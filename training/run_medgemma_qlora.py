#!/usr/bin/env python3
"""Full-data MedGemma QLoRA pipeline for META-CXR Stage 2.

The primary mode injects Stage-1 Q-Former embeddings as trainable soft tokens.
``--image-mode native`` is the MedGemma image-tower ablation and ``both`` runs
the two experiments sequentially on one GPU. Validation chooses checkpoints;
the test cohort is generated exactly once after training.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_eval_figure9_llm_variants_200 as fig9  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", default="checkpoints")
    parser.add_argument("--stage1-run", default="mimic_cxr_full_l4_blip2")
    parser.add_argument(
        "--stage1-config",
        type=Path,
        default=fig9.PROJECT_DIR / "pretraining/configs/mimic_cxr_full_l4.yaml",
        help="Config used to construct the Stage-1 model.",
    )
    parser.add_argument(
        "--stage1-checkpoint",
        type=Path,
        help="Direct checkpoint_best.pth override; otherwise uses <checkpoint-root>/<run>/checkpoint_best.pth.",
    )
    parser.add_argument("--output-dir", default="training/outputs/medgemma_qlora_full")
    parser.add_argument(
        "--gcs-output",
        help="Opt-in private gs:// destination. Omit to keep all outputs local.",
    )
    parser.add_argument(
        "--threshold-path",
        type=Path,
        help="Optional Stage-1 validation-calibrated thresholds; default is image-only argmax.",
    )
    parser.add_argument(
        "--image-mode",
        choices=["qformer", "native", "both"],
        default="qformer",
        help="Q-Former soft-token primary model, native MedGemma image ablation, or both.",
    )
    parser.add_argument("--train-limit", type=int, default=0, help="0 uses the complete train cohort")
    parser.add_argument("--val-limit", type=int, default=0, help="0 uses the complete validation cohort")
    parser.add_argument("--test-limit", type=int, default=0, help="0 uses the complete held-out test cohort")
    parser.add_argument(
        "--eval-limit",
        type=int,
        dest="test_limit",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--val-generation-limit", type=int, default=300)
    parser.add_argument("--train-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lora-lr", type=float, default=1e-4)
    parser.add_argument("--projector-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2, help="Stage-1 image-loading workers")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--no-upload", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.grad_accum < 1:
        parser.error("--batch-size and --grad-accum must be positive")
    if args.train_epochs < 1:
        parser.error("--train-epochs must be positive")
    if not 0 <= args.warmup_ratio < 1:
        parser.error("--warmup-ratio must be in [0, 1)")
    return args


def deterministic_subset(records: list[dict], limit: int, seed: int) -> list[dict]:
    if limit <= 0 or limit >= len(records):
        return records
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(records)), limit))
    return [records[index] for index in indices]


def resumable_adapter(path: Path, image_mode: str) -> bool:
    weights = (path / "adapter_model.safetensors").is_file() or (path / "adapter_model.bin").is_file()
    projector_ok = image_mode != "qformer" or (path / "img_proj.pt").is_file()
    return weights and projector_ok and (path / "adapter_config.json").is_file() and (path / "trainer_state.pt").is_file()


def upload_safe_run(root: Path, adapter_dirs: list[Path], gcs_output: str) -> None:
    fig9.assert_private_gcs_destination(gcs_output)
    for path in sorted((root / "eval").glob("*.json*")):
        fig9.upload_path(path, f"{gcs_output}/eval")
    for name in ("summary.json", "run_manifest.json"):
        path = root / name
        if path.is_file():
            fig9.upload_path(path, gcs_output)
    for adapter_dir in adapter_dirs:
        # Explicit allow-list: never traverse the run root or nested local
        # checkpoints/cache. These files contain no MIMIC rows.
        for name in (
            "adapter_config.json",
            "adapter_model.safetensors",
            "adapter_model.bin",
            "img_proj.pt",
            "meta.json",
            "manifest.json",
            "trainer_state.pt",
        ):
            path = adapter_dir / name
            if path.is_file():
                fig9.upload_path(path, f"{gcs_output}/adapters/{adapter_dir.name}")


def train_mode(
    args: argparse.Namespace,
    image_mode: str,
    train_records: list[dict],
    val_records: list[dict],
    test_records: list[dict],
    root: Path,
) -> tuple[dict, Path]:
    adapter_dir = root / "adapters" / f"medgemma_qlora_{image_mode}"
    last_dir = adapter_dir / "checkpoints" / "last"
    training_summary: dict = {}
    complete = fig9.adapter_is_complete(adapter_dir, image_mode)
    if args.force_retrain or not complete:
        resume_dir = args.resume_from
        if (
            resume_dir is None
            and not args.force_retrain
            and resumable_adapter(last_dir, image_mode)
        ):
            resume_dir = last_dir
        if resume_dir is not None and not resumable_adapter(Path(resume_dir), image_mode):
            raise RuntimeError(f"incomplete --resume-from checkpoint: {resume_dir}")
        print(f"[train:{image_mode}] adapter -> {adapter_dir}", flush=True)
        llm = fig9.VariantLLM(
            "medgemma",
            adapter=resume_dir,
            train_adapter=True,
            quantize_4bit=True,
            image_mode=image_mode,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
        )
        llm.load_img_proj_if_present(resume_dir)
        training_summary = llm.train_fine(
            train_records,
            adapter_dir,
            args.train_epochs,
            grad_accum=args.grad_accum,
            val_records=val_records,
            batch_size=args.batch_size,
            lora_lr=args.lora_lr,
            projector_lr=args.projector_lr,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            max_grad_norm=args.max_grad_norm,
            max_length=args.max_length,
            patience=args.patience,
            resume_state=resume_dir,
        )
        del llm
        fig9.clear_memory()
    else:
        print(f"[train:{image_mode}] reusing complete adapter {adapter_dir}", flush=True)
        training_summary = json.loads(
            (adapter_dir / "manifest.json").read_text(encoding="utf-8")
        ).get("training_config", {})
    if not fig9.adapter_is_complete(adapter_dir, image_mode):
        raise RuntimeError(f"training did not produce a complete adapter: {adapter_dir}")

    llm = fig9.VariantLLM(
        "medgemma",
        adapter=adapter_dir,
        quantize_4bit=True,
        image_mode=image_mode,
    )
    llm.load_img_proj_if_present(adapter_dir)
    val_eval_records = deterministic_subset(val_records, args.val_generation_limit, fig9.SEED + 1)
    val_cohort, _ = fig9.stage1_cohort_fingerprint(
        Path(args.checkpoint_root), "val", args.val_limit
    )
    val_metrics = fig9.evaluate_variant(
        "medgemma",
        "qlora_validation",
        llm,
        val_eval_records,
        root / "eval",
        args.max_new_tokens,
        "fine",
        cohort_id=fig9.stable_fingerprint(
            {"full_val_cohort": val_cohort, "sample_keys": [r["sample_key"] for r in val_eval_records]}
        ),
    )
    test_metrics = None
    if not args.skip_test:
        test_cohort, _ = fig9.stage1_cohort_fingerprint(
            Path(args.checkpoint_root), "test", args.test_limit
        )
        test_metrics = fig9.evaluate_variant(
            "medgemma",
            "qlora_test",
            llm,
            test_records,
            root / "eval",
            args.max_new_tokens,
            "fine",
            cohort_id=test_cohort,
        )
    del llm
    fig9.clear_memory()
    return (
        {
            "image_mode": image_mode,
            "method": (
                "QLoRA NF4 + Q-Former soft tokens + trainable projector"
                if image_mode == "qformer"
                else "QLoRA NF4 + native MedGemma image tower"
            ),
            "adapter": str(adapter_dir),
            "training": training_summary,
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
        },
        adapter_dir,
    )


def main() -> None:
    args = parse_args()
    fig9.RUN_NAME = args.stage1_run
    fig9.STAGE1_CONFIG_PATH_OVERRIDE = args.stage1_config
    fig9.STAGE1_CHECKPOINT_PATH_OVERRIDE = args.stage1_checkpoint
    fig9.THRESHOLDS = fig9.load_thresholds(args.threshold_path)
    fig9.set_seed(fig9.SEED)
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "eval").mkdir(parents=True, exist_ok=True)
    checkpoint_root = Path(args.checkpoint_root)
    if not Path(args.stage1_config).is_file():
        raise FileNotFoundError(f"Stage-1 config not found: {args.stage1_config}")
    modes = ["qformer", "native"] if args.image_mode == "both" else [args.image_mode]
    requires_stage1 = "qformer" in modes
    resolved_stage1_checkpoint = fig9.stage1_checkpoint_path(checkpoint_root)
    if requires_stage1 and not resolved_stage1_checkpoint.is_file():
        raise FileNotFoundError(
            f"Stage-1 checkpoint not found: {resolved_stage1_checkpoint}. "
            "Pass --stage1-checkpoint or mount --checkpoint-root."
        )

    print(f"[stage1] train limit={args.train_limit or 'all'}", flush=True)
    train_records = fig9.build_stage1_records(
        checkpoint_root,
        root,
        "train",
        args.train_limit,
        args.num_workers,
        include_stage1_features=requires_stage1,
    )
    print(f"[stage1] validation limit={args.val_limit or 'all'}", flush=True)
    val_records = fig9.build_stage1_records(
        checkpoint_root,
        root,
        "val",
        args.val_limit,
        args.num_workers,
        include_stage1_features=requires_stage1,
    )
    print(f"[stage1] held-out test limit={args.test_limit or 'all'}", flush=True)
    test_records = fig9.build_stage1_records(
        checkpoint_root,
        root,
        "test",
        args.test_limit,
        args.num_workers,
        include_stage1_features=requires_stage1,
    )
    print(
        f"[data] train={len(train_records)} val={len(val_records)} test={len(test_records)}",
        flush=True,
    )

    results: dict[str, dict] = {}
    adapter_dirs: list[Path] = []
    for image_mode in modes:
        results[image_mode], adapter_dir = train_mode(
            args, image_mode, train_records, val_records, test_records, root
        )
        adapter_dirs.append(adapter_dir)

    summary = {
        "schema_version": fig9.SCHEMA_VERSION,
        "model": fig9.MEDGEMMA_MODEL_ID,
        "primary_model": "qformer" if "qformer" in modes else modes[0],
        "ablation": "native MedGemma image tower" if "native" in modes else None,
        "stage1_checkpoint": fig9.RUN_NAME if requires_stage1 else None,
        "target_section": "FINDINGS",
        "selection_split": "val",
        "test_used_for_selection": False,
        "train_samples": len(train_records),
        "val_samples": len(val_records),
        "test_samples": len(test_records),
        "results": results,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    run_manifest = {
        "schema_version": fig9.SCHEMA_VERSION,
        "status": "complete",
        "private_data_cache": ".sensitive_stage1_cache (local only; excluded from upload)",
        "uploaded_artifacts_contain_references": False,
        "image_modes": modes,
    }
    (root / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    print("[done]", json.dumps(summary, indent=2), flush=True)

    if args.gcs_output and not args.no_upload:
        print(f"[upload-safe-artifacts] -> {args.gcs_output}", flush=True)
        upload_safe_run(root, adapter_dirs, args.gcs_output)
    print("PIPELINE_DONE", flush=True)


if __name__ == "__main__":
    main()
