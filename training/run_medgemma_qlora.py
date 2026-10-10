#!/usr/bin/env python3
"""Full-data MedGemma QLoRA pipeline for META-CXR Stage 2.

The default pipeline is ``medgemma_direct``: MedGemma's own image tower and
multimodal projector, with no Stage-1 checkpoint, no Q-Former, no MHCAC and no
structured findings in the prompt. ``meta_cxr_qformer`` is the hybrid ablation
that injects Stage-1 Q-Former embeddings as trainable soft tokens, and
``both_for_ablation`` runs the two sequentially on one GPU.

Validation chooses checkpoints; the test cohort is generated exactly once after
training.

Single-process, single-GPU by design. Multi-GPU would require DDP, which this
script does not implement.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# VariantLLM and the evaluation helpers still live in the Figure-9 module, so
# this pulls in torch/transformers/peft -- which medgemma_direct needs anyway.
# It no longer pulls in LAVIS, the vision encoders, the Q-Former or MHCAC:
# those are confined to training/stage1/lavis_loader.py and imported only from
# inside build_stage1_records(). Enforced by tests/test_native_independence.py.
import train_eval_figure9_llm_variants_200 as fig9  # noqa: E402
from dataio.manifest import (  # noqa: E402
    DEFAULT_SECTION_MODE,
    FINDINGS_AND_IMPRESSION,
    SECTION_MODES,
    assert_no_leakage,
    build_records,
)
from pipeline_modes import (  # noqa: E402
    CHOICES,
    DEFAULT_PIPELINE_MODE,
    DEFAULT_PROMPT_STYLE,
    LEGACY_IMAGE_MODE_ALIASES,
    PROMPT_STYLES,
    PipelineMode,
    check_adapter_prompt_style,
    resolve_lora_size,
    resolve_pipeline_modes,
    validate_prompt_style,
)
from pipeline_modes import requires_stage1 as modes_require_stage1  # noqa: E402
from run_context import Stage1Context  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", default="checkpoints")
    parser.add_argument("--stage1-run", default="mimic_cxr_full_blip2")
    parser.add_argument(
        "--stage1-config",
        type=Path,
        default=fig9.PROJECT_DIR / "pretraining/configs/mimic_cxr_full.yaml",
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
        help="Per-class thresholds from scripts/calibrate_thresholds.py (fitted on "
             "Stage-1 validation); required by --cue-rule paper_thresholds. With "
             "--cue-rule cutpoints: a file from calibrate_thresholds.py --rule cutpoints.",
    )
    parser.add_argument(
        "--finding-tokens", choices=fig9.FINDING_TOKEN_MODES, default=fig9.FINDING_TOKENS_OFF,
        help="EXPERIMENTAL, default off. Feed Stage-1's per-finding three-class "
             "probabilities to Stage 2 as 13 learnable tokens instead of (or "
             "alongside) text cues. Requires a Stage-1 pipeline mode and a guided "
             "--prompt-config.",
    )
    parser.add_argument(
        "--cue-rule", choices=fig9.CUE_RULES, default=fig9.DEFAULT_CUE_RULE,
        help="How MHCAC's three-class predictions become P/N/U cues, identical to "
             "generation --cue-rule. argmax (default) | paper_thresholds (needs "
             "--threshold-path from scripts/calibrate_thresholds.py) | none | "
             "cutpoints (Stage-1 headline rule; --threshold-path from "
             "calibrate_thresholds.py --rule cutpoints). "
             "Abstaining rules require a matching guided --prompt-config.",
    )
    parser.add_argument(
        "--pipeline-mode",
        choices=list(CHOICES),
        default=DEFAULT_PIPELINE_MODE,
        help=(
            "medgemma_direct (default): native MedGemma image tower, no Stage-1. "
            "meta_cxr_qformer: Q-Former soft-token hybrid ablation. "
            "both_for_ablation: primary then ablation."
        ),
    )
    parser.add_argument(
        "--image-mode",
        choices=["qformer", "native", "both"],
        dest="legacy_image_mode",
        help="Deprecated alias for --pipeline-mode; kept so existing runbooks work.",
    )
    parser.add_argument(
        "--section-mode",
        choices=list(SECTION_MODES),
        default=DEFAULT_SECTION_MODE,
        help="Report sections to train and evaluate on (default: findings_and_impression).",
    )
    parser.add_argument(
        "--prompt-config",
        type=Path,
        help=(
            "Opt-in stage2.prompts YAML (e.g. configs/stage2_prompt_v2.yaml). Omit to "
            "keep the exact legacy prompt. Applies to a mode only when the config's "
            "visual_mode matches that mode's image_mode; other modes stay legacy."
        ),
    )
    parser.add_argument(
        "--prompt-style",
        choices=PROMPT_STYLES,
        default=DEFAULT_PROMPT_STYLE,
        help=(
            "Wording of the legacy prompt (no --prompt-config). fine (default): "
            "the short instruction every recorded legacy run used. paper: the "
            "META-CXR paper's instruction verbatim; requires --pipeline-mode "
            "meta_cxr_qformer_with_mhcac_prompt, i.e. no native image, only the "
            "32 Q-Former soft tokens plus MHCAC P/N/U text."
        ),
    )
    parser.add_argument(
        "--require-image",
        action="store_true",
        help="Skip manifest rows whose JPG is absent instead of failing at load time.",
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
    # 0 keeps the historical behaviour: the adapter is written only at epoch
    # end. Set it for long single-epoch runs -- a full-cohort epoch is ~70 h,
    # and this machine has hung without warning before.
    parser.add_argument("--save-every-updates", type=int, default=0,
                        help="write a recovery adapter every N optimizer "
                             "updates. Marked status=in_progress; resuming "
                             "from it still restarts the epoch.")
    parser.add_argument("--lora-lr", type=float, default=1e-4)
    parser.add_argument("--projector-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=0,
        help="0 auto-sizes: 512 for findings_and_impression, 256 for a single section.",
    )
    parser.add_argument("--patience", type=int, default=1)
    # None = the prompt style's default: r=16/alpha=32, or the paper's
    # r=8/alpha=16 under --prompt-style paper (pipeline_modes.resolve_lora_size).
    parser.add_argument("--lora-rank", type=int, default=None,
                        help="default 16; 8 under --prompt-style paper (as the paper)")
    parser.add_argument("--lora-alpha", type=int, default=None,
                        help="default 32; 16 under --prompt-style paper (as the paper)")
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
    if args.legacy_image_mode:
        if args.pipeline_mode != DEFAULT_PIPELINE_MODE:
            parser.error(
                "pass either --pipeline-mode or the deprecated --image-mode, not both"
            )
        args.pipeline_mode = LEGACY_IMAGE_MODE_ALIASES[args.legacy_image_mode]
        print(
            f"[deprecated] --image-mode {args.legacy_image_mode} "
            f"-> --pipeline-mode {args.pipeline_mode}",
            flush=True,
        )
    try:
        args.lora_rank, args.lora_alpha = resolve_lora_size(
            args.prompt_style, args.lora_rank, args.lora_alpha
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.max_new_tokens <= 0:
        args.max_new_tokens = (
            512 if args.section_mode == FINDINGS_AND_IMPRESSION else 256
        )
    return args


def deterministic_subset(records: list[dict], limit: int, seed: int) -> list[dict]:
    if limit <= 0 or limit >= len(records):
        return records
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(records)), limit))
    return [records[index] for index in indices]


def resumable_adapter(path: Path, image_mode: str, finding_tokens: str = "off") -> bool:
    weights = (path / "adapter_model.safetensors").is_file() or (path / "adapter_model.bin").is_file()
    # Any mode that carries soft tokens owns a trained img_proj; resuming
    # without it would silently restart the bridge from a fresh nn.Linear.
    projector_ok = image_mode not in fig9.SOFT_TOKEN_MODES or (path / "img_proj.pt").is_file()
    findings_ok = finding_tokens in (None, "off") or (path / "finding_tokens.pt").is_file()
    return (
        weights
        and projector_ok
        and findings_ok
        and (path / "adapter_config.json").is_file()
        and (path / "trainer_state.pt").is_file()
    )


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


def assert_adapter_prompt_style(adapter_dir: Path, prompt_style: str) -> None:
    """Refuse to reuse or resume an adapter trained under another prompt style."""
    manifest_path = Path(adapter_dir) / "manifest.json"
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    try:
        check_adapter_prompt_style(manifest, prompt_style)
    except ValueError as exc:
        raise SystemExit(f"{adapter_dir}: {exc}") from None


def train_mode(
    context: Stage1Context,
    args: argparse.Namespace,
    mode: PipelineMode,
    train_records: list[dict],
    val_records: list[dict],
    test_records: list[dict],
    root: Path,
    prompt_config=None,
) -> tuple[dict, Path]:
    image_mode = mode.image_mode
    # A prompt config applies only to a mode whose visual channel it targets; an
    # ablation run keeps the non-matching mode on the legacy prompt.
    mode_prompt_config = (
        prompt_config
        if prompt_config is not None and prompt_config.visual_mode.image_mode == image_mode
        else None
    )
    if prompt_config is not None and mode_prompt_config is None:
        print(
            f"[train:{mode.name}] prompt config visual_mode "
            f"{prompt_config.visual_mode.value} does not target image_mode "
            f"{image_mode!r}; using the legacy prompt for this mode",
            flush=True,
        )
    adapter_dir = root / "adapters" / f"medgemma_qlora_{mode.name}"
    last_dir = adapter_dir / "checkpoints" / "last"
    training_summary: dict = {}
    complete = fig9.adapter_is_complete(adapter_dir, image_mode, args.finding_tokens)
    if complete and not args.force_retrain:
        assert_adapter_prompt_style(adapter_dir, args.prompt_style)
    if args.force_retrain or not complete:
        resume_dir = args.resume_from
        if (
            resume_dir is None
            and not args.force_retrain
            and resumable_adapter(last_dir, image_mode, args.finding_tokens)
        ):
            resume_dir = last_dir
        if resume_dir is not None and not resumable_adapter(
            Path(resume_dir), image_mode, args.finding_tokens
        ):
            raise RuntimeError(f"incomplete --resume-from checkpoint: {resume_dir}")
        if resume_dir is not None:
            assert_adapter_prompt_style(Path(resume_dir), args.prompt_style)
        print(f"[train:{mode.name}] adapter -> {adapter_dir}", flush=True)
        llm = fig9.VariantLLM(
            "medgemma",
            adapter=resume_dir,
            train_adapter=True,
            quantize_4bit=True,
            image_mode=image_mode,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            prompt_config=mode_prompt_config,
            finding_tokens=args.finding_tokens,
            legacy_prompt_style=args.prompt_style,
        )
        llm.load_img_proj_if_present(resume_dir)
        if resume_dir is not None:
            llm.load_finding_encoder_if_present(resume_dir)
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
            save_every_updates=args.save_every_updates,
        )
        del llm
        fig9.clear_memory()
    else:
        print(f"[train:{mode.name}] reusing complete adapter {adapter_dir}", flush=True)
        training_summary = json.loads(
            (adapter_dir / "manifest.json").read_text(encoding="utf-8")
        ).get("training_config", {})
    if not fig9.adapter_is_complete(adapter_dir, image_mode, args.finding_tokens):
        raise RuntimeError(f"training did not produce a complete adapter: {adapter_dir}")

    llm = fig9.VariantLLM(
        "medgemma",
        adapter=adapter_dir,
        quantize_4bit=True,
        image_mode=image_mode,
        prompt_config=mode_prompt_config,
        finding_tokens=args.finding_tokens,
        legacy_prompt_style=args.prompt_style,
    )
    llm.load_img_proj_if_present(adapter_dir)
    llm.load_finding_encoder_if_present(adapter_dir)
    val_eval_records = deterministic_subset(val_records, args.val_generation_limit, fig9.SEED + 1)
    val_cohort, _ = fig9.stage1_cohort_fingerprint(
        context, Path(args.checkpoint_root), "val", args.val_limit,
        cue_rule=args.cue_rule, finding_tokens=args.finding_tokens,
    )
    val_metrics = fig9.evaluate_variant(
        "medgemma",
        f"{mode.name}_validation",
        llm,
        val_eval_records,
        root / "eval",
        args.max_new_tokens,
        args.prompt_style,
        section_mode=args.section_mode,
        context=context,
        cohort_id=fig9.stable_fingerprint(
            {"full_val_cohort": val_cohort, "sample_keys": [r["sample_key"] for r in val_eval_records]}
        ),
    )
    test_metrics = None
    if not args.skip_test:
        test_cohort, _ = fig9.stage1_cohort_fingerprint(
            context, Path(args.checkpoint_root), "test", args.test_limit,
            cue_rule=args.cue_rule, finding_tokens=args.finding_tokens,
        )
        test_metrics = fig9.evaluate_variant(
            "medgemma",
            f"{mode.name}_test",
            llm,
            test_records,
            root / "eval",
            args.max_new_tokens,
            args.prompt_style,
            section_mode=args.section_mode,
            context=context,
            cohort_id=test_cohort,
        )
    del llm
    fig9.clear_memory()
    return (
        {
            "pipeline_mode": mode.name,
            "image_mode": image_mode,
            "architecture": mode.description,
            "requires_stage1": mode.requires_stage1,
            "section_mode": args.section_mode,
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


def load_split_frame(split: str, cache_dir: Path):
    """Read one preprocessed split CSV, downloading it first when it is on GCS."""
    import pandas as pd

    path = fig9._split_csv_for(split)
    if str(path).startswith("gs://"):
        return fig9.read_gcs_csv(str(path), cache_dir)
    return pd.read_csv(path)


def build_native_records(
    args: argparse.Namespace, root: Path
) -> tuple[list[dict], list[dict], list[dict]]:
    """Build medgemma_direct records straight from the split manifests.

    No Stage-1 model, config, checkpoint, Q-Former or MHCAC is touched here.
    """
    cache_dir = root / ".sensitive_stage1_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames = {split: load_split_frame(split, cache_dir) for split in ("train", "val", "test")}
    # Re-check the invariant on the CSVs this run actually consumes, not just at
    # the time the splits were generated.
    assert_no_leakage(frames)
    print("[manifest] no subject/study/image overlap across splits", flush=True)
    limits = {"train": args.train_limit, "val": args.val_limit, "test": args.test_limit}
    records = []
    for split in ("train", "val", "test"):
        cohort_id = fig9.stable_fingerprint(
            {
                "split": split,
                "section_mode": args.section_mode,
                "limit": limits[split],
                "source": fig9.data_object_identity(fig9._split_csv_for(split)),
            }
        )
        records.append(
            build_records(
                frames[split],
                split=split,
                section_mode=args.section_mode,
                vis_root=fig9.VIS_ROOT,
                cohort_id=cohort_id,
                limit=limits[split],
                seed=fig9.SEED,
                require_image=args.require_image,
            )
        )
    return tuple(records)  # type: ignore[return-value]


def main() -> None:
    args = parse_args()
    modes = resolve_pipeline_modes(args.pipeline_mode)
    needs_stage1 = modes_require_stage1(modes)
    prompt_config = None
    if args.prompt_config is not None:
        from stage2.prompts import load_prompt_config

        prompt_config = load_prompt_config(args.prompt_config)
        print(
            f"[prompt] {args.prompt_config} -> version={prompt_config.version} "
            f"visual_mode={prompt_config.visual_mode.value}",
            flush=True,
        )
    if args.cue_rule == fig9.CUE_RULE_PAPER and args.threshold_path is None:
        raise SystemExit(
            "--cue-rule paper_thresholds needs --threshold-path (per-class file "
            "from scripts/calibrate_thresholds.py)"
        )
    if args.cue_rule == fig9.CUE_RULE_CUTPOINTS and args.threshold_path is None:
        raise SystemExit(
            "--cue-rule cutpoints needs --threshold-path (cutpoint file from "
            "scripts/calibrate_thresholds.py --rule cutpoints)"
        )
    try:
        validate_prompt_style(
            args.prompt_style, modes,
            has_prompt_config=prompt_config is not None, cue_rule=args.cue_rule,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    paper_prompt = args.prompt_style == fig9.PROMPT_STYLE_PAPER
    if args.cue_rule != fig9.DEFAULT_CUE_RULE:
        if not needs_stage1:
            raise SystemExit("--cue-rule requires a Stage-1 pipeline mode")
        # The paper prompt renders the P/N/U groups straight from the record,
        # listing only findings that cleared a threshold -- which is what the
        # paper describes for Eq. 22 -- so it needs no v2 cue-state contract.
        if (not paper_prompt and args.cue_rule not in fig9.NON_ABSTAINING_CUE_RULES
                and (prompt_config is None or any(
            mode.requires_stage1 and (
                prompt_config.visual_mode.image_mode != mode.image_mode
                or not prompt_config.visual_mode.includes_structured
            ) for mode in modes
        ))):
            raise SystemExit("abstaining --cue-rule requires a matching guided --prompt-config")
    if args.finding_tokens != fig9.FINDING_TOKENS_OFF:
        if not needs_stage1:
            raise SystemExit(
                "--finding-tokens needs a Stage-1 pipeline mode: the features are "
                "MHCAC's three-class probabilities"
            )
        if prompt_config is None:
            raise SystemExit(
                "--finding-tokens requires a guided --prompt-config; the "
                "placeholders come from the v2 prompt builder"
            )
    context = Stage1Context(
        run_name=args.stage1_run,
        config_path=args.stage1_config,
        checkpoint_path=args.stage1_checkpoint,
        thresholds=(fig9.load_thresholds(args.threshold_path)
                    if args.cue_rule != fig9.CUE_RULE_CUTPOINTS else {}),
        cutpoints=(fig9.load_cue_cutpoints(args.threshold_path)
                   if args.cue_rule == fig9.CUE_RULE_CUTPOINTS else {}),
    )
    fig9.set_seed(fig9.SEED)
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "eval").mkdir(parents=True, exist_ok=True)
    checkpoint_root = Path(args.checkpoint_root)

    print(f"[pipeline] {args.pipeline_mode} -> {[mode.name for mode in modes]}", flush=True)
    for mode in modes:
        print(f"[pipeline]   {mode.name}: {mode.description}", flush=True)

    # The Stage-1 Q-Former records come from ReportDataset, whose text_output is
    # the FINDINGS target only. Refuse rather than silently training the hybrid
    # on a different target than the primary pipeline.
    if needs_stage1 and args.section_mode != "findings_only":
        raise SystemExit(
            f"--section-mode {args.section_mode} is not available for a Stage-1 "
            "Q-Former mode: ReportDataset emits FINDINGS only. Use "
            "--section-mode findings_only, or run --pipeline-mode medgemma_direct."
        )

    if needs_stage1:
        if not Path(args.stage1_config).is_file():
            raise FileNotFoundError(f"Stage-1 config not found: {args.stage1_config}")
        resolved_stage1_checkpoint = fig9.stage1_checkpoint_path(context, checkpoint_root)
        if not resolved_stage1_checkpoint.is_file():
            raise FileNotFoundError(
                f"Stage-1 checkpoint not found: {resolved_stage1_checkpoint}. "
                "Pass --stage1-checkpoint or mount --checkpoint-root."
            )
        print(f"[stage1] train limit={args.train_limit or 'all'}", flush=True)
        train_records = fig9.build_stage1_records(
            context, checkpoint_root, root, "train", args.train_limit, args.num_workers,
            cue_rule=args.cue_rule, finding_tokens=args.finding_tokens,
        )
        print(f"[stage1] validation limit={args.val_limit or 'all'}", flush=True)
        val_records = fig9.build_stage1_records(
            context, checkpoint_root, root, "val", args.val_limit, args.num_workers,
            cue_rule=args.cue_rule, finding_tokens=args.finding_tokens,
        )
        print(f"[stage1] held-out test limit={args.test_limit or 'all'}", flush=True)
        test_records = fig9.build_stage1_records(
            context, checkpoint_root, root, "test", args.test_limit, args.num_workers,
            cue_rule=args.cue_rule, finding_tokens=args.finding_tokens,
        )
    else:
        print(
            "[pipeline] medgemma_direct: no Stage-1 checkpoint, config, Q-Former "
            "or MHCAC is loaded; the image is the only clinical input",
            flush=True,
        )
        train_records, val_records, test_records = build_native_records(args, root)

    print(
        f"[data] train={len(train_records)} val={len(val_records)} test={len(test_records)}",
        flush=True,
    )

    results: dict[str, dict] = {}
    adapter_dirs: list[Path] = []
    for mode in modes:
        results[mode.name], adapter_dir = train_mode(
            context, args, mode, train_records, val_records, test_records, root, prompt_config
        )
        adapter_dirs.append(adapter_dir)

    summary = {
        "schema_version": fig9.SCHEMA_VERSION,
        "model": fig9.MEDGEMMA_MODEL_ID,
        "pipeline_mode": args.pipeline_mode,
        "primary_model": modes[0].name,
        "ablation": [mode.name for mode in modes[1:]] or None,
        "prompt_config": str(args.prompt_config) if args.prompt_config else None,
        "prompt_version": (
            prompt_config.version if prompt_config
            else "paper_build_instruction" if paper_prompt
            else "legacy_build_instruction"
        ),
        "prompt_style": None if prompt_config or not needs_stage1 else args.prompt_style,
        "stage1_checkpoint": context.run_name if needs_stage1 else None,
        "cue_rule": args.cue_rule if needs_stage1 else None,
        "finding_tokens": args.finding_tokens,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "section_mode": args.section_mode,
        "target_section": args.section_mode.replace("_", " ").upper(),
        "max_new_tokens": args.max_new_tokens,
        "selection_split": "val",
        "selection_metric": "validation_cross_entropy",
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
        "pipeline_modes": [mode.name for mode in modes],
        "image_modes": [mode.image_mode for mode in modes],
        "section_mode": args.section_mode,
        "stage1_required": needs_stage1,
        "cue_rule": args.cue_rule if needs_stage1 else None,
        "finding_tokens": args.finding_tokens,
        "prompt_style": None if prompt_config or not needs_stage1 else args.prompt_style,
    }
    (root / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    print("[done]", json.dumps(summary, indent=2), flush=True)

    if args.gcs_output and not args.no_upload:
        print(f"[upload-safe-artifacts] -> {args.gcs_output}", flush=True)
        upload_safe_run(root, adapter_dirs, args.gcs_output)
    print("PIPELINE_DONE", flush=True)


if __name__ == "__main__":
    main()
