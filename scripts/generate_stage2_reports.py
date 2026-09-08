#!/usr/bin/env python3
"""Generate FINDINGS with MedGemma and write the JSONL evaluate_stage2.py reads.

Reuses the project's own Stage-2 engine (``VariantLLM``) rather than a parallel
implementation, so the prompt, the chat template and the generation kwargs are
exactly what training would have used.

Two record sources, chosen by ``--pipeline-mode`` and never mixed:

* ``medgemma_direct`` reads the split CSV with pandas and joins ``--image-root``.
  No Stage-1 checkpoint, Q-Former or MHCAC is loaded.
* every ``meta_cxr_*`` mode calls ``build_stage1_records``, which loads the
  Stage-1 checkpoint and produces the 32 soft tokens and the MHCAC P/N/U cues.
  ``--manifest`` / ``--image-root`` are unused there: those records already
  carry an absolute ``image_path`` joined onto the dataset's ``vis_root``.

⚠ WITH NO ADAPTER THIS IS A ZERO-SHOT BASELINE, NOT THIS PROJECT'S STAGE 2.
``mode`` in the summary says which, and a soft-token mode refuses to run
zero-shot at all -- see the ``img_proj`` note below.

⚠ THE COHORTS OF THE TWO SOURCES DIFFER, so two runs are not comparable just
because they used the same ``--limit``. The native path filters to frontal views
and takes a seeded random sample of the CSV; the Stage-1 path takes whatever
``MIMIC_CXR_Dataset`` yields, in dataset order, filtered by ``generation_mask``.
Comparing arm A against arm C on "3,102 studies each" would repeat a mistake
this project has already made twice. Use ``--restrict-to <earlier.jsonl>`` to
score the SAME studies; ``sample_key`` is the blake2b of the anchor DICOM id on
both paths, so it joins across them.

PRIVACY: generated and reference report text are PhysioNet-derived. The output
goes through the same guard as the other commands, filenames carry no
identifier, and rows are keyed by a blake2 ``sample_key``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections.abc import Sequence
from hashlib import blake2b
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.evaluate_explanation import _assert_private_output_location  # noqa: E402
from training.pipeline_modes import resolve_pipeline_modes  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--pipeline-mode", default="medgemma_direct",
                        help="Architecture to generate with. medgemma_direct needs "
                             "--manifest/--image-root; meta_cxr_* need "
                             "--checkpoint-root and a Stage-1 checkpoint.")
    parser.add_argument("--limit", type=int, default=300,
                        help="studies to generate; 0 means the whole split")
    parser.add_argument("--adapter", type=Path, default=None,
                        help="Stage-2 LoRA adapter. Without it this is zero-shot.")
    parser.add_argument("--prompt-config", type=Path, default=None,
                        help="Prompt v2 config. REQUIRED by native_qformer.")
    parser.add_argument("--restrict-to", type=Path, default=None,
                        help="Earlier generated_*.jsonl; keep only its sample_keys "
                             "so two runs score the same studies.")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--seed", type=int, default=16)
    # Both default to OFF. Every recorded Stage-2 number was produced without
    # them, and changing that here would silently make those results
    # irreproducible -- see VariantLLM.generate for the measured n-gram table.
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0,
                        help="Block any n-gram from repeating. 0 = off. 5 is the "
                             "measured operating point: it constrains 2.4%% of real "
                             "reports and 44.6%% of generated ones.")
    parser.add_argument("--repetition-penalty", type=float, default=0.0,
                        help="HF repetition_penalty. 0 = off. Rescales every seen "
                             "token, including clinical terms a report legitimately "
                             "repeats, so prefer --no-repeat-ngram-size.")

    native = parser.add_argument_group("medgemma_direct record source")
    native.add_argument("--manifest", type=Path, default=None)
    native.add_argument("--image-root", type=Path, default=None)
    # BooleanOptionalAction, because `action="store_true", default=True` made
    # this flag impossible to switch off: passing it or omitting it both gave
    # True, so the frontal filter could never be lifted.
    native.add_argument("--frontal-only", action=argparse.BooleanOptionalAction,
                        default=True)

    stage1 = parser.add_argument_group("meta_cxr_* record source")
    stage1.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints"))
    stage1.add_argument("--stage1-run", default="mimic_cxr_full_blip2")
    stage1.add_argument("--stage1-config", type=Path, default=None)
    stage1.add_argument("--stage1-checkpoint", type=Path, default=None)
    stage1.add_argument("--threshold-path", type=Path, default=None)
    stage1.add_argument("--num-workers", type=int, default=4)
    stage1.add_argument("--cue-rule", default="conditional_positive",
                        choices=("conditional_positive", "mention_gated"),
                        help="How MHCAC predictions become P/N/U cues. "
                             "conditional_positive (default, what every recorded "
                             "run used) sorts all 13 findings on q alone. "
                             "mention_gated opens the mention gate first and "
                             "emits nothing for a finding the radiologist would "
                             "not have written about. Changing this changes the "
                             "Stage-1 cache identity, so it rebuilds.")
    stage1.add_argument("--stage1-cache-dir", type=Path, default=None,
                        help="Where .sensitive_stage1_cache lives. Point it at the "
                             "TRAINING output dir to reuse that run's encode pass; "
                             "defaults to --output-dir, which rebuilds it.")
    return parser.parse_args(argv)


SPLIT_ALIASES = {"val": ("val", "validate"), "test": ("test",)}

#: Mirrors ``train_eval_figure9_llm_variants_200.SOFT_TOKEN_MODES``, duplicated
#: so the argument check can run before torch/transformers/nltk are imported --
#: otherwise a forgotten flag costs a full model-stack import before it is
#: reported. ``test_generate_stage2_reports.py`` pins the two together wherever
#: the heavy module can be imported.
SOFT_TOKEN_IMAGE_MODES = frozenset({"qformer", "native_qformer"})


def validate_invocation(args: argparse.Namespace, mode) -> None:
    """Reject impossible combinations before anything expensive is imported."""
    if not mode.requires_stage1:
        if args.manifest is None or args.image_root is None:
            raise SystemExit(f"{mode.name} needs --manifest and --image-root")
    elif not Path(args.checkpoint_root).is_dir() and args.stage1_checkpoint is None:
        raise SystemExit(
            f"{mode.name} loads a Stage-1 checkpoint, but --checkpoint-root "
            f"{args.checkpoint_root} does not exist. Pass --checkpoint-root or "
            "--stage1-checkpoint."
        )
    if mode.image_mode == "native_qformer" and args.prompt_config is None:
        raise SystemExit(
            f"{mode.name} requires --prompt-config: the soft-token placeholders "
            "come from the v2 prompt builder, and the legacy instruction emits "
            "none, so the substitution would find zero positions and quietly "
            "produce plain native MedGemma"
        )
    if mode.image_mode in SOFT_TOKEN_IMAGE_MODES:
        # img_proj is created freshly initialised. Generating through a random
        # 768->hidden projector produces fluent, wrong reports and no error,
        # so require the trained one rather than letting
        # load_img_proj_if_present() return quietly.
        if args.adapter is None:
            raise SystemExit(
                f"{mode.name} has no zero-shot form: img_proj is trained in "
                "Stage 2. Pass --adapter."
            )
        projector = Path(args.adapter) / "img_proj.pt"
        if not projector.is_file():
            raise SystemExit(
                f"{projector} is missing. {mode.name} substitutes projected soft "
                "tokens at the placeholder positions; without the trained "
                "projector every report would describe noise."
            )


def sample_key_for(image_path: str | Path) -> str:
    """Stable per-study key, identical on both record sources.

    The anchor DICOM id is the image filename stem, so this reproduces the
    ``blake2b(dicom_id)`` the native path has always written -- existing JSONL
    files keep joining -- while also being computable from a Stage-1 record,
    which carries no ``dicom_id`` column of its own.
    """
    return blake2b(Path(str(image_path)).stem.encode(), digest_size=12).hexdigest()


def native_records(args: argparse.Namespace, limit: int) -> list[dict]:
    """Split-CSV records. ``limit <= 0`` keeps the whole filtered split.

    The subsetting stays `DataFrame.sample(random_state=seed)` rather than
    moving to the generic `subsample()` below, because that is the selection
    every recorded native run was made with -- arm A's 3,102-study test cohort
    included. Reproducing those numbers requires reproducing that draw.
    """
    import pandas as pd

    if args.manifest is None or args.image_root is None:
        raise SystemExit("medgemma_direct needs --manifest and --image-root")
    frame = pd.read_csv(args.manifest)
    if "split" in frame.columns:
        frame = frame[frame["split"].isin(SPLIT_ALIASES[args.split])]
    frame = frame[frame["target_valid"]]
    if args.frontal_only:
        frame = frame[frame["ViewPosition"].isin(["PA", "AP"])]
    if frame.empty:
        raise SystemExit(f"no study for split {args.split!r}")
    if 0 < limit < len(frame):
        frame = frame.sample(n=limit, random_state=args.seed)
    frame = frame.reset_index(drop=True)
    return [
        {
            "image_path": str(args.image_root / row.image_path),
            "ref": str(row.findings_clean).strip(),
            "pred_groups": {},
            "view_position": str(row.ViewPosition),
        }
        for row in frame.itertuples()
    ]


def stage1_records(args: argparse.Namespace) -> list[dict]:
    """Records carrying the 32 soft tokens and the MHCAC cues.

    Imported here, not at module scope: this is the only branch that may touch
    LAVIS, and `tests/test_native_independence.py` enforces that the native
    route never does.
    """
    from training import train_eval_figure9_llm_variants_200 as fig9
    from training.run_context import Stage1Context

    context = Stage1Context(
        run_name=args.stage1_run,
        config_path=args.stage1_config
        or (fig9.PROJECT_DIR / "pretraining/configs/mimic_cxr_full.yaml"),
        checkpoint_path=args.stage1_checkpoint,
        thresholds=fig9.load_thresholds(args.threshold_path),
    )
    fig9.set_seed(args.seed)
    cache_dir = args.stage1_cache_dir or args.output_dir
    # sample_limit stays None so the cache key matches the training run's
    # "all" pass; --limit is applied afterwards, on the records themselves.
    records = fig9.build_stage1_records(
        context, args.checkpoint_root, cache_dir, args.split, None, args.num_workers,
        use_mention_gate=(args.cue_rule == "mention_gated"),
    )
    for record in records:
        record["view_position"] = None
    return records


def subsample(records: list[dict], limit: int, seed: int) -> list[dict]:
    """Seeded subset, order preserved so the JSONL stays comparable."""
    if limit <= 0 or limit >= len(records):
        return records
    chosen = sorted(random.Random(seed).sample(range(len(records)), limit))
    return [records[i] for i in chosen]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = _assert_private_output_location(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    modes = resolve_pipeline_modes(args.pipeline_mode)
    if len(modes) != 1:
        raise SystemExit(
            f"--pipeline-mode {args.pipeline_mode!r} expands to {len(modes)} modes; "
            "this command writes one JSONL to one path, so run each mode separately"
        )
    mode = modes[0]
    validate_invocation(args, mode)

    import torch

    from training.train_eval_figure9_llm_variants_200 import SOFT_TOKEN_MODES, VariantLLM

    if SOFT_TOKEN_MODES != SOFT_TOKEN_IMAGE_MODES:
        raise RuntimeError(
            f"SOFT_TOKEN_MODES={sorted(SOFT_TOKEN_MODES)} has drifted from this "
            f"module's {sorted(SOFT_TOKEN_IMAGE_MODES)}; the early argument check "
            "would pass or refuse the wrong modes"
        )

    prompt_config = None
    if args.prompt_config is not None:
        from stage2.prompts import load_prompt_config

        prompt_config = load_prompt_config(args.prompt_config)
        print(f"[gen] prompt {args.prompt_config} version={prompt_config.version} "
              f"visual_mode={prompt_config.visual_mode.value}", flush=True)

    restricting = args.restrict_to is not None
    if mode.requires_stage1:
        # Built whole either way: sample_limit is part of the Stage-1 cache key,
        # so narrowing it there would miss the training run's cached encode pass
        # and spend ~73 minutes re-encoding the split.
        records = stage1_records(args)
    else:
        records = native_records(args, 0 if restricting else args.limit)
    for record in records:
        record["sample_key"] = sample_key_for(record["image_path"])

    if restricting:
        wanted = {
            json.loads(line)["sample_key"]
            for line in args.restrict_to.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        before = len(records)
        records = [r for r in records if r["sample_key"] in wanted]
        print(f"[gen] --restrict-to kept {len(records)} of {before} "
              f"({len(wanted)} keys in {args.restrict_to.name})", flush=True)
        if not records:
            raise SystemExit(
                "--restrict-to matched no study. The two runs describe different "
                "cohorts, so there is nothing to compare."
            )
    elif mode.requires_stage1:
        records = subsample(records, args.limit, args.seed)

    n = len(records)
    if n == 0:
        raise SystemExit(f"no study for split {args.split!r}")
    suffix = "finetuned" if args.adapter else "zeroshot"
    run_mode = f"{mode.name}_{suffix}"
    print(f"[gen] mode={run_mode} n={n} split={args.split}", flush=True)

    llm = VariantLLM(
        family="medgemma",
        image_mode=mode.image_mode,
        adapter=args.adapter,
        train_adapter=False,
        quantize_4bit=False,
        prompt_config=prompt_config,
    )
    # validate_invocation() has already established that img_proj.pt is there.
    if mode.image_mode in SOFT_TOKEN_IMAGE_MODES:
        llm.load_img_proj_if_present(args.adapter)
        print(f"[gen] img_proj loaded from {args.adapter}/img_proj.pt", flush=True)

    # Inference only: nothing trains, so freeze everything before asserting.
    # Without this the assert fires a FALSE POSITIVE -- with no adapter applied,
    # no LoRA has narrowed anything, so every parameter still carries the base
    # model's requires_grad=True and the guard counts 419M "trainable" vision
    # params. Freezing first makes the guard mean what it says: zero trainable
    # vision parameters in the run that is about to happen.
    for parameter in llm.model.parameters():
        parameter.requires_grad_(False)
    llm.model.eval()
    llm.assert_vision_tower_frozen()

    path = output_dir / f"generated_{args.split}.jsonl"
    started, failures = time.time(), 0
    with path.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records):
            try:
                generated = llm.generate(
                    record, "fine", args.max_new_tokens,
                    no_repeat_ngram_size=args.no_repeat_ngram_size or None,
                    repetition_penalty=args.repetition_penalty or None,
                )
            except Exception as exc:  # one bad study must not lose the run
                failures += 1
                print(f"[gen] study {index} failed: {type(exc).__name__}", flush=True)
                continue
            handle.write(json.dumps({
                "sample_key": record["sample_key"],
                "generated": generated,
                "reference": record["ref"],
                "view_position": record.get("view_position"),
            }, ensure_ascii=False) + "\n")
            if (index + 1) % 25 == 0:
                rate = (index + 1) / (time.time() - started)
                print(f"[gen] {index + 1}/{n}  {rate:.2f} study/s", flush=True)

    summary = {
        "mode": run_mode,
        "pipeline_mode": mode.name,
        "image_mode": mode.image_mode,
        "requires_stage1": mode.requires_stage1,
        "adapter": str(args.adapter) if args.adapter else None,
        "prompt_config": str(args.prompt_config) if args.prompt_config else None,
        "prompt_version": prompt_config.version if prompt_config else None,
        "prompt_visual_mode": (
            prompt_config.visual_mode.value if prompt_config else None
        ),
        "restrict_to": str(args.restrict_to) if args.restrict_to else None,
        "cue_rule": args.cue_rule if mode.requires_stage1 else None,
        "model_id": llm.model_id,
        "split": args.split,
        "n_requested": n,
        "n_written": n - failures,
        "n_failed": failures,
        "max_new_tokens": args.max_new_tokens,
        "no_repeat_ngram_size": args.no_repeat_ngram_size or None,
        "repetition_penalty": args.repetition_penalty or None,
        "seed": args.seed,
        "wall_seconds": round(time.time() - started, 1),
        "peak_vram_bytes": (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
        ),
        "warning": (
            None if args.adapter else
            "ZERO-SHOT: no Stage-2 adapter was loaded. These are base-model "
            "outputs, not this project's fine-tuned Stage 2."
        ),
    }
    (output_dir / "generation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[gen] n={summary['n_written']} written to {path} in "
          f"{summary['wall_seconds']:.0f}s", flush=True)
    if summary["warning"]:
        print(f"[gen] ⚠ {summary['warning']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
