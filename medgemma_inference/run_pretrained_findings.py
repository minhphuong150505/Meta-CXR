#!/usr/bin/env python3
"""Run FINDINGS inference with the external MedGemma checkpoint.

    python -m medgemma_inference.run_pretrained_findings \
        --config configs/experiments/pretrained_medgemma_findings_first.yaml \
        --split validation --max-samples 100

A full split never runs by accident: ``--max-samples all`` requires
``--confirm-full-run`` as well.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from medgemma_inference.config import ConfigError, load_config  # noqa: E402
from medgemma_inference.runner import (  # noqa: E402
    COST_ESTIMATE_FILENAME,
    run_findings_inference,
)
from model.pretrained_medgemma.errors import (  # noqa: E402
    ImpressionPhaseDisabledError,
)
from training.dataio.manifest import (  # noqa: E402
    FINDINGS_ONLY,
    ManifestError,
    assert_columns,
    build_records,
)
from training.stage2_utils import stable_fingerprint  # noqa: E402

SPLIT_ALIASES = {"validation": "val", "val": "val", "test": "test", "train": "train"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--split", default="validation", choices=sorted(SPLIT_ALIASES)
    )
    parser.add_argument(
        "--max-samples",
        default="10",
        help="Number of studies to run, or 'all' (requires --confirm-full-run).",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=None, help="Override split CSV.")
    parser.add_argument("--vis-root", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument(
        "--estimate-full-cost",
        action="store_true",
        help="Project the pilot's measured rate onto the whole split.",
    )
    parser.add_argument(
        "--confirm-full-run",
        action="store_true",
        help="Required to run every study in the split.",
    )
    return parser.parse_args(argv)


def resolve_split_csv(split: str, override: Path | None) -> Path:
    if override is not None:
        return override
    from local_config import (
        PROCESSED_TEST_CSV,
        PROCESSED_TRAIN_CSV,
        PROCESSED_VAL_CSV,
    )

    return Path(
        {
            "train": PROCESSED_TRAIN_CSV,
            "val": PROCESSED_VAL_CSV,
            "test": PROCESSED_TEST_CSV,
        }[split]
    )


def resolve_vis_root(override: Path | None) -> Path:
    if override is not None:
        return override
    from local_config import VIS_ROOT

    return Path(VIS_ROOT)


def previous_projection(output_dir: Path) -> dict | None:
    path = output_dir / COST_ESTIMATE_FILENAME
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError:
        return None


def print_banner(
    *, config, split: str, requested: int, total: int, output_dir: Path
) -> None:
    projection = previous_projection(output_dir)
    print("=" * 70)
    print("Pretrained MedGemma -- FINDINGS-first inference")
    print("=" * 70)
    print(f"  model id        : {config.findings.model_id}")
    print(f"  revision        : {config.findings.revision}")
    print("  provenance      : external checkpoint, NOT fine-tuned by this project")
    print(f"  split           : {split}  ({total} anchor studies available)")
    print(f"  samples to run  : {requested}")
    print(f"  hourly cost     : ${config.runtime.hourly_cost_usd:.2f}/h")
    print(f"  budget limit    : ${config.runtime.budget_limit_usd:.2f}")
    print(f"  max runtime     : {config.runtime.max_runtime_hours:.1f}h")
    print("  Impression      : DISABLED (Phase 2, not budgeted)")
    print(f"  output dir      : {output_dir}")
    if projection:
        print(
            f"  prior estimate  : {projection.get('projected_hours', 0):.2f}h / "
            f"${projection.get('projected_cost_usd', 0):.2f} for "
            f"{projection.get('target_samples', 0)} samples"
        )
    else:
        print("  prior estimate  : none yet (this run will measure it)")
    print("=" * 70, flush=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    split = SPLIT_ALIASES[args.split]
    csv_path = resolve_split_csv(split, args.csv)
    if not csv_path.is_file():
        print(f"error: split CSV not found: {csv_path}", file=sys.stderr)
        return 2

    frame = pd.read_csv(csv_path)
    try:
        assert_columns(frame, FINDINGS_ONLY, f"{split} manifest")
    except ManifestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    dataset_fingerprint = stable_fingerprint(
        {
            "split": split,
            "rows": int(len(frame)),
            "columns": sorted(map(str, frame.columns)),
            "section": FINDINGS_ONLY,
        }
    )

    all_records = build_records(
        frame,
        split=split,
        section_mode=FINDINGS_ONLY,
        vis_root=resolve_vis_root(args.vis_root),
        cohort_id=stable_fingerprint({"split": split, "section": FINDINGS_ONLY}),
        limit=0,
        seed=args.seed,
    )
    total = len(all_records)

    if str(args.max_samples).lower() == "all":
        if not args.confirm_full_run:
            print(
                f"error: --max-samples all would run {total} studies. "
                "Re-run with --confirm-full-run once the pilot cost estimate is "
                "acceptable.",
                file=sys.stderr,
            )
            return 2
        records = all_records
    else:
        try:
            limit = int(args.max_samples)
        except ValueError:
            print(
                f"error: --max-samples must be an integer or 'all', got "
                f"{args.max_samples!r}",
                file=sys.stderr,
            )
            return 2
        records = all_records[:limit]

    output_dir = args.output_dir or (
        _REPO_ROOT / "outputs" / f"findings_{split}_{len(records)}"
    )
    output_dir = Path(output_dir)
    print_banner(
        config=config,
        split=split,
        requested=len(records),
        total=total,
        output_dir=output_dir,
    )

    try:
        summary = run_findings_inference(
            config,
            records,
            output_dir,
            split=split,
            dataset_fingerprint=dataset_fingerprint,
            target_samples=total if args.estimate_full_cost else len(records),
        )
    except ImpressionPhaseDisabledError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3

    print("-" * 70)
    print(f"  generated       : {summary.generated}")
    print(f"  already done    : {summary.skipped_already_done}")
    print(f"  elapsed         : {summary.elapsed_seconds / 3600.0:.3f}h")
    print(f"  estimated cost  : ${summary.estimated_cost_usd:.3f}")
    print(f"  rate            : {summary.samples_per_second:.3f} sample/s")
    if summary.warnings_seen:
        print(f"  warnings        : {summary.warnings_seen}")
    if summary.stopped_on_budget:
        print("  STOPPED ON BUDGET -- rerun the same command to resume.")
    print(f"  cost estimate   : {output_dir / COST_ESTIMATE_FILENAME}")
    print("-" * 70, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
