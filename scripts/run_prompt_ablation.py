#!/usr/bin/env python3
"""Prompt ablation over a validation subset — WITHOUT training.

Default is a **dry run**: it renders each prompt variant for every record and writes
per-sample prompt metadata plus prompt-level aggregates. No model is loaded and NO
generation metrics are produced, so nothing here is a model result.

    python scripts/run_prompt_ablation.py \
        --records outputs/validation_records.jsonl \
        --prompt-configs configs/prompt_ablation/P5_visual_primary.yaml \
        --max-samples 1000 --output-dir outputs/prompt_ablation

Passing ``--checkpoint`` would drive real MedGemma generation; that path requires the
GPU runtime and is intentionally guarded (raises with guidance) rather than emitting
fabricated numbers. Choose prompts on the VALIDATION split only — never the test split.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts._stage2_fixtures import synthetic_records  # noqa: E402
from stage2.prompts import (  # noqa: E402
    PromptBuilder,
    contains_temporal_language,
    context_from_record,
    load_prompt_config,
)


def _load_records(path: Path | None, limit: int) -> list[dict]:
    if path is None:
        rows = synthetic_records(min(limit, 200) if limit else 200)
    else:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def _resolve_configs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matched = [Path(p) for p in glob.glob(pattern)]
        paths.extend(sorted(matched) if matched else [Path(pattern)])
    if not paths:
        raise SystemExit("no prompt configs matched --prompt-configs")
    return paths


def _dry_run_variant(config_path: Path, records: list[dict], out_dir: Path) -> dict:
    config = load_prompt_config(config_path)
    builder = PromptBuilder(config)
    per_sample_path = out_dir / f"{config.version}_per_sample_results.jsonl"
    prompt_lengths: list[int] = []
    negatives_shown: list[int] = []
    guard_present = 0
    temporal_refs_without_prior = 0

    with per_sample_path.open("w", encoding="utf-8") as handle:
        for record in records:
            context = context_from_record(record, visual_mode=config.visual_mode)
            rendered = builder.build(context)
            user_text = rendered.user_text()
            reference = str(record.get("ref", ""))
            absent_line = next(
                (p.text for p in rendered.parts if p.text and p.text.startswith("- Clinically relevant absent")),
                "",
            )
            n_negatives = len([x for x in absent_line.split(":", 1)[-1].split(",") if x.strip()]) if absent_line else 0
            has_guard = "If prior comparison is unavailable" in user_text
            ref_temporal_no_prior = contains_temporal_language(reference) and not context.prior_available

            prompt_lengths.append(len(user_text.split()))
            negatives_shown.append(n_negatives)
            guard_present += int(has_guard)
            temporal_refs_without_prior += int(ref_temporal_no_prior)

            handle.write(
                json.dumps(
                    {
                        "study_id": context.study_id,
                        "prompt_version": rendered.prompt_version,
                        "prompt_variant": config.version,
                        "prompt_hash": rendered.prompt_hash,
                        "reference": reference,
                        "generated_report": None,  # dry run: no generation
                        "positive_cues": list(context.positive_findings),
                        "uncertain_cues": list(context.uncertain_findings),
                        "negative_cues_shown": n_negatives,
                        "anchor_view": context.anchor_view,
                        "auxiliary_views": list(context.auxiliary_views),
                        "prior_available": context.prior_available,
                        "no_prior_guard_present": has_guard,
                        "possible_temporal_in_reference_without_prior": ref_temporal_no_prior,
                        "prompt_token_count_approx": len(user_text.split()),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    n = len(records)
    return {
        "prompt_variant": config.version,
        "config": str(config_path),
        "visual_mode": config.visual_mode.value,
        "num_samples": n,
        "prompt_token_count_approx": {
            "mean": round(sum(prompt_lengths) / n, 2) if n else 0,
            "max": max(prompt_lengths) if prompt_lengths else 0,
        },
        "avg_negatives_shown": round(sum(negatives_shown) / n, 2) if n else 0,
        "no_prior_guard_rate": round(guard_present / n, 4) if n else 0,
        "reference_temporal_without_prior_rate": round(temporal_refs_without_prior / n, 4) if n else 0,
        "per_sample_results": str(per_sample_path),
        "note": "DRY RUN — no generation, no NLG/clinical metrics computed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, help="validation records JSONL; omit for synthetic")
    parser.add_argument("--prompt-configs", nargs="+", required=True, help="YAML paths or globs")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/prompt_ablation")
    parser.add_argument("--checkpoint", type=Path, help="Stage-2 adapter to actually generate with (GPU runtime)")
    args = parser.parse_args()

    if args.checkpoint is not None:
        raise SystemExit(
            "generation from a checkpoint needs the MedGemma GPU runtime and is not "
            "run from this script; it would otherwise emit unverified metrics. Use "
            "training/run_medgemma_qlora.py --prompt-config <yaml> for a real run."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = _load_records(args.records, args.max_samples)
    summaries = [_dry_run_variant(path, records, args.output_dir) for path in _resolve_configs(args.prompt_configs)]

    summary = {
        "mode": "dry_run",
        "uses_synthetic_records": args.records is None,
        "num_records": len(records),
        "variants": summaries,
        "selection_rule": "choose on validation; clinical factuality > hallucination > omission > clinical > lexical",
    }
    (args.output_dir / "ablation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
