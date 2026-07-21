#!/usr/bin/env python3
"""Evaluate generated radiology reports from a JSONL prediction file.

    python scripts/evaluate_stage2.py \\
        --predictions outputs/generated_reports.jsonl \\
        --metrics bleu,rouge,meteor,cider,bertscore \\
        --clinical-metrics chexbert,radgraph \\
        --bootstrap-samples 1000 \\
        --output-dir outputs/stage2_evaluation

Input format: one JSON object per line with ``sample_key``, ``generated`` and
``reference``. Optional ``view_position``/``num_views`` enable subgroup analysis.

A metric whose package is missing is reported as **unavailable with an install
command**, never as a score of 0. Use ``--skip-clinical-metrics`` on a light
environment.

Exit codes: 0 success, 1 evaluation failed, 2 bad arguments or input.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.evaluation.bootstrap import (  # noqa: E402
    DEFAULT_CONFIDENCE,
    DEFAULT_SAMPLES,
    DEFAULT_SEED,
    bootstrap_sample_metric,
)
from training.evaluation.error_analysis import analyse_sample, summarise_errors  # noqa: E402
from training.evaluation.generation_metrics import (  # noqa: E402
    DEFAULT_METRICS,
    LEXICAL_METRICS,
    compute_generation_metrics,
)
from training.evaluation.report_writer import (  # noqa: E402
    ExperimentMetadata,
    build_markdown_report,
    write_csv,
    write_json,
    write_jsonl,
)
from training.evaluation.schemas import load_generation_records  # noqa: E402
from training.evaluation.subgroup_analysis import (  # noqa: E402
    evaluate_subgroups,
    length_subgroups,
    subgroup_table,
    view_subgroups,
)

logger = logging.getLogger("evaluate_stage2")


def comma_list(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument(
        "--metrics",
        type=comma_list,
        default=DEFAULT_METRICS,
        help=f"Comma-separated. Available: {','.join(LEXICAL_METRICS)}",
    )
    parser.add_argument(
        "--clinical-metrics",
        type=comma_list,
        default=(),
        help="Comma-separated, e.g. chexbert,radgraph. Requires optional deps.",
    )
    parser.add_argument("--skip-clinical-metrics", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail instead of reporting a metric as unavailable.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--bootstrap-confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--evaluation-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--no-bootstrap", action="store_true")
    parser.add_argument("--bertscore-model", default="distilbert-base-uncased")
    parser.add_argument("--bertscore-device", default="cpu")
    parser.add_argument(
        "--include-text",
        action="store_true",
        help="Write the report text into per_sample_results.jsonl. Off by "
        "default: generated and reference report text is restricted data.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", default="unknown")
    parser.add_argument("--checkpoint", default="unknown")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if not args.predictions.is_file():
        logger.error("prediction file not found: %s", args.predictions)
        return 2

    try:
        records = load_generation_records(args.predictions)
    except Exception as exc:  # noqa: BLE001
        logger.error("could not read %s: %s", args.predictions, exc)
        return 2

    generated = [r.generated for r in records]
    references = [r.reference for r in records]
    logger.info("evaluating %d generated reports", len(records))

    unknown = set(args.metrics) - set(LEXICAL_METRICS)
    if unknown:
        logger.error(
            "unknown metric(s): %s. Available: %s",
            ", ".join(sorted(unknown)),
            ", ".join(LEXICAL_METRICS),
        )
        return 2

    try:
        suite = compute_generation_metrics(
            generated,
            references,
            metrics=tuple(args.metrics),
            bertscore_model=args.bertscore_model,
            bertscore_device=args.bertscore_device,
            strict=args.strict,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("metric computation failed: %s", exc)
        return 1

    for metric, reason in suite.unavailable.items():
        logger.warning("metric %s unavailable: %s", metric, reason)

    # ---- clinical metrics ------------------------------------------------
    clinical: dict[str, str] = {}
    if args.clinical_metrics and not args.skip_clinical_metrics:
        from training.evaluation.clinical import (
            MissingOptionalDependency,
            build_metric,
        )

        for name in args.clinical_metrics:
            try:
                result = build_metric(name).compute(generated, references)
            except MissingOptionalDependency as exc:
                logger.warning("clinical metric %s unavailable: %s", name, exc)
                clinical[name] = f"unavailable — {exc}"
            except NotImplementedError as exc:
                # The package is installed but the adapter is deliberately not
                # wired; see training/evaluation/clinical.py. Reported as a gap
                # rather than as a score.
                logger.warning("clinical metric %s not wired: %s", name, exc)
                clinical[name] = f"not implemented — {exc}"
            except ValueError as exc:
                logger.error("clinical metric %s rejected the input: %s", name, exc)
                return 1
            else:
                clinical[name] = result
    elif args.skip_clinical_metrics:
        logger.info("clinical metrics skipped by --skip-clinical-metrics")

    # ---- per-sample error analysis --------------------------------------
    sample_reports = []
    for index, record in enumerate(records):
        scores = {
            name: values[index]
            for name, values in suite.per_sample.items()
            if index < len(values)
        }
        sample_reports.append(
            analyse_sample(
                record.sample_key,
                record.generated,
                record.reference,
                scores=scores,
            )
        )
    errors = summarise_errors(sample_reports)

    # ---- bootstrap -------------------------------------------------------
    intervals = {}
    if not args.no_bootstrap and args.bootstrap_samples > 0:
        for name, values in suite.per_sample.items():
            if name.startswith("rouge") or name in {"meteor", "cider"} or "bertscore" in name:
                intervals[name] = bootstrap_sample_metric(
                    values,
                    metric_name=name,
                    samples=args.bootstrap_samples,
                    confidence=args.bootstrap_confidence,
                    seed=args.evaluation_seed,
                )

    # ---- subgroups -------------------------------------------------------
    subgroups = view_subgroups(
        [r.view_position for r in records] if records[0].view_position else None,
        [r.num_views for r in records] if records[0].num_views else None,
    )
    subgroups.extend(length_subgroups([r.reference_length for r in sample_reports]))

    subgroup_payload = None
    subgroup_markdown = None
    if subgroups and suite.per_sample:

        def subgroup_metrics(indices: np.ndarray) -> dict[str, float]:
            return {
                name: float(np.mean([values[i] for i in indices]))
                for name, values in suite.per_sample.items()
                if name in {"rouge_l", "meteor", "cider", "bertscore_f1"}
            }

        results = evaluate_subgroups(subgroups, subgroup_metrics)
        subgroup_payload = [r.to_dict() for r in results]
        columns = [
            c
            for c in ("rouge_l", "meteor", "cider", "bertscore_f1")
            if c in suite.per_sample
        ]
        if columns:
            subgroup_markdown = subgroup_table(results, columns)

    # ---- artifacts -------------------------------------------------------
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "corpus": suite.corpus,
        "provenance": suite.provenance,
        "unavailable": suite.unavailable,
        "clinical": clinical,
        "errors": errors,
        "intervals": {k: v.to_dict() for k, v in intervals.items()},
    }
    if subgroup_payload:
        payload["subgroups"] = subgroup_payload
    if subgroup_markdown:
        payload["subgroup_table"] = subgroup_markdown

    metadata = ExperimentMetadata(
        split=args.split,
        num_samples=len(records),
        checkpoint=args.checkpoint,
        seed=args.evaluation_seed,
        threshold_source="n/a (generation)",
        uncertain_policy="n/a (generation)",
    )

    write_json({"metadata": metadata.to_dict(), "generation": payload}, output_dir / "metrics.json")
    write_csv(
        [{"metric": k, "value": v} for k, v in suite.corpus.items()],
        output_dir / "summary.csv",
    )
    write_jsonl(
        [r.to_dict(include_text=args.include_text) for r in sample_reports],
        output_dir / "per_sample_results.jsonl",
    )

    markdown = build_markdown_report(metadata, generation=payload)
    (output_dir / "evaluation_report.md").write_text(markdown, encoding="utf-8")

    logger.info("wrote evaluation artifacts to %s", output_dir)
    for name in ("bleu_4", "rouge_l", "meteor", "cider", "bertscore_f1"):
        if name in suite.corpus:
            print(f"{name:<16}: {suite.corpus[name]:.4f}")
    print(f"empty reports   : {errors['empty_output_rate']:.4f}")
    print(f"possible temporal hallucination: "
          f"{errors['possible_temporal_hallucination_rate']:.4f}")
    print(f"report          : {output_dir / 'evaluation_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
