#!/usr/bin/env python3
"""Evaluate Stage-1 classification the way the META-CXR paper does.

Three classes per finding (Negative / Positive / Uncertain), argmax decision,
per-finding sklearn-weighted precision / recall / F1 averaged over the 14
findings (paper Fig. 10), mean F1 over the five common findings (Tables 5 and
7), and one-vs-rest AUC-ROC per class (Fig. 5). See
``training/evaluation/classification_metrics.py`` for the exact definitions.
There is no binary framing, no positive-only F1 and no uncertain policy: those
were removed on 2026-09-29 (D-023).

No model, GPU or dataset is needed -- only the ``.npz`` the run wrote.

    python scripts/evaluate_stage1.py \\
        --predictions <run>/result/test_predictions_epoch_best.npz \\
        --output-dir <private dir>/stage1_test

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

from training.evaluation.baselines import (  # noqa: E402
    BASELINE_COLUMNS,
    BaselineRow,
    baseline_table,
    compute_baselines,
)
from training.evaluation.bootstrap import (  # noqa: E402
    DEFAULT_CONFIDENCE,
    DEFAULT_SAMPLES,
    DEFAULT_SEED,
    bootstrap_many,
)
from training.evaluation.classification_metrics import evaluate_classification  # noqa: E402
from training.evaluation.report_writer import (  # noqa: E402
    HEADLINE_CLASSIFICATION_METRICS,
    ExperimentMetadata,
    build_markdown_report,
    write_csv,
    write_json,
)
from training.evaluation.schemas import (  # noqa: E402
    CLASS_NAMES,
    MISSING,
    ClassificationPredictions,
)
from training.evaluation.subgroup_analysis import (  # noqa: E402
    evaluate_subgroups,
    label_subgroups,
    subgroup_table,
    view_subgroups,
)

logger = logging.getLogger("evaluate_stage1")

SUBGROUP_COLUMNS = ("weighted_f1", "mean_weighted_f1_5", "auroc_positive_mean")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--bootstrap-confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--evaluation-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--no-bootstrap", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-baselines", action="store_true")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", default=None)
    parser.add_argument("--checkpoint", default="unknown")
    parser.add_argument("--config", default="unknown")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def _subset(predictions: ClassificationPredictions, indices: np.ndarray):
    return ClassificationPredictions(
        labels=predictions.labels[indices],
        probabilities=predictions.probabilities[indices],
        pathology_names=predictions.pathology_names,
        sample_keys=predictions.sample_keys[indices],
    )


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
        predictions = ClassificationPredictions.load(args.predictions)
    except Exception as exc:  # noqa: BLE001 - surfaced to the user
        logger.error("could not read %s: %s", args.predictions, exc)
        return 2
    if predictions.num_samples == 0:
        logger.error("%s contains no samples", args.predictions)
        return 2
    logger.info(
        "loaded %d studies x %d findings from %s",
        predictions.num_samples, predictions.num_pathologies, args.predictions,
    )

    report = evaluate_classification(predictions)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "aggregates": report.aggregates,
        "per_pathology": [m.to_dict() for m in report.per_pathology],
        "confusion": {m.name: m.confusion for m in report.per_pathology},
        "settings": report.settings,
    }

    if not args.no_bootstrap and args.bootstrap_samples > 0:
        logger.info("bootstrapping %d replicates by study", args.bootstrap_samples)

        def getter(name: str):
            return lambda idx: evaluate_classification(_subset(predictions, idx)).aggregates[name]

        intervals = bootstrap_many(
            {name: getter(name) for name in HEADLINE_CLASSIFICATION_METRICS},
            predictions.num_samples,
            samples=args.bootstrap_samples,
            confidence=args.bootstrap_confidence,
            seed=args.evaluation_seed,
        )
        payload["intervals"] = {k: v.to_dict() for k, v in intervals.items()}

    if not args.no_baselines:
        rows = compute_baselines(predictions, seed=args.evaluation_seed)
        model_row = BaselineRow(
            name="model",
            metrics={k: report.aggregates[k] for k in BASELINE_COLUMNS},
            description="the evaluated model",
        )
        payload["baselines"] = [row.to_dict() for row in rows]
        payload["baseline_table"] = baseline_table(rows, model_row)

    subgroups = view_subgroups(predictions.view_positions, predictions.num_views)
    subgroups.extend(label_subgroups(predictions.labels, predictions.pathology_names))
    if subgroups:
        def subgroup_metrics(indices: np.ndarray) -> dict[str, float]:
            aggregates = evaluate_classification(_subset(predictions, indices)).aggregates
            return {k: aggregates[k] for k in SUBGROUP_COLUMNS}

        results = evaluate_subgroups(subgroups, subgroup_metrics)
        payload["subgroups"] = [r.to_dict() for r in results]
        payload["subgroup_table"] = subgroup_table(results, list(SUBGROUP_COLUMNS))

    if not args.no_plots:
        try:
            _write_plots(predictions, report, output_dir)
        except Exception as exc:  # noqa: BLE001 - never lose computed metrics
            logger.warning("plotting failed (metrics are unaffected): %s", exc)

    metadata = ExperimentMetadata(
        split=args.split or str(predictions.metadata.get("split", "unknown")),
        num_samples=predictions.num_samples,
        num_pathologies=predictions.num_pathologies,
        checkpoint=args.checkpoint or str(predictions.metadata.get("checkpoint", "unknown")),
        config=args.config,
        seed=args.evaluation_seed,
        threshold_source="argmax (paper)",
    )
    write_json({"metadata": metadata.to_dict(), "classification": payload},
               output_dir / "metrics.json")
    write_csv([m.to_dict() for m in report.per_pathology],
              output_dir / "per_pathology_metrics.csv")
    write_csv([{"metric": k, "value": v} for k, v in report.aggregates.items()],
              output_dir / "summary.csv")
    (output_dir / "evaluation_report.md").write_text(
        build_markdown_report(metadata, classification=payload), encoding="utf-8"
    )

    agg = report.aggregates
    print("\nPaper protocol (three classes, argmax), mean over the 14 findings:")
    print(f"  weighted precision / recall / F1 : {agg['weighted_precision']:.4f} / "
          f"{agg['weighted_recall']:.4f} / {agg['weighted_f1']:.4f}   (paper 0.87 / 0.78 / 0.73)")
    print(f"  mean weighted F1, 5 findings     : {agg['mean_weighted_f1_5']:.4f}   (paper Table 5: 0.701)")
    print(f"  AUROC one-vs-rest  Pos/Neg/Unc   : {agg['auroc_positive_mean']:.4f} / "
          f"{agg['auroc_negative_mean']:.4f} / {agg['auroc_uncertain_mean']:.4f}")
    print(f"  macro recall, 3 classes          : {agg['macro_recall']:.4f}   (not in the paper; selection metric)")
    print(f"  report                           : {output_dir / 'evaluation_report.md'}")
    return 0


def _write_plots(predictions, report, output_dir: Path) -> None:
    """Fig. 5 of the paper: one ROC panel per class, every finding overlaid."""
    from training.evaluation import visualization

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    valid = predictions.labels != MISSING
    for c, cls in enumerate(CLASS_NAMES):
        class_dir = plots_dir / f"{cls}_vs_rest"
        class_dir.mkdir(exist_ok=True)
        visualization.plot_roc_curves(
            predictions.probabilities[..., c],
            predictions.labels == c,
            valid,
            predictions.pathology_names,
            class_dir,
        )
    visualization.plot_confusion_matrices(
        np.asarray([m.confusion for m in report.per_pathology]),
        predictions.pathology_names,
        CLASS_NAMES,
        output_dir,
    )
    visualization.plot_per_pathology_bars(
        {m.name: m.weighted_f1 for m in report.per_pathology},
        "Weighted three-class F1 by finding",
        "F1",
        plots_dir,
        "weighted_f1_by_finding.png",
    )


if __name__ == "__main__":
    raise SystemExit(main())
