"""End-to-end evaluator integration test on synthetic data.

Exercises the real CLI entrypoints, not just the library functions:

    fake logits -> save predictions -> calibrate on validation
      -> evaluate test with the saved thresholds -> JSON/CSV/Markdown

The whole point is proving the evaluator runs **from a prediction file with no
model, no GPU and no dataset**. Nothing here touches MIMIC-CXR.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import calibrate_thresholds as calibrate_cli  # noqa: E402
from scripts import evaluate_stage1 as stage1_cli  # noqa: E402
from scripts import evaluate_stage2 as stage2_cli  # noqa: E402
from training.evaluation.schemas import (  # noqa: E402
    ClassificationPredictions,
    GenerationRecord,
    save_generation_records,
)

PATHOLOGIES = (
    "No Finding",
    "Cardiomegaly",
    "Edema",
    "Consolidation",
    "Pleural Effusion",
    "Support Devices",
)


def synthetic_predictions(
    num_samples: int, seed: int, *, split: str
) -> ClassificationPredictions:
    """A model that is informative but imperfect, with realistic imbalance.

    Positive-class scores are drawn higher for true positives than for true
    negatives with substantial overlap, so AUROC lands well above chance without
    the problem being trivially separable.
    """
    rng = np.random.default_rng(seed)
    num_pathologies = len(PATHOLOGIES)

    labels = np.zeros((num_samples, num_pathologies), dtype=int)
    positive = np.zeros((num_samples, num_pathologies))

    for p in range(num_pathologies):
        prevalence = 0.05 + 0.08 * p  # 5% .. 45%
        is_positive = rng.random(num_samples) < prevalence
        labels[:, p] = is_positive.astype(int)

        # A handful of uncertain labels, and a few missing ones.
        uncertain = rng.random(num_samples) < 0.05
        labels[uncertain, p] = 2
        missing = rng.random(num_samples) < 0.02
        labels[missing, p] = -1

        scores = np.where(
            is_positive,
            rng.beta(5, 3, num_samples),
            rng.beta(2, 6, num_samples),
        )
        positive[:, p] = np.clip(scores, 0.001, 0.999)

    remainder = 1.0 - positive
    probabilities = np.stack([remainder * 0.7, positive, remainder * 0.3], axis=-1)

    views = rng.choice(["PA", "AP", "LATERAL"], size=num_samples)
    num_views = rng.integers(1, 3, size=num_samples)

    return ClassificationPredictions(
        labels=labels,
        probabilities=probabilities,
        pathology_names=PATHOLOGIES,
        sample_keys=np.asarray([f"{split}_{i:05d}" for i in range(num_samples)]),
        view_positions=views,
        num_views=num_views,
        metadata={"split": split, "checkpoint": "synthetic.pth"},
    )


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return tmp_path


def test_full_stage1_pipeline_runs_without_a_model(workspace: Path):
    validation = synthetic_predictions(400, seed=1, split="validation")
    test = synthetic_predictions(300, seed=2, split="test")
    validation_path = validation.save(workspace / "validation_predictions.npz")
    test_path = test.save(workspace / "test_predictions.npz")

    # ---- 1. per-class thresholds on validation (paper Eq. 22) --------------
    thresholds_path = workspace / "thresholds.json"
    assert calibrate_cli.main(
        ["--predictions", str(validation_path), "--output", str(thresholds_path)]
    ) == 0
    payload = json.loads(thresholds_path.read_text())
    assert payload["metadata"]["split"] == "validation"
    assert set(payload["thresholds"]) == set(PATHOLOGIES)
    assert set(payload["thresholds"]["Edema"]) == {"negative", "positive", "uncertain"}

    # ---- 2. evaluate test with the paper's three-class protocol -----------
    output_dir = workspace / "stage1_evaluation"
    assert stage1_cli.main(
        [
            "--predictions", str(test_path),
            "--bootstrap-samples", "20",
            "--evaluation-seed", "7",
            "--no-plots",
            "--output-dir", str(output_dir),
        ]
    ) == 0
    for name in ("metrics.json", "summary.csv", "per_pathology_metrics.csv", "evaluation_report.md"):
        assert (output_dir / name).is_file(), f"{name} was not written"

    classification = json.loads((output_dir / "metrics.json").read_text())["classification"]
    aggregates = classification["aggregates"]
    for key in ("weighted_precision", "weighted_recall", "weighted_f1",
                "mean_weighted_f1_5", "auroc_positive_mean", "auroc_negative_mean"):
        assert key in aggregates
    assert not [k for k in aggregates if "positive_macro" in k or "presence" in k]
    # All six findings, No Finding and Support Devices included (paper Fig. 5).
    assert len(classification["per_pathology"]) == len(PATHOLOGIES)
    assert aggregates["auroc_positive_mean"] > 0.7

    interval = classification["intervals"]["weighted_f1"]
    assert interval["lower"] <= interval["point_estimate"] <= interval["upper"]

    baselines = {row["baseline"]: row for row in classification["baselines"]}
    assert {"all_negative", "majority_class", "prior_random"} <= set(baselines)

    subgroups = {row["subgroup"] for row in classification["subgroups"]}
    assert "view_PA" in subgroups and "multi_view" in subgroups

    report = (output_dir / "evaluation_report.md").read_text()
    for section in ("Experiment metadata", "Baseline comparison", "Limitations"):
        assert section in report


def test_metrics_json_is_parseable_and_has_no_nan_token(workspace: Path):
    """A class that never occurs yields null AUROC, not a bare NaN token."""
    labels = np.zeros((20, 2), dtype=int)
    labels[:5, 0] = 1  # P0 has positives, P1 has only negatives
    probabilities = np.zeros((20, 2, 3))
    probabilities[..., 1] = 0.3
    probabilities[..., 0] = 0.7
    predictions = ClassificationPredictions(
        labels=labels,
        probabilities=probabilities,
        pathology_names=("Cardiomegaly", "Fracture"),
        sample_keys=np.asarray([f"s{i}" for i in range(20)]),
        metadata={"split": "test"},
    )
    path = predictions.save(workspace / "test.npz")
    output_dir = workspace / "out"
    assert stage1_cli.main(
        ["--predictions", str(path), "--no-bootstrap", "--no-plots", "--output-dir", str(output_dir)]
    ) == 0
    raw = (output_dir / "metrics.json").read_text()
    assert "NaN" not in raw and "Infinity" not in raw
    per_pathology = {
        row["pathology"]: row for row in json.loads(raw)["classification"]["per_pathology"]
    }
    assert per_pathology["Fracture"]["auroc_positive"] is None
    assert per_pathology["Fracture"]["auroc_uncertain"] is None


def test_stage1_cli_scores_test_with_validation_thresholds(workspace: Path):
    validation = synthetic_predictions(400, seed=1, split="validation")
    test = synthetic_predictions(300, seed=2, split="test")
    validation_path = validation.save(workspace / "validation_predictions.npz")
    test_path = test.save(workspace / "test_predictions.npz")
    thresholds_path = workspace / "thresholds.json"
    assert calibrate_cli.main(
        ["--predictions", str(validation_path), "--output", str(thresholds_path)]
    ) == 0
    out = workspace / "with_thresholds"
    assert stage1_cli.main([
        "--predictions", str(test_path), "--thresholds", str(thresholds_path),
        "--no-bootstrap", "--no-plots", "--output-dir", str(out),
    ]) == 0
    payload = json.loads((out / "metrics.json").read_text())
    assert "NOT the paper" in payload["classification"]["settings"]["decision"]
    assert "Eq. 22" in payload["metadata"]["threshold_source"]


def test_stage1_cli_refuses_thresholds_fitted_on_the_scored_split(workspace: Path):
    validation = synthetic_predictions(400, seed=1, split="validation")
    validation_path = validation.save(workspace / "validation_predictions.npz")
    thresholds_path = workspace / "thresholds.json"
    assert calibrate_cli.main(
        ["--predictions", str(validation_path), "--output", str(thresholds_path)]
    ) == 0
    assert stage1_cli.main([
        "--predictions", str(validation_path), "--thresholds", str(thresholds_path),
        "--no-bootstrap", "--no-plots", "--output-dir", str(workspace / "x"),
    ]) == 2


def test_calibration_cli_refuses_a_test_split(workspace: Path):
    predictions = synthetic_predictions(100, seed=4, split="test")
    path = predictions.save(workspace / "test.npz")
    assert calibrate_cli.main(
        ["--predictions", str(path), "--output", str(workspace / "thresholds.json")]
    ) == 2


def test_missing_prediction_file_exits_nonzero(workspace: Path):
    assert stage1_cli.main(
        [
            "--predictions", str(workspace / "nope.npz"),
            "--output-dir", str(workspace / "out"),
        ]
    ) == 2


def test_full_stage2_pipeline_runs_without_a_model(workspace: Path):
    records = [
        GenerationRecord(
            sample_key="s1",
            generated="The lungs are clear. No focal consolidation.",
            reference="The lungs are clear. No focal consolidation.",
            view_position="PA",
            num_views=2,
        ),
        GenerationRecord(
            sample_key="s2",
            generated="There is a pneumothorax.",
            reference="There is no pneumothorax.",
            view_position="AP",
            num_views=1,
        ),
        GenerationRecord(
            sample_key="s3",
            generated="",
            reference="Small right pleural effusion.",
            view_position="PA",
            num_views=1,
        ),
        GenerationRecord(
            sample_key="s4",
            generated="Unchanged since prior study. Unchanged since prior study.",
            reference="Stable cardiomegaly.",
            view_position="LATERAL",
            num_views=2,
        ),
    ]
    predictions_path = save_generation_records(records, workspace / "reports.jsonl")
    output_dir = workspace / "stage2_evaluation"

    exit_code = stage2_cli.main(
        [
            "--predictions", str(predictions_path),
            "--metrics", "bleu,rouge",
            "--skip-clinical-metrics",
            "--bootstrap-samples", "20",
            "--output-dir", str(output_dir),
        ]
    )
    assert exit_code == 0

    for name in ("metrics.json", "summary.csv", "per_sample_results.jsonl", "evaluation_report.md"):
        assert (output_dir / name).is_file(), f"{name} was not written"

    metrics = json.loads((output_dir / "metrics.json").read_text())
    generation = metrics["generation"]

    assert "bleu_4" in generation["corpus"]
    assert "rouge_l" in generation["corpus"]

    errors = generation["errors"]
    assert errors["num_samples"] == 4
    assert errors["empty_output_rate"] == pytest.approx(0.25)  # s3
    assert errors["possible_temporal_hallucination_rate"] == pytest.approx(0.25)  # s4
    assert errors["flag_counts"]["possible_negation_error"] >= 1  # s2

    lines = (output_dir / "per_sample_results.jsonl").read_text().strip().split("\n")
    assert len(lines) == 4
    per_sample = [json.loads(line) for line in lines]
    assert {row["sample_key"] for row in per_sample} == {"s1", "s2", "s3", "s4"}
    # Report text is restricted data and must be opt-in.
    assert "generated" not in per_sample[0]

    report = (output_dir / "evaluation_report.md").read_text()
    assert "Stage 2" in report
    assert "heuristic" in report


def test_stage2_include_text_flag_opts_into_report_text(workspace: Path):
    records = [
        GenerationRecord(sample_key="s1", generated="clear lungs", reference="clear lungs")
    ]
    path = save_generation_records(records, workspace / "reports.jsonl")
    output_dir = workspace / "out"

    assert stage2_cli.main(
        [
            "--predictions", str(path),
            "--metrics", "bleu",
            "--skip-clinical-metrics",
            "--no-bootstrap",
            "--include-text",
            "--output-dir", str(output_dir),
        ]
    ) == 0

    row = json.loads((output_dir / "per_sample_results.jsonl").read_text().strip())
    assert row["generated"] == "clear lungs"


def test_reevaluating_the_same_file_is_deterministic(workspace: Path):
    predictions = synthetic_predictions(200, seed=5, split="test")
    path = predictions.save(workspace / "test.npz")

    outputs = []
    for index in range(2):
        output_dir = workspace / f"run{index}"
        assert stage1_cli.main(
            [
                "--predictions", str(path),
                "--bootstrap-samples", "20",
                "--evaluation-seed", "11",
                "--no-plots",
                "--output-dir", str(output_dir),
            ]
        ) == 0
        metrics = json.loads((output_dir / "metrics.json").read_text())
        outputs.append(metrics["classification"])

    assert outputs[0]["aggregates"] == outputs[1]["aggregates"]
    assert outputs[0]["intervals"] == outputs[1]["intervals"]
