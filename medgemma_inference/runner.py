"""Findings-first inference runner.

Order of operations matters here. The Impression guard runs *before* anything
is loaded, so a misconfigured run fails in milliseconds instead of after
downloading a second 4B checkpoint. The model is constructed lazily, only once
there is unfinished work, so a fully-resumed run costs no GPU time at all.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from medgemma_inference.config import ExperimentConfig
from medgemma_inference.prediction_writer import PredictionWriter
from medgemma_inference.progress import ProgressFile, RunIdentity
from model.pretrained_medgemma.impression_reporter import assert_impression_disabled
from model.pretrained_medgemma.output_schema import FindingsPrediction
from runtime.budget import BudgetExceeded, BudgetState

COST_ESTIMATE_FILENAME = "cost_estimate_findings.json"


@dataclass
class RunSummary:
    """What happened, for the CLI to print and for the cost estimate."""

    requested: int
    skipped_already_done: int = 0
    generated: int = 0
    stopped_on_budget: bool = False
    elapsed_seconds: float = 0.0
    estimated_cost_usd: float = 0.0
    samples_per_second: float = 0.0
    warnings_seen: dict[str, int] = field(default_factory=dict)

    @property
    def completed(self) -> int:
        return self.skipped_already_done + self.generated


def load_image(path: str):
    """Open one chest X-ray as RGB.

    MedGemma's SigLIP tower expects 3 channels; MIMIC-CXR JPEGs are greyscale.
    """
    from PIL import Image

    with Image.open(path) as handle:
        return handle.convert("RGB")


def _build_reporter(config: ExperimentConfig):
    """Construct the real Findings reporter. Imported lazily: no GPU stack
    is touched unless a run actually reaches this point."""
    from model.pretrained_medgemma.findings_loader import PretrainedFindingsLoader
    from model.pretrained_medgemma.findings_reporter import (
        GenerationSettings,
        PretrainedFindingsReporter,
    )

    bundle = PretrainedFindingsLoader(
        model_id=config.findings.model_id,
        revision=config.findings.revision,
        device=config.findings.device,
        dtype=config.findings.dtype,
        load_in_4bit=config.findings.load_in_4bit,
    ).load()
    settings = GenerationSettings(
        max_new_tokens=config.findings.max_new_tokens,
        do_sample=config.findings.do_sample,
        num_beams=config.findings.num_beams,
    )
    return PretrainedFindingsReporter(bundle, settings)


def build_run_identity(
    config: ExperimentConfig, *, split: str, dataset_fingerprint: str
) -> RunIdentity:
    return RunIdentity(
        dataset_fingerprint=dataset_fingerprint,
        split=split,
        model_id=config.findings.model_id,
        model_revision=config.findings.revision,
        max_new_tokens=config.findings.max_new_tokens,
        do_sample=config.findings.do_sample,
        num_beams=config.findings.num_beams,
        section=config.evaluation.section,
    )


def write_cost_estimate(
    output_dir: str | Path,
    *,
    config: ExperimentConfig,
    summary: RunSummary,
    target_samples: int,
    budget: BudgetState,
) -> Path:
    """Record measured throughput and the extrapolated full-split cost.

    Every number here comes from this run. Nothing is assumed or carried over
    from an earlier hardware generation.
    """
    projection = budget.project(target_samples)
    payload = {
        "model": config.findings.model_id,
        "hourly_cost_usd": config.runtime.hourly_cost_usd,
        "pilot_samples": summary.generated,
        "elapsed_hours": round(summary.elapsed_seconds / 3600.0, 6),
        "samples_per_second": round(summary.samples_per_second, 6),
        "target_samples": int(target_samples),
        "projected_hours": round(projection["projected_hours"], 4),
        "projected_cost_usd": round(projection["projected_cost_usd"], 4),
        "impression_cost_not_included": True,
    }
    path = Path(output_dir) / COST_ESTIMATE_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path


def run_findings_inference(
    config: ExperimentConfig,
    records: Sequence[dict[str, Any]],
    output_dir: str | Path,
    *,
    split: str,
    dataset_fingerprint: str,
    target_samples: int = 0,
    reporter_factory: Callable[[], Any] | None = None,
    image_loader: Callable[[str], Any] = load_image,
    clock: Callable[[], float] = time.monotonic,
) -> RunSummary:
    """Generate FINDINGS for ``records``, resuming and staying inside budget."""
    # Guard first: before the manifest, before the model, before any download.
    assert_impression_disabled(
        model_enabled=config.impression.enabled,
        run_impression=config.evaluation.run_impression,
    )

    identity = build_run_identity(
        config, split=split, dataset_fingerprint=dataset_fingerprint
    )
    progress = ProgressFile.open(output_dir, identity)
    summary = RunSummary(requested=len(records))

    with PredictionWriter(output_dir) as writer:
        pending = [r for r in records if not writer.already_done(r["sample_key"])]
        summary.skipped_already_done = len(records) - len(pending)

        budget = BudgetState(
            hourly_cost_usd=config.runtime.hourly_cost_usd,
            budget_limit_usd=config.runtime.budget_limit_usd,
            max_runtime_hours=config.runtime.max_runtime_hours,
            prior_elapsed_seconds=progress.prior_elapsed_seconds(),
            processed_samples=summary.skipped_already_done,
            clock=clock,
        )

        if not pending:
            print(
                f"[runner] nothing to do: all {len(records)} samples already "
                "present in predictions.jsonl",
                flush=True,
            )
            progress.write(
                completed_samples=summary.skipped_already_done,
                elapsed_seconds=budget.elapsed_seconds,
                finished=True,
            )
            summary.elapsed_seconds = budget.elapsed_seconds
            return summary

        # Only now is it worth paying to load a 4B checkpoint.
        reporter = (reporter_factory or (lambda: _build_reporter(config)))()
        resolved_revision = getattr(
            getattr(reporter, "bundle", None),
            "resolved_revision",
            config.findings.revision,
        )

        for processed, record in enumerate(pending, start=1):
            try:
                budget.assert_within_budget()
            except BudgetExceeded as exc:
                print(f"[runner] {exc}", flush=True)
                summary.stopped_on_budget = True
                break

            image = image_loader(record["image_path"])
            generation = reporter.generate(image)
            budget.record_samples()

            prediction = FindingsPrediction(
                sample_key=record["sample_key"],
                findings=generation.findings,
                model_id=config.findings.model_id,
                model_revision=str(resolved_revision),
                elapsed_seconds=generation.elapsed_seconds,
                estimated_cost_usd=budget.estimated_cost_usd,
                warnings=list(generation.warnings),
            )
            writer.write(prediction.to_dict())

            summary.generated += 1
            for warning in generation.warnings:
                summary.warnings_seen[warning] = (
                    summary.warnings_seen.get(warning, 0) + 1
                )

            if config.runtime.log_every and processed % config.runtime.log_every == 0:
                print(budget.progress_line(target_samples), flush=True)
                progress.write(
                    completed_samples=budget.processed_samples,
                    elapsed_seconds=budget.elapsed_seconds,
                )

        summary.elapsed_seconds = budget.elapsed_seconds
        summary.estimated_cost_usd = budget.estimated_cost_usd
        summary.samples_per_second = budget.samples_per_second
        progress.write(
            completed_samples=budget.processed_samples,
            elapsed_seconds=budget.elapsed_seconds,
            finished=not summary.stopped_on_budget,
        )
        write_cost_estimate(
            output_dir,
            config=config,
            summary=summary,
            target_samples=target_samples or len(records),
            budget=budget,
        )

    return summary
