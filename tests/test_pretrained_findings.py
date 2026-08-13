"""Phase-1 invariants for the external-MedGemma Findings pipeline.

This box has no GPU and no transformers, so the model itself is a fake. What is
verified here is everything *around* the model: the Impression guard, the
budget, resume, privacy, output schema and postprocessing. Nothing in this file
demonstrates that MedGemma runs -- that is the GPU pilot's job, and no test
here should ever be cited as evidence that it does.
"""

import ast
import json
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from medgemma_inference import config as cfg  # noqa: E402
from medgemma_inference.prediction_writer import (  # noqa: E402
    PredictionWriter,
    PrivacyViolation,
    read_completed_keys,
)
from medgemma_inference.progress import (  # noqa: E402
    ProgressFile,
    ResumeMismatch,
    RunIdentity,
)
from medgemma_inference.runner import run_findings_inference  # noqa: E402
from model.pretrained_medgemma import output_schema  # noqa: E402
from model.pretrained_medgemma.errors import ImpressionPhaseDisabledError  # noqa: E402
from model.pretrained_medgemma.impression_reporter import (  # noqa: E402
    PretrainedImpressionReporter,
    assert_impression_disabled,
)
from runtime.budget import BudgetExceeded, BudgetState  # noqa: E402


@dataclass
class FakeGeneration:
    findings: str
    warnings: list
    elapsed_seconds: float = 0.01


class FakeReporter:
    """Stands in for the real reporter; records how often it was asked to run."""

    def __init__(self, text="Lungs are clear.", clock=None, seconds_per_sample=0.0):
        self.text = text
        self.calls = 0
        self._clock = clock
        self._seconds = seconds_per_sample

    def generate(self, image):
        self.calls += 1
        if self._clock is not None:
            self._clock.advance(self._seconds)
        findings, warnings = output_schema.postprocess_findings(self.text)
        return FakeGeneration(findings=findings, warnings=warnings)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def make_config(**overrides):
    raw = {
        "pipeline": {"mode": cfg.PIPELINE_MODE},
        "models": {
            "findings": {"model_id": "erjui/medgemma-4b-srrg-findings"},
            "impression": {"enabled": False},
        },
        "runtime": {"hourly_cost_usd": 1.0, "budget_limit_usd": 100.0, "log_every": 0},
        "evaluation": {"run_impression": False},
    }
    for key, value in overrides.items():
        raw.setdefault(key, {}).update(value)
    return cfg.parse_config(raw, source="<test>")


def make_records(count, start=0):
    return [
        {
            "index": index,
            "sample_key": f"key{index:04d}",
            "ref": "reference report text",
            "image_path": f"/nowhere/{index}.jpg",
        }
        for index in range(start, start + count)
    ]


class ImpressionIsDisabled(unittest.TestCase):
    def test_guard_blocks_enabled_model(self):
        with self.assertRaises(ImpressionPhaseDisabledError):
            assert_impression_disabled(model_enabled=True, run_impression=False)

    def test_guard_blocks_run_impression(self):
        with self.assertRaises(ImpressionPhaseDisabledError):
            assert_impression_disabled(model_enabled=False, run_impression=True)

    def test_guard_allows_phase_one(self):
        assert_impression_disabled(model_enabled=False, run_impression=False)

    def test_reporter_cannot_be_constructed(self):
        with self.assertRaises(ImpressionPhaseDisabledError):
            PretrainedImpressionReporter()

    def test_module_import_touches_no_model_stack(self):
        # Importing the Phase-2 module must not pull in transformers or torch,
        # and must contain no download call. Checked against the parsed AST so
        # prose in the docstring cannot satisfy or break it.
        import model.pretrained_medgemma.impression_reporter as impression

        tree = ast.parse(Path(impression.__file__).read_text(encoding="utf-8"))
        imported = set()
        called = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                called.add(node.func.attr)
        self.assertNotIn("torch", imported)
        self.assertNotIn("transformers", imported)
        self.assertNotIn("peft", imported)
        self.assertNotIn("from_pretrained", called)

    def test_config_rejects_enabled_impression(self):
        with self.assertRaises(cfg.ConfigError):
            make_config(models={"impression": {"enabled": True}})

    def test_runner_guard_runs_before_any_model_load(self):
        # reporter_factory would raise if called; the guard must fire first.
        def exploding_factory():
            raise AssertionError("a model was loaded despite Impression being on")

        raw_config = make_config()
        bad = cfg.ExperimentConfig(
            pipeline_mode=raw_config.pipeline_mode,
            findings=raw_config.findings,
            impression=cfg.ImpressionModelConfig(enabled=True),
            runtime=raw_config.runtime,
            evaluation=raw_config.evaluation,
            privacy=raw_config.privacy,
        )
        with self.assertRaises(ImpressionPhaseDisabledError):
            run_findings_inference(
                bad,
                make_records(1),
                "/tmp/never-created",
                split="val",
                dataset_fingerprint="fp",
                reporter_factory=exploding_factory,
            )


class ObsoleteFineTuningConfigIsRejected(unittest.TestCase):
    def test_legacy_training_block_is_rejected(self):
        with self.assertRaises(cfg.ObsoleteFineTuningConfigError):
            cfg.parse_config(
                {
                    "pipeline": {"mode": cfg.PIPELINE_MODE},
                    "training": {"epochs": 1, "learning_rate": 0.0001},
                },
                source="<legacy>",
            )

    def test_nested_lora_keys_are_rejected(self):
        with self.assertRaises(cfg.ObsoleteFineTuningConfigError):
            cfg.parse_config(
                {
                    "pipeline": {"mode": cfg.PIPELINE_MODE},
                    "models": {"findings": {"lora_rank": 8}},
                },
                source="<legacy>",
            )

    def test_error_names_the_offending_key_and_explains(self):
        with self.assertRaises(cfg.ObsoleteFineTuningConfigError) as caught:
            cfg.parse_config(
                {"pipeline": {"mode": cfg.PIPELINE_MODE}, "optimizer": "adamw"},
                source="<legacy>",
            )
        message = str(caught.exception)
        self.assertIn("optimizer", message)
        self.assertIn("inference-only", message)

    def test_stage1_training_config_is_not_parsed_here(self):
        # Guard against scope creep: this validator must never be pointed at a
        # Stage-1 YAML, whose learning_rate is entirely legitimate.
        stage1 = _REPO_ROOT / "pretraining" / "configs" / "mimic_cxr_full.yaml"
        self.assertTrue(stage1.is_file())
        text = stage1.read_text(encoding="utf-8")
        self.assertNotIn(cfg.PIPELINE_MODE, text)


class ShippedConfigIsValid(unittest.TestCase):
    def test_repository_config_loads_and_disables_impression(self):
        config = cfg.load_config(
            _REPO_ROOT / "configs" / "experiments"
            / "pretrained_medgemma_findings_first.yaml"
        )
        self.assertTrue(config.findings.enabled)
        self.assertFalse(config.impression.enabled)
        self.assertFalse(config.evaluation.run_impression)
        self.assertEqual(config.evaluation.section, "findings")
        self.assertEqual(config.findings.model_id, "erjui/medgemma-4b-srrg-findings")

    def test_privacy_flags_cannot_be_turned_on(self):
        with self.assertRaises(cfg.ConfigError):
            make_config(privacy={"save_identifiers": True})

    def test_unknown_key_is_rejected_not_ignored(self):
        with self.assertRaises(cfg.ConfigError):
            make_config(models={"findings": {"temperture": 0.7}})


class OutputSchema(unittest.TestCase):
    def test_phase_one_nulls_impression_and_full_report(self):
        record = output_schema.FindingsPrediction(
            sample_key="abc",
            findings="Lungs are clear.",
            model_id="erjui/medgemma-4b-srrg-findings",
            model_revision="deadbeef",
        ).to_dict()
        self.assertIsNone(record["impression"])
        self.assertIsNone(record["full_report"])
        self.assertFalse(record["impression_enabled"])

    def test_provenance_is_honest(self):
        record = output_schema.FindingsPrediction(
            sample_key="abc", findings="x", model_id="m", model_revision="r"
        ).to_dict()
        self.assertFalse(record["fine_tuned_by_this_project"])
        self.assertTrue(record["external_checkpoint"])

    def test_no_training_fields_survive(self):
        record = output_schema.FindingsPrediction(
            sample_key="abc", findings="x", model_id="m", model_revision="r"
        ).to_dict()
        for field in (
            "training_epoch",
            "global_step",
            "optimizer_step",
            "train_loss",
            "learning_rate",
            "best_checkpoint",
            "training_run_id",
        ):
            self.assertNotIn(field, record)

    def test_unexpected_impression_is_dropped_and_warned(self):
        raw = "FINDINGS: Lungs are clear.\n\nIMPRESSION: No acute process."
        findings, warnings = output_schema.postprocess_findings(raw)
        self.assertEqual(findings, "Lungs are clear.")
        self.assertNotIn("No acute process", findings)
        self.assertIn(output_schema.WARN_UNEXPECTED_IMPRESSION, warnings)

    def test_findings_header_is_stripped(self):
        findings, warnings = output_schema.postprocess_findings("FINDINGS: Clear.")
        self.assertEqual(findings, "Clear.")
        self.assertEqual(warnings, [])

    def test_blank_generation_is_warned(self):
        findings, warnings = output_schema.postprocess_findings("   ")
        self.assertEqual(findings, "")
        self.assertIn(output_schema.WARN_EMPTY_FINDINGS, warnings)


class Budget(unittest.TestCase):
    def test_cost_tracks_wall_clock(self):
        clock = FakeClock()
        budget = BudgetState(
            hourly_cost_usd=2.0, budget_limit_usd=10.0, clock=clock
        )
        clock.advance(1800)  # half an hour
        self.assertAlmostEqual(budget.estimated_cost_usd, 1.0, places=6)

    def test_limit_raises(self):
        clock = FakeClock()
        budget = BudgetState(hourly_cost_usd=10.0, budget_limit_usd=1.0, clock=clock)
        budget.assert_within_budget()
        clock.advance(3600)
        with self.assertRaises(BudgetExceeded):
            budget.assert_within_budget()

    def test_runtime_ceiling_raises(self):
        clock = FakeClock()
        budget = BudgetState(
            hourly_cost_usd=0.0,
            budget_limit_usd=0.0,
            max_runtime_hours=1.0,
            clock=clock,
        )
        clock.advance(3601)
        with self.assertRaises(BudgetExceeded):
            budget.assert_within_budget()

    def test_prior_elapsed_makes_the_ceiling_bind_across_resumes(self):
        # Without this, every resume would restart from a zeroed budget and the
        # limit would never be reached.
        clock = FakeClock()
        budget = BudgetState(
            hourly_cost_usd=1.0,
            budget_limit_usd=1.0,
            prior_elapsed_seconds=3500,
            clock=clock,
        )
        clock.advance(200)
        with self.assertRaises(BudgetExceeded):
            budget.assert_within_budget()

    def test_projection_is_zero_before_any_sample(self):
        # No invented throughput numbers.
        budget = BudgetState(hourly_cost_usd=1.0, budget_limit_usd=1.0)
        projection = budget.project(1000)
        self.assertEqual(projection["projected_hours"], 0.0)
        self.assertEqual(projection["projected_cost_usd"], 0.0)

    def test_projection_extrapolates_measured_rate(self):
        clock = FakeClock()
        budget = BudgetState(
            hourly_cost_usd=2.0, budget_limit_usd=1000.0, clock=clock
        )
        clock.advance(10)
        budget.record_samples(10)  # 1 sample/second
        projection = budget.project(3600)
        self.assertAlmostEqual(projection["projected_hours"], 1.0, places=6)
        self.assertAlmostEqual(projection["projected_cost_usd"], 2.0, places=6)


class Privacy(unittest.TestCase):
    def test_writer_rejects_identifiers(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            with PredictionWriter(directory) as writer:
                for bad in ("subject_id", "study_id", "dicom_id", "image_path", "ref"):
                    with self.assertRaises(PrivacyViolation):
                        writer.write({"sample_key": "k", "findings": "x", bad: "v"})

    def test_written_records_carry_no_identifiers(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            config = make_config()
            run_findings_inference(
                config,
                make_records(3),
                directory,
                split="val",
                dataset_fingerprint="fp",
                reporter_factory=FakeReporter,
                image_loader=lambda path: object(),
            )
            lines = (Path(directory) / "predictions.jsonl").read_text().splitlines()
            self.assertEqual(len(lines), 3)
            for line in lines:
                record = json.loads(line)
                self.assertNotIn("image_path", record)
                self.assertNotIn("ref", record)
                self.assertNotIn("subject_id", record)
                self.assertTrue(record["sample_key"].startswith("key"))


class ResumeInference(unittest.TestCase):
    def test_completed_samples_are_not_regenerated(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            config = make_config()
            records = make_records(5)
            first = FakeReporter()
            run_findings_inference(
                config,
                records[:3],
                directory,
                split="val",
                dataset_fingerprint="fp",
                reporter_factory=lambda: first,
                image_loader=lambda path: object(),
            )
            self.assertEqual(first.calls, 3)

            second = FakeReporter()
            summary = run_findings_inference(
                config,
                records,
                directory,
                split="val",
                dataset_fingerprint="fp",
                reporter_factory=lambda: second,
                image_loader=lambda path: object(),
            )
            # Only the two new records are generated.
            self.assertEqual(second.calls, 2)
            self.assertEqual(summary.skipped_already_done, 3)
            self.assertEqual(summary.generated, 2)
            lines = (Path(directory) / "predictions.jsonl").read_text().splitlines()
            self.assertEqual(len(lines), 5)

    def test_fully_resumed_run_loads_no_model(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            config = make_config()
            records = make_records(2)
            run_findings_inference(
                config,
                records,
                directory,
                split="val",
                dataset_fingerprint="fp",
                reporter_factory=FakeReporter,
                image_loader=lambda path: object(),
            )

            def exploding_factory():
                raise AssertionError("model loaded for an already-complete run")

            summary = run_findings_inference(
                config,
                records,
                directory,
                split="val",
                dataset_fingerprint="fp",
                reporter_factory=exploding_factory,
                image_loader=lambda path: object(),
            )
            self.assertEqual(summary.generated, 0)
            self.assertEqual(summary.skipped_already_done, 2)

    def test_changed_model_refuses_to_resume(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            identity = RunIdentity(
                dataset_fingerprint="fp",
                split="val",
                model_id="erjui/medgemma-4b-srrg-findings",
                model_revision="main",
                max_new_tokens=512,
                do_sample=False,
                num_beams=1,
            )
            progress = ProgressFile.open(directory, identity)
            progress.write(completed_samples=1, elapsed_seconds=5.0)

            changed = RunIdentity(
                dataset_fingerprint="fp",
                split="val",
                model_id="some/other-model",
                model_revision="main",
                max_new_tokens=512,
                do_sample=False,
                num_beams=1,
            )
            with self.assertRaises(ResumeMismatch):
                ProgressFile.open(directory, changed)

    def test_partial_trailing_line_is_repaired(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            path.write_text(
                json.dumps({"sample_key": "a", "findings": "x"}) + "\n"
                + '{"sample_key": "b", "findi'
            )
            completed = read_completed_keys(path)
            self.assertEqual(completed, {"a"})
            # The fragment is gone, so the file stays parseable.
            for line in path.read_text().splitlines():
                json.loads(line)


class BudgetStopsRunCleanly(unittest.TestCase):
    def test_run_stops_and_keeps_valid_output(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            clock = FakeClock()
            # $1/h with a $0.01 limit: 36 seconds of budget.
            config = make_config(
                runtime={"hourly_cost_usd": 1.0, "budget_limit_usd": 0.01}
            )
            reporter = FakeReporter(clock=clock, seconds_per_sample=20.0)
            summary = run_findings_inference(
                config,
                make_records(10),
                directory,
                split="val",
                dataset_fingerprint="fp",
                reporter_factory=lambda: reporter,
                image_loader=lambda path: object(),
                clock=clock,
            )
            self.assertTrue(summary.stopped_on_budget)
            self.assertLess(summary.generated, 10)
            self.assertGreater(summary.generated, 0)
            # Everything generated before the stop is on disk and parseable.
            lines = (Path(directory) / "predictions.jsonl").read_text().splitlines()
            self.assertEqual(len(lines), summary.generated)
            for line in lines:
                json.loads(line)
            # And the run can be resumed from exactly where it stopped.
            self.assertTrue((Path(directory) / "progress.json").is_file())

    def test_cost_estimate_is_written_from_measured_rate(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            clock = FakeClock()
            config = make_config(
                runtime={"hourly_cost_usd": 3.6, "budget_limit_usd": 1000.0}
            )
            reporter = FakeReporter(clock=clock, seconds_per_sample=1.0)
            run_findings_inference(
                config,
                make_records(5),
                directory,
                split="val",
                dataset_fingerprint="fp",
                target_samples=100,
                reporter_factory=lambda: reporter,
                image_loader=lambda path: object(),
                clock=clock,
            )
            payload = json.loads(
                (Path(directory) / "cost_estimate_findings.json").read_text()
            )
            self.assertEqual(payload["pilot_samples"], 5)
            self.assertEqual(payload["target_samples"], 100)
            self.assertTrue(payload["impression_cost_not_included"])
            self.assertGreater(payload["projected_cost_usd"], 0.0)
            self.assertEqual(payload["model"], "erjui/medgemma-4b-srrg-findings")


if __name__ == "__main__":
    unittest.main()
