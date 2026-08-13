"""Static guarantees that the external-MedGemma path is inference-only.

SCOPE. These checks apply to the new external-checkpoint packages *only*:
``model/pretrained_medgemma/``, ``medgemma_inference/`` and ``runtime/``.

They are deliberately NOT repository-wide. Stage-1 META-CXR/MHCAC is a
project-owned, trainable research verifier; ``torch.optim``, ``.backward()``
and ``model.train()`` are entirely legitimate under ``pretraining/``, ``mhcac/``
and ``model/lavis/``, and banning them globally would break the contribution
this repository is actually about.
"""

import ast
import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

#: Packages that must contain no training machinery whatsoever.
INFERENCE_ONLY_PACKAGES = (
    "model/pretrained_medgemma",
    "medgemma_inference",
    "runtime",
)

#: Modules the inference path must never reach: they carry the MedGemma
#: fine-tuning implementation.
FORBIDDEN_IMPORTS = {
    "training.trainer",
    "training.trainer.state",
    "training.trainer.checkpointing",
    "training.run_medgemma_qlora",
    "training.train_eval_figure9_llm_variants_200",
}


def inference_only_sources():
    for package in INFERENCE_ONLY_PACKAGES:
        for path in sorted((_REPO_ROOT / package).rglob("*.py")):
            yield path, ast.parse(path.read_text(encoding="utf-8"))


class NoTrainingCodeInInferencePackages(unittest.TestCase):
    def test_no_backward_call(self):
        for path, tree in inference_only_sources():
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    self.assertNotEqual(
                        node.func.attr,
                        "backward",
                        f"{path.relative_to(_REPO_ROOT)} calls .backward()",
                    )

    def test_no_optimizer_construction(self):
        for path, tree in inference_only_sources():
            source = path.read_text(encoding="utf-8")
            relative = path.relative_to(_REPO_ROOT)
            self.assertNotIn("torch.optim", source, f"{relative} builds an optimizer")
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module == "torch.optim":
                    self.fail(f"{relative} imports torch.optim")

    def test_no_peft_training_setup(self):
        forbidden = {"get_peft_model", "prepare_model_for_kbit_training"}
        for path, tree in inference_only_sources():
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    self.assertNotIn(
                        node.func.id,
                        forbidden,
                        f"{path.relative_to(_REPO_ROOT)} sets up LoRA training",
                    )

    def test_never_puts_a_model_into_train_mode(self):
        for path, tree in inference_only_sources():
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "train"
                    and not node.args
                ):
                    self.fail(f"{path.relative_to(_REPO_ROOT)} calls .train()")

    def test_does_not_import_medgemma_finetuning_modules(self):
        for path, tree in inference_only_sources():
            relative = path.relative_to(_REPO_ROOT)
            for node in ast.walk(tree):
                module = None
                if isinstance(node, ast.ImportFrom):
                    module = node.module
                elif isinstance(node, ast.Import):
                    module = node.names[0].name
                if module and module in FORBIDDEN_IMPORTS:
                    self.fail(f"{relative} imports fine-tuning module {module}")

    def test_loader_never_falls_back_to_a_text_only_class(self):
        source = (
            _REPO_ROOT / "model" / "pretrained_medgemma" / "findings_loader.py"
        ).read_text(encoding="utf-8")
        # Mentioned once, in the comment explaining why it is absent.
        self.assertEqual(source.count("AutoModelForCausalLM"), 1)
        self.assertIn("deliberately absent", source)

    def test_generation_runs_without_grad(self):
        source = (
            _REPO_ROOT / "model" / "pretrained_medgemma" / "findings_reporter.py"
        ).read_text(encoding="utf-8")
        self.assertTrue(
            "torch.inference_mode()" in source or "torch.no_grad()" in source
        )

    def test_loader_puts_the_model_in_eval_mode(self):
        source = (
            _REPO_ROOT / "model" / "pretrained_medgemma" / "findings_loader.py"
        ).read_text(encoding="utf-8")
        self.assertIn("model.eval()", source)


class ActiveInferenceConfigHasNoTrainingKeys(unittest.TestCase):
    def test_experiment_config_carries_no_learning_rate(self):
        import yaml

        from medgemma_inference.config import OBSOLETE_FINETUNING_KEYS

        for path in sorted((_REPO_ROOT / "configs" / "experiments").glob("*.yaml")):
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            stack = [raw]
            while stack:
                node = stack.pop()
                if isinstance(node, dict):
                    for key, value in node.items():
                        self.assertNotIn(
                            str(key),
                            OBSOLETE_FINETUNING_KEYS,
                            f"{path.name} carries fine-tuning key {key!r}",
                        )
                        stack.append(value)
                elif isinstance(node, list):
                    stack.extend(node)


class ExternalModesAreNotRunnableFromTheFineTuningCli(unittest.TestCase):
    def test_findings_mode_is_rejected_by_the_stage2_resolver(self):
        from training import pipeline_modes as pm

        with self.assertRaises(ValueError) as caught:
            pm.resolve_pipeline_modes("pretrained_medgemma_findings_first")
        self.assertIn("medgemma_inference.run_pretrained_findings", str(caught.exception))

    def test_external_modes_are_absent_from_stage2_cli_choices(self):
        from training import pipeline_modes as pm

        for name in pm.EXTERNAL_INFERENCE_MODES:
            self.assertNotIn(name, pm.CHOICES)

    def test_phase2_impression_mode_is_not_runnable(self):
        from training import pipeline_modes as pm

        with self.assertRaises(ValueError):
            pm.resolve_pipeline_modes("pretrained_medgemma_impression_phase2")


class Stage1RemainsIntact(unittest.TestCase):
    """Phase A must not have touched the project-owned research verifier."""

    def test_stage1_training_entrypoints_still_exist(self):
        for relative in (
            "pretraining/train.py",
            "pretraining/precompute_features.py",
            "pretraining/configs/mimic_cxr_full.yaml",
        ):
            self.assertTrue(
                (_REPO_ROOT / relative).is_file(), f"{relative} was removed"
            )

    def test_stage1_packages_still_exist(self):
        for relative in ("mhcac", "biovil_t", "vision_encoders", "model/lavis"):
            self.assertTrue(
                (_REPO_ROOT / relative).is_dir(), f"{relative} was removed"
            )

    def test_medgemma_finetuning_files_are_untouched_in_phase_a(self):
        # Phase A is additive. These are scheduled for removal only after a
        # successful GPU pilot; see docs/pending_medgemma_finetuning_teardown.md
        for relative in (
            "training/run_medgemma_qlora.py",
            "training/train_eval_figure9_llm_variants_200.py",
        ):
            self.assertTrue(
                (_REPO_ROOT / relative).is_file(),
                f"{relative} was deleted during Phase A; teardown is Phase B",
            )


class DocumentedCommandsMatchTheCli(unittest.TestCase):
    def test_readme_pilot_command_parses(self):
        from medgemma_inference.run_pretrained_findings import parse_args

        args = parse_args(
            [
                "--config",
                "configs/experiments/pretrained_medgemma_findings_first.yaml",
                "--split",
                "validation",
                "--max-samples",
                "100",
                "--estimate-full-cost",
            ]
        )
        self.assertEqual(args.split, "validation")
        self.assertEqual(args.max_samples, "100")
        self.assertTrue(args.estimate_full_cost)

    def test_full_run_command_parses_and_needs_confirmation(self):
        from medgemma_inference.run_pretrained_findings import parse_args

        args = parse_args(
            [
                "--config",
                "configs/experiments/pretrained_medgemma_findings_first.yaml",
                "--split",
                "test",
                "--max-samples",
                "all",
                "--confirm-full-run",
            ]
        )
        self.assertEqual(args.max_samples, "all")
        self.assertTrue(args.confirm_full_run)

    def test_full_run_is_refused_without_confirmation(self):
        from medgemma_inference.run_pretrained_findings import main

        code = main(
            [
                "--config",
                str(
                    _REPO_ROOT
                    / "configs/experiments/pretrained_medgemma_findings_first.yaml"
                ),
                "--split",
                "test",
                "--max-samples",
                "all",
                "--csv",
                str(_REPO_ROOT / "does-not-exist.csv"),
            ]
        )
        # Exits non-zero without running anything.
        self.assertNotEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
