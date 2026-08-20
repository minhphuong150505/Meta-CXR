"""Partial encoder fine-tuning must fail loudly, never silently.

The failure this guards against is specific: a pattern that matches no
parameter unfreezes nothing, and the resulting run looks exactly like a
completed run that simply did not improve. On this hardware that is 12.5 GPU
hours to discover a spelling mistake.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[1]
_QFORMER = _ROOT / "model" / "lavis" / "models" / "blip2_models" / "blip2_qformer.py"
_RUNNER = _ROOT / "model" / "lavis" / "runners" / "runner_base.py"
_CONFIG = _ROOT / "pretraining" / "configs" / "mimic_cxr_full.yaml"


def _function(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name}() not found in {path.name}")


class TestMechanism:
    def test_unmatched_pattern_raises(self):
        fn = _function(_QFORMER, "apply_encoder_finetune")
        raises = [n for n in ast.walk(fn) if isinstance(n, ast.Raise)]
        assert len(raises) >= 3, (
            "apply_encoder_finetune must raise on an unmatched pattern, on "
            "enabled-with-empty-patterns, and on a non-mapping config"
        )

    def test_from_config_actually_calls_it(self):
        source = _QFORMER.read_text(encoding="utf-8")
        assert "model.apply_encoder_finetune(cfg.get(\"encoder_finetune\", None))" in source

    def test_train_override_keeps_encoders_in_eval(self):
        fn = _function(_QFORMER, "train")
        body = ast.dump(fn)
        assert "_encoder_finetune_keep_bn_eval" in body
        assert "eval" in body


class TestOptimizerGrouping:
    def test_encoder_group_exists_and_has_its_own_lr(self):
        source = _RUNNER.read_text(encoding="utf-8")
        assert '"encoder_decay": []' in source
        assert '"encoder_no_decay": []' in source
        assert 'init_lr_enc' in source

    def test_named_but_unreached_encoder_params_raise(self):
        """A silent name-mapping break would train the encoder at init_lr."""
        fn = _function(_RUNNER, "optimizer")
        raises = [n for n in ast.walk(fn) if isinstance(n, ast.Raise)]
        messages = " ".join(
            ast.dump(n) for n in raises
        )
        assert "reached the optimizer" in messages

    def test_encoder_wins_over_the_classifier_token_match(self):
        """`is_classifier` is a substring test over the whole parameter name.

        An encoder parameter must be routed by exact membership first, or a
        name that happens to contain one of those tokens would silently land in
        the classifier group at init_lr_cls.
        """
        source = _RUNNER.read_text(encoding="utf-8")
        assert "is_classifier = not is_encoder and any(" in source


class TestShippedConfig:
    @pytest.fixture(scope="class")
    def cfg(self):
        return yaml.safe_load(_CONFIG.read_text(encoding="utf-8"))

    def test_encoder_finetune_is_on_with_patterns(self, cfg):
        block = cfg["model"]["encoder_finetune"]
        assert block["enabled"] is True
        assert len(block["patterns"]) == 5
        assert block["keep_batchnorm_eval"] is True

    def test_clip_text_tower_is_never_unfrozen(self, cfg):
        """63.17M parameters this project never runs."""
        for pattern in cfg["model"]["encoder_finetune"]["patterns"]:
            assert not pattern.startswith("pubmedclip.model.text_model")

    def test_encoder_lr_is_well_below_the_heads(self, cfg):
        run = cfg["run"]
        assert float(run["init_lr_enc"]) <= float(run["init_lr"]) / 5, (
            "a pretrained encoder fine-tuned at the head learning rate is "
            "overwritten, which is strictly worse than leaving it frozen"
        )

    def test_every_kappa_is_one(self, cfg):
        """w_positive must be pure inverse frequency: (n_neg/n_pos), no kappa.

        Checked against the counts the gate table carries, which are the same
        studies. A kappa creeping back in shows up as w_pos exceeding the
        label's own ratio.
        """
        weights = cfg["model"]["mhcac"]["class_weights"]
        assert len(weights) == 14
        for row in weights:
            assert len(row) == 3
            assert row[0] == 1.0
        # Pneumothorax was the one label pinned at the 10.0 cap by kappa 4.
        assert weights[9][1] == pytest.approx(4.060), (
            "Pneumothorax should sit at its raw ratio once kappa is 1, not at "
            "the cap"
        )

    def test_gate_kappa_is_one(self, cfg):
        """gate weight = n_not_mentioned / n_mentioned, capped at 10, no kappa."""
        gate = cfg["model"]["mhcac"]["gate_class_weights"]
        assert len(gate) == 14
        counts = {  # (mentioned, not mentioned), from the config's own comments
            0: (74305, 148453),
            2: (65084, 157674),
            10: (85013, 137745),
            13: (68520, 154238),
        }
        for index, (mentioned, not_mentioned) in counts.items():
            expected = min(not_mentioned / mentioned, 10.0)
            assert gate[index] == pytest.approx(expected, abs=1e-3), (
                f"gate weight {index} is not plain inverse frequency; a kappa "
                "has crept back in"
            )
