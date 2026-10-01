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

    # The DEEP unfreeze shipped by run_20260821_deep: ResNet50 layer3 + layer4 +
    # projector, and CLIP vision blocks 8-11 + post_layernorm. 53.12M of 181.3M
    # encoder parameters.
    #
    # Pinned as a SET, not a count. This assertion used to read
    # ``len(...) == 5`` -- the shallow unfreeze of run_20260820_ft -- and went
    # stale in silence when 814b778 deepened the config to 8 patterns. A count
    # tells you only that the number moved; the set names which slice moved,
    # which is the thing a reviewer actually has to agree to.
    DEEP_UNFREEZE_PATTERNS = frozenset(
        {
            "visual_encoder.encoder.encoder.layer3",
            "visual_encoder.encoder.encoder.layer4",
            "visual_encoder.projector",
            "pubmedclip.model.vision_model.encoder.layers.8",
            "pubmedclip.model.vision_model.encoder.layers.9",
            "pubmedclip.model.vision_model.encoder.layers.10",
            "pubmedclip.model.vision_model.encoder.layers.11",
            "pubmedclip.model.vision_model.post_layernorm",
        }
    )

    def test_encoder_finetune_is_on_with_patterns(self, cfg):
        block = cfg["model"]["encoder_finetune"]
        assert block["enabled"] is True
        assert set(block["patterns"]) == self.DEEP_UNFREEZE_PATTERNS
        # A duplicated pattern would survive the set comparison and then make
        # apply_encoder_finetune do the same work twice.
        assert len(block["patterns"]) == len(self.DEEP_UNFREEZE_PATTERNS)
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

    def test_classification_uses_logit_adjustment_from_train_counts(self, cfg):
        """Logit adjustment replaced the capped class_weights on 2026-10-01 (D-025).

        Counts are train, study level, `blank_label_policy: negative`
        (2026-09-24, scripts/count_chexpert_blank_policy.py), as
        [n_negative, n_positive, n_uncertain]. Every row covers the same
        220,379 studies with CheXpert information.
        """
        mhcac = cfg["model"]["mhcac"]
        assert mhcac["blank_label_policy"] == "negative"
        assert "class_weights" not in mhcac, "class_weights and logit_adjustment are exclusive"
        adj = mhcac["logit_adjustment"]
        assert float(adj["tau"]) == 1.0
        counts = adj["class_counts"]
        assert len(counts) == 14
        assert all(len(row) == 3 and sum(row) == 220379 for row in counts)
        expected = {  # index: (n_neg, n_pos, n_unc)
            0: (146074, 74305, 0),      # No Finding -- never uncertain
            2: (170908, 43602, 5869),   # Cardiomegaly
            5: (181509, 26093, 12777),  # Edema
            8: (165620, 44718, 10041),  # Atelectasis
            9: (209102, 10171, 1106),   # Pneumothorax
            11: (217703, 1933, 743),    # Pleural Other
            13: (155281, 64868, 230),   # Support Devices
        }
        for index, row in expected.items():
            assert tuple(counts[index]) == row, index
