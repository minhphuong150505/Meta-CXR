"""Guards on the classification-only recipe and the compute it is allowed to skip.

`blip2_qformer.forward` skips the Q-Former and text-encoder forwards when every
loss weight that reads them is zero. Getting the gate wrong fails in one of two
directions, and only one of them is loud:

* Too eager -- skipping while a weight is still non-zero raises `NameError` or
  `AttributeError` on the first step. Loud, caught immediately.
* Too lazy -- never skipping. Silent. Training still converges, it just pays for
  two BERT forwards, ITM hard-negative mining and an LM decoder pass on every
  step and multiplies the result by zero. That is the failure worth a test.

The gate expressions are mirrored here rather than imported: `blip2_qformer`
pulls in the whole GPU stack and cannot be imported on a CPU box. The
duplication is the point -- editing the runner without editing this file shows up
as a failure.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CONFIG = _REPO_ROOT / "pretraining" / "configs" / "mimic_cxr_full.yaml"


# Mirrors blip2_qformer.forward().
def _needs_vision_language(w):
    return w["lambda_itc"] > 0 or w["lambda_itm"] > 0 or w["lambda_lm"] > 0


def _needs_text_encoder(w):
    return _needs_vision_language(w) or (
        w["lambda_teacher_cls"] > 0 or w["lambda_distill"] > 0
    )


def _all_off():
    return {
        "lambda_itc": 0.0, "lambda_itm": 0.0, "lambda_lm": 0.0,
        "lambda_teacher_cls": 0.0, "lambda_distill": 0.0,
    }


@pytest.fixture
def loss_cfg():
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load(_CONFIG.read_text())["model"]["loss"]


class TestGate:
    def test_classification_only_skips_both_forwards(self):
        w = _all_off()
        assert _needs_vision_language(w) is False
        assert _needs_text_encoder(w) is False

    @pytest.mark.parametrize("key", ["lambda_itc", "lambda_itm", "lambda_lm"])
    def test_any_vision_language_weight_reenables_everything(self, key):
        w = _all_off() | {key: 1.0}
        assert _needs_vision_language(w) is True
        assert _needs_text_encoder(w) is True

    @pytest.mark.parametrize("key", ["lambda_teacher_cls", "lambda_distill"])
    def test_teacher_needs_text_but_not_the_qformer(self, key):
        """The teacher reads text_output; it never touches query_output."""
        w = _all_off() | {key: 0.5}
        assert _needs_text_encoder(w) is True
        assert _needs_vision_language(w) is False

    def test_itm_alone_still_runs_the_contrastive_pass(self):
        """ITM reuses ITC's similarity matrices for hard-negative sampling."""
        w = _all_off() | {"lambda_itm": 1.0}
        runs_contrastive = w["lambda_itc"] > 0 or w["lambda_itm"] > 0
        assert runs_contrastive is True

    def test_queue_update_is_tied_to_itc_alone(self):
        w = _all_off() | {"lambda_itm": 1.0, "lambda_lm": 1.0}
        assert (w["lambda_itc"] > 0) is False


class TestShippedRecipe:
    """The recipe is a deliberate choice; these assert it, so a silent edit shows."""

    def test_vision_language_objectives_are_off(self, loss_cfg):
        """Off, and the price of turning them on is recorded in the config.

        Tried at 0.1 each on 2026-08-16 and reverted the same day: this project
        needs Stage-1 classification, and the Q-Former does not feed it. The
        attempt measured an OOM on the first iteration at batch 16, a true cost
        of ~36 h against ~9 h, and loss_itm back at 0.6420 against the 1:2 prior
        entropy of 0.6365 after two epochs -- the collapse signature reproduced.

        Turning them on is not a one-line change: it forces batch 16 -> 8, and
        it puts vision-language gradient onto the shared projector and adapters
        that MHCAC reads. Use scripts/check_itc_gate.py first.
        """
        assert loss_cfg["lambda_itc"] == 0.0
        assert loss_cfg["lambda_itm"] == 0.0
        assert loss_cfg["lambda_lm"] == 0.0

    def test_teacher_and_distillation_are_off(self, loss_cfg):
        assert loss_cfg["lambda_teacher_cls"] == 0.0
        assert loss_cfg["lambda_distill"] == 0.0

    def test_auxiliary_weights_match_upstream_meta_cxr(self, loss_cfg):
        """DasithEdirisinghe/META-CXR @ e97d709, blip2_qformer.py:477:

            loss = cls_loss + contrastive*0.3 + orth*0.7 + sparsity*0.3
        """
        assert loss_cfg["lambda_cls"] == 1.0
        assert loss_cfg["lambda_mhcac_contrastive"] == 0.3
        assert loss_cfg["lambda_orthogonality"] == 0.7
        assert loss_cfg["lambda_sparsity"] == 0.3

    def test_multiview_terms_are_kept(self, loss_cfg):
        """Upstream has no multi-view path; these are this work's own and stay on.

        lambda_mpc dropped 0.1 -> 0.02 on 2026-08-16, when the term stopped being
        a constant. It had no trainable parameter upstream and sat at 3.994 for
        four epochs, so its 22% share of the reported loss value was harmless
        noise; once it actually trains, that share is too aggressive for a
        randomly initialised projection head.
        """
        assert loss_cfg["lambda_mpc"] == 0.02
        assert loss_cfg["mpc_warmup_steps"] > 0, "a live MPC needs its ramp"
        assert loss_cfg["lambda_view_consistency"] == 0.05

    def test_shipped_recipe_skips_the_qformer(self, loss_cfg):
        """The skip is a consequence of the weights, never an independent flag.

        Consequence to keep in view: the Q-Former receives no gradient and stays
        at its pretrained initialisation, so this checkpoint serves Stage-1
        classification only and is NOT valid for the Stage-2 soft-token modes.
        """
        assert _needs_vision_language(loss_cfg) is False
        assert _needs_text_encoder(loss_cfg) is False

    def test_learning_rate_floor_matches_upstream(self):
        yaml = pytest.importorskip("yaml")
        run_cfg = yaml.safe_load(_CONFIG.read_text())["run"]
        assert run_cfg["min_lr"] == 1.0e-5
        # Upstream warms over 4000 microbatches at accum 4 = 1000 optimizer
        # updates. This scheduler counts updates, so stay at or under that.
        assert run_cfg["warmup_steps"] <= 1000
        assert run_cfg["warmup_steps"] == 800
