"""Three-phase schedule (pretraining/phases.py) against the shipped YAML.

The model here is a stand-in with the real Blip2Qformer module NAMES, so the
prefix rules are checked against the names they must match, without building
the GPU stack. The runner-side pieces that need the real RunnerBase live at
the bottom and skip on a box without the training stack.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from pretraining.phases import (
    PhaseConfigError,
    apply_phase_to_config,
    apply_trainable,
    build_spec,
    checkpoint_keep,
    param_role,
    resolve_init_checkpoint,
    transition_multipliers,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "pretraining/configs/mimic_cxr_full.yaml"


def _config():
    from omegaconf import OmegaConf

    return OmegaConf.load(CONFIG)


def _spec(name):
    config = _config()
    return apply_phase_to_config(config, name), config


class FakeBlip2(nn.Module):
    """Parameter names as Blip2Qformer has them (one tensor per module)."""

    def __init__(self):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.randn(1, 4, 8))
        self.Qformer = nn.ModuleDict({"bert": nn.Linear(8, 8)})
        self.vision_proj = nn.Linear(8, 4)
        self.text_proj = nn.Linear(8, 4)
        self.itm_head = nn.Linear(8, 2)
        self.temp = nn.Parameter(torch.tensor(0.07))
        self.siglip_logit_scale = nn.Parameter(torch.tensor(2.3))
        self.siglip_bias = nn.Parameter(torch.tensor(-10.0))
        self.shared_visual_projector = nn.ModuleDict({"projections": nn.Linear(8, 8)})
        self.ln_vision = nn.LayerNorm(8)
        self.visual_encoder = nn.ModuleDict(
            {"encoder": nn.Linear(8, 8), "projector": nn.Linear(8, 8)}
        )
        self.pubmedclip = nn.Linear(8, 8)
        self.swin = nn.Linear(8, 8)
        self.mhcac = nn.Linear(8, 8)
        self.stream_adapters = nn.Linear(8, 8)
        self.view_fusion = nn.Linear(8, 8)
        self.mpc_heads = nn.Linear(8, 8)

    def forward(self, x):
        parts = [
            self.Qformer["bert"](x), self.vision_proj(x).sum(-1, keepdim=True),
            self.text_proj(x).sum(-1, keepdim=True), self.itm_head(x).sum(-1, keepdim=True),
            self.shared_visual_projector["projections"](x), self.ln_vision(x),
            self.visual_encoder["encoder"](x), self.visual_encoder["projector"](x),
            self.pubmedclip(x), self.swin(x), self.mhcac(x), self.stream_adapters(x),
            self.view_fusion(x), self.mpc_heads(x),
        ]
        return (sum(p.sum() for p in parts) + self.query_tokens.sum() + self.temp
                + self.siglip_logit_scale + self.siglip_bias)


# --------------------------------------------------------------------------
# Config plumbing
# --------------------------------------------------------------------------


def test_the_shipped_yaml_defines_the_three_paper_phases():
    config = _config()
    assert list(config.run.phases) == ["phase1a", "phase1b", "phase1c"]
    assert config.run.phase is None


def test_a_phased_yaml_refuses_to_run_as_one_phase():
    with pytest.raises(PhaseConfigError, match="pass run.phase"):
        apply_phase_to_config(_config(), None)
    with pytest.raises(PhaseConfigError, match="not one of"):
        apply_phase_to_config(_config(), "phase9")


def test_no_phases_block_means_the_historical_single_phase():
    from omegaconf import OmegaConf

    config = OmegaConf.create({"run": {"max_epoch": 10}, "model": {}, "datasets": {}})
    assert apply_phase_to_config(config, None) is None
    assert config.run.max_epoch == 10
    with pytest.raises(PhaseConfigError, match="no run.phases"):
        apply_phase_to_config(config, "phase1a")
    # Unphased checkpoints keep exactly what they always kept.
    assert checkpoint_keep("mhcac.w", True, None) is True
    assert checkpoint_keep("mhcac.w", False, None) is False


@pytest.mark.parametrize(
    ("phase", "epochs", "itc", "cls", "batch", "aug"),
    [("phase1a", 4, 1.0, 0.0, 128, False), ("phase1b", 5, 0.0, 1.0, 16, True),
     ("phase1c", 1, 1.0, 1.0, 8, True)],
)
def test_phase_overrides_reach_the_config(phase, epochs, itc, cls, batch, aug):
    spec, config = _spec(phase)
    assert spec.epochs == epochs == config.run.max_epoch
    assert config.model.loss.lambda_itc == itc and config.model.loss.lambda_cls == cls
    assert config.run.batch_size_train == batch
    assert config.datasets.mimic_cxr.vis_processor.train.augmentation.enabled is aug
    # Encoder blocks stay frozen except in 1c, where the user reopened the
    # shallow set of run_20260820_ft (D-021).
    assert config.model.encoder_finetune.enabled is (phase == "phase1c")
    if phase == "phase1c":
        assert len(config.model.encoder_finetune.patterns) == 5
    # The paper's MHCAC has no teacher in any phase.
    assert config.model.loss.lambda_teacher_cls == 0 and config.model.loss.lambda_distill == 0


def test_phase1a_optimises_only_itc_itm_lm():
    _, config = _spec("phase1a")
    loss = config.model.loss
    on = {k for k, v in loss.items() if k.startswith("lambda_") and float(v) > 0}
    assert on == {"lambda_itc", "lambda_itm", "lambda_lm"}


def test_phase1b_uses_the_papers_learning_rates():
    _, config = _spec("phase1b")
    assert config.run.warmup_lr == pytest.approx(5e-5)
    assert config.run.init_lr == pytest.approx(2e-4)
    assert config.run.min_lr == pytest.approx(1e-5)
    assert config.run.max_epoch == 5


def test_init_checkpoint_chain(tmp_path):
    assert resolve_init_checkpoint(_spec("phase1a")[0], tmp_path) is None
    assert resolve_init_checkpoint(_spec("phase1b")[0], tmp_path) == tmp_path / "checkpoint_phase1a.pth"
    assert resolve_init_checkpoint(_spec("phase1c")[0], tmp_path) == tmp_path / "checkpoint_phase1b.pth"
    with pytest.raises(PhaseConfigError, match="phase_root"):
        resolve_init_checkpoint(_spec("phase1b")[0], "")


# --------------------------------------------------------------------------
# requires_grad per phase
# --------------------------------------------------------------------------

EXPECTED_TRAINABLE = {
    "phase1a": {"query_tokens", "Qformer", "vision_proj", "text_proj", "itm_head", "temp",
                "siglip_logit_scale", "siglip_bias", "shared_visual_projector", "ln_vision"},
    "phase1b": {"query_tokens", "Qformer", "vision_proj", "text_proj", "itm_head", "temp",
                "siglip_logit_scale", "siglip_bias", "shared_visual_projector", "ln_vision", "mhcac", "stream_adapters",
                "view_fusion", "mpc_heads"},
    "phase1c": {"query_tokens", "Qformer", "vision_proj", "text_proj", "itm_head", "temp",
                "siglip_logit_scale", "siglip_bias", "shared_visual_projector", "ln_vision", "mhcac", "stream_adapters",
                "view_fusion", "mpc_heads", "visual_encoder"},
}


@pytest.mark.parametrize("phase", ["phase1a", "phase1b", "phase1c"])
def test_each_phase_trains_exactly_its_modules(phase):
    model = FakeBlip2()
    apply_trainable(model, _spec(phase)[0])
    trainable = {n.split(".")[0] for n, p in model.named_parameters() if p.requires_grad}
    assert trainable == EXPECTED_TRAINABLE[phase]
    # Encoders are frozen in every phase; only BioViL's projection head opens, in 1c.
    assert not model.pubmedclip.weight.requires_grad
    assert not model.swin.weight.requires_grad
    assert not model.visual_encoder["encoder"].weight.requires_grad
    assert model.visual_encoder["projector"].weight.requires_grad is (phase == "phase1c")


@pytest.mark.parametrize("phase", ["phase1a", "phase1b", "phase1c"])
def test_frozen_parameters_get_no_gradient(phase):
    model = FakeBlip2()
    apply_trainable(model, _spec(phase)[0])
    model(torch.randn(2, 8)).backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, name
        else:
            assert param.grad is None, name


def test_phase1a_leaves_mhcac_untouched_by_an_optimizer_step():
    model = FakeBlip2()
    apply_trainable(model, _spec("phase1a")[0])
    before = model.mhcac.weight.detach().clone()
    q_before = model.query_tokens.detach().clone()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    model(torch.randn(2, 8)).backward()
    optimizer.step()
    assert torch.equal(model.mhcac.weight, before)
    assert not torch.equal(model.query_tokens, q_before)


def test_a_pinned_temperature_is_never_switched_back_on():
    model = FakeBlip2()
    apply_trainable(model, _spec("phase1a")[0], never_train=("temp",))
    assert not model.temp.requires_grad


def test_checkpoints_keep_what_any_phase_trains():
    spec = _spec("phase1b")[0]
    # The Q-Former is frozen at the end of 1b and must still be written.
    assert checkpoint_keep("Qformer.bert.weight", False, spec)
    assert checkpoint_keep("visual_encoder.projector.weight", False, spec)  # 1c trains it
    assert not checkpoint_keep("pubmedclip.weight", False, spec)
    assert not checkpoint_keep("visual_encoder.encoder.weight", False, spec)


# --------------------------------------------------------------------------
# The 1b hand-over
# --------------------------------------------------------------------------


def test_transition_roles():
    t = _spec("phase1b")[0].transition
    assert param_role("mhcac.attention_layers.0.w", t) == "fade_in"
    assert param_role("Qformer.bert.encoder.w", t) == "fade_out"
    assert param_role("shared_visual_projector.projections.swin.weight", t) == "fade_out"
    assert param_role("stream_adapters.biovil.w", t) == "steady"
    assert param_role("mhcac.w", None) == "steady"


def test_transition_is_linear_and_clamped():
    steps = 100
    for update in (0, 25, 50, 99, 100, 250):
        m = transition_multipliers(update, steps)
        expected = min(update / steps, 1.0)
        assert m["fade_in"] == pytest.approx(expected)
        assert m["fade_out"] == pytest.approx(1.0 - expected)
        assert m["steady"] == 1.0
        assert m["fade_in"] + m["fade_out"] == pytest.approx(1.0)


def test_transition_length_is_ten_percent_of_the_phase():
    t = _spec("phase1b")[0].transition
    updates_per_epoch = math.ceil(13922 / 4)
    assert t.steps(5 * updates_per_epoch) == round(0.1 * 5 * updates_per_epoch)


def test_set_lr_applies_the_phase_multiplier_and_nothing_else():
    from model.lavis.common.optims import _set_lr

    group_a = {"params": [nn.Parameter(torch.zeros(1))], "lr_scale": 0.5}
    group_b = {"params": [nn.Parameter(torch.zeros(1))], "lr_scale": 1.0, "phase_lr_mult": 0.25}
    optimizer = torch.optim.AdamW([group_a, group_b], lr=1.0)
    _set_lr(optimizer, 2e-4)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)  # unchanged rule
    assert optimizer.param_groups[1]["lr"] == pytest.approx(5e-5)


def test_malformed_blocks_fail_loudly():
    with pytest.raises(PhaseConfigError, match="unknown keys"):
        build_spec({"p": {"epochs": 1, "trainable": ["mhcac"], "lambdas": {}}}, "p")
    with pytest.raises(PhaseConfigError, match="trailing dot"):
        build_spec({"p": {"epochs": 1, "trainable": ["mhcac."]}}, "p")
    with pytest.raises(PhaseConfigError, match="not trainable"):
        build_spec(
            {"p": {"epochs": 1, "trainable": ["mhcac"],
                   "transition": {"fade_in": ["mhcac"], "fade_out": ["Qformer"]}}},
            "p",
        )


# --------------------------------------------------------------------------
# RunnerBase (host only: needs the training stack)
# --------------------------------------------------------------------------


def _runner_with(model, spec, updates_per_epoch=10):
    pytest.importorskip("torchinfo")
    pytest.importorskip("wandb")
    try:
        from model.lavis.runners.runner_base import RunnerBase
    except Exception as exc:  # GPU stack missing on the CPU box
        pytest.skip(f"RunnerBase not importable here ({type(exc).__name__})")
    class _Runner(RunnerBase):
        # The real property moves the model to a device; the stand-in has none.
        model = property(lambda self: self._wrapped_model)

    runner = _Runner.__new__(_Runner)
    runner.config = SimpleNamespace(
        run_cfg={
            "init_lr": 1e-4, "init_lr_q": 1e-4, "init_lr_cls": 2e-4, "weight_decay": 0.02,
            "accum_grad_iters": 1,
        }
    )
    runner.config.run_cfg = _AttrDict(runner.config.run_cfg)
    runner._model = model
    runner._wrapped_model = model
    runner._optimizer = None
    runner.phase = spec
    runner._phase_updates = 0
    runner._phase_transition_done = False
    runner._phase_gate_failed = False
    runner._updates_per_epoch = lambda: updates_per_epoch
    runner.unwrap_dist_model = lambda m: m
    return runner


class _AttrDict(dict):
    __getattr__ = dict.get


def test_runner_optimizer_holds_only_trainable_parameters_and_hands_over():
    model = FakeBlip2()
    spec = _spec("phase1b")[0]
    apply_trainable(model, spec)
    runner = _runner_with(model, spec, updates_per_epoch=100)
    ids = {id(p) for g in runner.optimizer.param_groups for p in g["params"]}
    assert ids == {id(p) for p in model.parameters() if p.requires_grad}
    roles = {g["phase_role"] for g in runner.optimizer.param_groups}
    assert roles == {"fade_in", "fade_out", "steady"}
    for g in runner.optimizer.param_groups:
        assert g["phase_lr_mult"] == {"fade_in": 0.0, "fade_out": 1.0, "steady": 1.0}[g["phase_role"]]

    steps = runner._phase_transition_steps()  # 10% of 5 x 100 = 50
    assert steps == 50
    runner._on_phase_update(0, 25)
    mult = {g["phase_role"]: g["phase_lr_mult"] for g in runner.optimizer.param_groups}
    assert mult["fade_in"] == pytest.approx(0.5) and mult["fade_out"] == pytest.approx(0.5)

    runner._on_phase_update(0, 50)
    assert runner._phase_transition_done
    remaining = {id(p) for g in runner.optimizer.param_groups for p in g["params"]}
    frozen = {id(p) for p in model.parameters() if not p.requires_grad}
    assert not remaining & frozen, "a frozen parameter is still in the optimizer"
    assert not model.Qformer["bert"].weight.requires_grad
    assert model.mhcac.weight.requires_grad
