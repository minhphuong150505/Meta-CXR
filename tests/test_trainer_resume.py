"""Resume must continue the original run, not start a similar one.

The check that matters is equivalence: train a toy model uninterrupted for N
steps, then train the same model for k steps, checkpoint, resume, and finish.
The two must land on bit-identical parameters. A resume that restores weights
but drops the optimizer moments, the scheduler position or the shuffle RNG
passes a naive "did it load?" test and still diverges here.

Runs on CPU in milliseconds -- no GPU, no MedGemma, no MIMIC data.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from training.trainer.checkpointing import CheckpointManager  # noqa: E402
from training.trainer.state import (  # noqa: E402
    RngSnapshot,
    TrainingState,
    build_provenance,
    git_sha,
)

STEPS, BATCH, FEATURES = 12, 4, 5


def seed_everything(seed: int = 16) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_toy():
    """A model, optimizer and scheduler whose trajectory depends on all three."""
    seed_everything()
    model = nn.Sequential(nn.Linear(FEATURES, 8), nn.ReLU(), nn.Linear(8, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-2, total_steps=STEPS
    )
    return model, optimizer, scheduler


def train_steps(model, optimizer, scheduler, generator, n: int) -> None:
    """``generator`` drives batch sampling, so shuffle order is part of the state."""
    loss_fn = nn.MSELoss()
    for _ in range(n):
        x = torch.randn(BATCH, FEATURES, generator=generator)
        y = torch.randn(BATCH, 1, generator=generator)
        optimizer.zero_grad()
        loss_fn(model(x), y).backward()
        optimizer.step()
        scheduler.step()


def parameters_of(model) -> list[torch.Tensor]:
    return [p.detach().clone() for p in model.parameters()]


def test_resumed_run_matches_an_uninterrupted_run(tmp_path):
    # --- reference: uninterrupted ---
    model_a, opt_a, sched_a = build_toy()
    gen_a = torch.Generator().manual_seed(99)
    train_steps(model_a, opt_a, sched_a, gen_a, STEPS)
    reference = parameters_of(model_a)
    reference_lr = sched_a.get_last_lr()

    # --- interrupted: train half, checkpoint, resume, finish ---
    model_b, opt_b, sched_b = build_toy()
    gen_b = torch.Generator().manual_seed(99)
    half = STEPS // 2
    train_steps(model_b, opt_b, sched_b, gen_b, half)

    manager = CheckpointManager(tmp_path)
    state = TrainingState(epoch=1, micro_step=half, global_step=half)
    manager.save(
        state,
        optimizer=opt_b,
        scheduler=sched_b,
        data_generator=gen_b,
        extra={"model": model_b.state_dict()},
    )

    # Fresh objects, as after a process restart.
    model_c, opt_c, sched_c = build_toy()
    gen_c = torch.Generator()
    payload = torch.load(manager.state_path(), weights_only=False)
    model_c.load_state_dict(payload["model"])
    restored = manager.load(
        optimizer=opt_c, scheduler=sched_c, data_generator=gen_c
    )

    assert restored.global_step == half
    train_steps(model_c, opt_c, sched_c, gen_c, STEPS - half)

    for expected, actual in zip(reference, parameters_of(model_c), strict=True):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    assert sched_c.get_last_lr() == reference_lr


def test_resume_without_the_data_generator_state_diverges(tmp_path):
    """Negative control: proves the equivalence test above is actually strict.

    If shuffle order were not restored, the test above would still pass when the
    generator state was silently dropped. It must not.
    """
    model_a, opt_a, sched_a = build_toy()
    gen_a = torch.Generator().manual_seed(99)
    train_steps(model_a, opt_a, sched_a, gen_a, STEPS)
    reference = parameters_of(model_a)

    model_b, opt_b, sched_b = build_toy()
    gen_b = torch.Generator().manual_seed(99)
    half = STEPS // 2
    train_steps(model_b, opt_b, sched_b, gen_b, half)
    # Resume with a generator reset to its initial seed instead of restored.
    train_steps(model_b, opt_b, sched_b, torch.Generator().manual_seed(99), STEPS - half)

    assert not all(
        torch.equal(expected, actual)
        for expected, actual in zip(reference, parameters_of(model_b), strict=True)
    )


def test_optimizer_moments_are_restored(tmp_path):
    model, optimizer, scheduler = build_toy()
    gen = torch.Generator().manual_seed(1)
    train_steps(model, optimizer, scheduler, gen, 3)

    manager = CheckpointManager(tmp_path)
    manager.save(TrainingState(global_step=3), optimizer=optimizer, scheduler=scheduler)

    _, fresh_opt, fresh_sched = build_toy()
    assert fresh_opt.state_dict()["state"] == {}  # no moments yet
    manager.load(optimizer=fresh_opt, scheduler=fresh_sched)
    assert fresh_opt.state_dict()["state"] != {}


def test_scheduler_position_is_restored(tmp_path):
    model, optimizer, scheduler = build_toy()
    gen = torch.Generator().manual_seed(1)
    train_steps(model, optimizer, scheduler, gen, 5)
    expected_lr = scheduler.get_last_lr()

    manager = CheckpointManager(tmp_path)
    manager.save(TrainingState(global_step=5), optimizer=optimizer, scheduler=scheduler)

    _, fresh_opt, fresh_sched = build_toy()
    manager.load(optimizer=fresh_opt, scheduler=fresh_sched)
    assert fresh_sched.get_last_lr() == expected_lr


# --------------------------------------------------------------------------
# RNG capture
# --------------------------------------------------------------------------


def test_all_four_rng_streams_round_trip():
    seed_everything(7)
    snapshot = RngSnapshot.capture()
    expected = (random.random(), float(np.random.rand()), float(torch.rand(1)))

    seed_everything(999)  # move every stream somewhere else
    snapshot.restore()
    assert (random.random(), float(np.random.rand()), float(torch.rand(1))) == expected


def test_data_generator_state_round_trips():
    generator = torch.Generator().manual_seed(3)
    torch.randn(4, generator=generator)
    snapshot = RngSnapshot.capture(generator)
    expected = torch.randn(4, generator=generator)

    other = torch.Generator().manual_seed(12345)
    snapshot.restore(other)
    torch.testing.assert_close(torch.randn(4, generator=other), expected)


def test_cuda_state_absent_on_cpu_does_not_break_restore():
    """A GPU-written checkpoint must still resume on a CPU box."""
    snapshot = RngSnapshot.capture()
    snapshot.cuda = [torch.zeros(16, dtype=torch.uint8)] * 8  # 8 fictional GPUs
    snapshot.restore()  # must not raise


# --------------------------------------------------------------------------
# state bookkeeping
# --------------------------------------------------------------------------


def test_lower_is_better_tracking():
    state = TrainingState(lower_is_better=True)
    assert state.record_score(1.0) is True
    assert state.record_score(0.5) is True
    assert state.record_score(0.7) is False
    assert state.bad_epochs == 1
    assert state.best_score == 0.5


def test_higher_is_better_tracking():
    state = TrainingState(best_score=float("-inf"), lower_is_better=False)
    assert state.record_score(0.3) is True
    assert state.record_score(0.9) is True
    assert state.record_score(0.4) is False
    assert state.best_score == 0.9


def test_early_stopping_triggers_at_patience():
    state = TrainingState()
    state.record_score(1.0)
    for _ in range(2):
        state.record_score(2.0)
    assert state.should_stop(patience=2) is True
    assert state.should_stop(patience=3) is False


def test_incompatible_state_version_is_rejected():
    payload = TrainingState(global_step=5).to_dict()
    payload["state_version"] = 999
    with pytest.raises(ValueError, match="cannot be read by this code"):
        TrainingState.from_dict(payload)


# --------------------------------------------------------------------------
# checkpoint file behaviour
# --------------------------------------------------------------------------


def test_save_is_atomic_and_leaves_no_temp_file(tmp_path):
    manager = CheckpointManager(tmp_path)
    manager.save(TrainingState(global_step=1))
    assert manager.state_path().is_file()
    assert list(tmp_path.glob("*.tmp")) == []


def test_missing_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="no trainer state"):
        CheckpointManager(tmp_path).load()


def test_resuming_an_optimizer_that_was_never_saved_raises(tmp_path):
    """Silently resetting the optimizer would change the trajectory unannounced."""
    manager = CheckpointManager(tmp_path)
    manager.save(TrainingState(global_step=1))  # no optimizer passed

    _, optimizer, _ = build_toy()
    with pytest.raises(ValueError, match="no optimizer state"):
        manager.load(optimizer=optimizer)


def test_verify_resumable_names_what_is_missing(tmp_path):
    manager = CheckpointManager(tmp_path)
    manager.save(TrainingState())
    with pytest.raises(FileNotFoundError, match="adapter_model.safetensors"):
        manager.verify_resumable(required_files=("adapter_model.safetensors",))


def test_is_resumable_reports_false_for_an_empty_dir(tmp_path):
    assert CheckpointManager(tmp_path).is_resumable() is False


def test_subdir_checkpoints_are_independent(tmp_path):
    manager = CheckpointManager(tmp_path)
    manager.save(TrainingState(global_step=3), subdir="last")
    manager.save(TrainingState(global_step=7), subdir="best")
    assert manager.load(subdir="last").global_step == 3
    assert manager.load(subdir="best").global_step == 7


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------


def test_provenance_round_trips_through_a_checkpoint(tmp_path):
    provenance = build_provenance(
        config_snapshot={"lora_rank": 16},
        dataset_fingerprint="abc123",
        model_revision="google/medgemma-1.5-4b-it@main",
        threshold_provenance="calibrated:val_2026_07_21",
    )
    manager = CheckpointManager(tmp_path)
    manager.save(TrainingState(provenance=provenance))

    restored = manager.load().provenance
    assert restored["dataset_fingerprint"] == "abc123"
    assert restored["threshold_provenance"] == "calibrated:val_2026_07_21"
    assert restored["config_snapshot"] == {"lora_rank": 16}


def test_threshold_provenance_defaults_to_argmax_not_silence():
    """An unrecorded threshold source is the ambiguity this field exists to remove."""
    assert build_provenance()["threshold_provenance"] == "image_only_argmax"


def test_git_sha_is_recorded_in_this_repo():
    sha = git_sha()
    assert sha != "unknown"
    assert len(sha) == 40
