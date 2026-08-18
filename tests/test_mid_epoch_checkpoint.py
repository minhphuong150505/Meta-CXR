"""Guards on mid-epoch checkpointing — `run.save_every_iters`.

Why this exists: an epoch on the production recipe is ~4 h, and
`checkpoint_last.pth` was only written when one finished. Three separate
crashes (an OOM, a DataLoader deadlock, and a whole-machine hang) all landed
inside epoch 0, so each one destroyed every hour of work that preceded it. The
supervisor could restart the run but had nothing to restart it *from*.

Three things are load-bearing and fail silently if broken:

1. **The save must happen on a sync step, after `zero_grad`.** Anywhere else
   captures the optimizer holding a half-accumulated window, whose gradients are
   thrown away on resume — so the checkpoint encodes an update that was never
   applied, at the wrong effective batch size.
2. **The write must be atomic.** These files are ~3.8 GB. `torch.save` straight
   onto the destination means a crash partway through the write leaves a
   truncated `checkpoint_last.pth` — destroying the very state the feature
   exists to protect, and doing so precisely when a crash is most likely.
3. **Resume must re-enter the same epoch, not the next one.** `_load_checkpoint`
   computes `start_epoch = epoch + 1`, which is right for an end-of-epoch
   checkpoint and skips a whole epoch of training for a mid-epoch one.

The cadence and resume expressions are mirrored here rather than driven through
a real RunnerBase, which would pull in the whole GPU stack. The duplication is
the point: editing the runner without editing this file makes the change
visible.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RUNNER = _REPO_ROOT / "model" / "lavis" / "runners" / "runner_base.py"
_TASK = _REPO_ROOT / "model" / "lavis" / "tasks" / "base_task.py"


def _sync_steps(iters_per_epoch: int, accum_grad_iters: int) -> list[int]:
    """Mirrors `_train_inner_loop`: the `i + 1` values handed to `on_sync_step`.

    Note the final window is often short on the full dataset, and it still ends
    on a sync step — a checkpoint hook that assumed fixed-size windows would
    miss it.
    """
    out = []
    for i in range(iters_per_epoch):
        window_start = (i // accum_grad_iters) * accum_grad_iters
        window_size = min(accum_grad_iters, iters_per_epoch - window_start)
        if i + 1 == window_start + window_size:
            out.append(i + 1)
    return out


def _saves_at(iters_per_epoch: int, accum_grad_iters: int, every: int) -> list[int]:
    if every <= 0:
        return []
    return [n for n in _sync_steps(iters_per_epoch, accum_grad_iters) if n % every == 0]


def _start_epoch(checkpoint: dict) -> int:
    """Mirrors `RunnerBase._load_checkpoint`."""
    if checkpoint.get("mid_epoch", False):
        return checkpoint["epoch"]
    return checkpoint["epoch"] + 1


class TestSaveCadence:
    def test_disabled_by_default(self):
        assert _saves_at(13922, 4, every=0) == []

    def test_fires_on_the_configured_interval(self):
        saves = _saves_at(13922, 4, every=1000)
        assert saves == [1000 * k for k in range(1, 14)]

    def test_every_save_is_a_sync_step(self):
        """A save between accumulation boundaries would checkpoint discarded grads."""
        syncs = set(_sync_steps(13922, 4))
        assert set(_saves_at(13922, 4, every=1000)) <= syncs

    def test_interval_not_divisible_by_accum_still_saves(self):
        """1000 % 4 == 0 today; a future accum value must not silently disable saving."""
        assert _saves_at(13922, 3, every=1000) != []

    @pytest.mark.parametrize("accum", [1, 2, 3, 4, 8, 11])
    def test_short_final_window_still_ends_on_a_sync_step(self, accum):
        iters = 13922
        assert _sync_steps(iters, accum)[-1] == iters
        assert len(_sync_steps(iters, accum)) == math.ceil(iters / accum)


class TestResumeSemantics:
    def test_mid_epoch_checkpoint_re_enters_the_same_epoch(self):
        assert _start_epoch({"epoch": 0, "mid_epoch": True, "iters_done": 7000}) == 0

    def test_end_of_epoch_checkpoint_advances(self):
        assert _start_epoch({"epoch": 0, "mid_epoch": False}) == 1

    def test_checkpoint_without_the_flag_behaves_as_before(self):
        """Checkpoints written before this feature carry no `mid_epoch` key."""
        assert _start_epoch({"epoch": 3}) == 4


class TestSourceInvariants:
    """These cannot be expressed as pure functions, so assert on the source."""

    def test_checkpoint_write_is_atomic(self):
        src = _RUNNER.read_text(encoding="utf-8")
        assert "os.replace(tmp_path, save_to)" in src, (
            "checkpoint_last is ~3.8 GB; a non-atomic torch.save onto the "
            "destination truncates it if the run dies mid-write"
        )
        assert "torch.save(save_obj, save_to)" not in src

    def test_hook_fires_after_zero_grad_on_the_sync_step(self):
        src = _TASK.read_text(encoding="utf-8")
        assert "on_sync_step(i + 1)" in src
        zero_grad = src.rindex("optimizer.zero_grad(set_to_none=True)")
        hook = src.index("on_sync_step(i + 1)")
        assert zero_grad < hook, (
            "the hook must run after grads are cleared, or the checkpoint "
            "captures a half-accumulated window"
        )

    def test_hook_failure_cannot_kill_the_run(self):
        src = _TASK.read_text(encoding="utf-8")
        hook = src.index("on_sync_step(i + 1)")
        assert "try:" in src[hook - 200:hook]
        assert "training continues" in src[hook:hook + 400]


class TestShippedConfig:
    def test_production_yaml_enables_it(self):
        yaml = pytest.importorskip("yaml")
        cfg = yaml.safe_load(
            (_REPO_ROOT / "pretraining" / "configs" / "mimic_cxr_full.yaml").read_text()
        )["run"]
        every = cfg["save_every_iters"]
        assert every > 0
        # Bound the worst-case loss: at the measured 1.0364 s/it a crash must
        # not be able to cost more than ~30 minutes.
        assert every * 1.0364 < 30 * 60
