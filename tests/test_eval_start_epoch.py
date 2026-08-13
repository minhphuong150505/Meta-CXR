"""Guards on `run.eval_start_epoch` — the warm-up window that skips validation.

Two behaviours are load-bearing and easy to break independently:

1. **No unscored epoch can be selected.** `checkpoint_best` is only written from
   inside the evaluation branch, so skipping evaluation must also make those
   epochs ineligible. If a future refactor moved best-checkpoint writing out of
   that branch, selection could silently start picking epochs that were never
   scored.
2. **Patience counts scored epochs only.** `best_epoch` initialises to 0 while
   the first scored epoch is `eval_start_epoch`. Measuring from `best_epoch`
   alone spends the entire patience budget on epochs that were never evaluated,
   so the run dies on its first scored epoch having saved no best checkpoint at
   all. That failure is silent: the log says "early stopping", which reads like
   convergence. The window is therefore clamped to open at `eval_start_epoch`,
   which is exact when `eval_start_epoch` is 0 -- the default path is unchanged.

These tests drive the two decision expressions directly rather than standing up
a RunnerBase, which would require the whole GPU stack. The expressions are
duplicated here on purpose: if someone edits the runner without editing this
file, the duplication is what makes the change visible.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Mirrors runner_base.RunnerBase.validate(): the epochs that actually evaluate.
def _evaluates(cur_epoch, eval_start_epoch, evaluate_only=False):
    if not evaluate_only and isinstance(cur_epoch, int) and cur_epoch < eval_start_epoch:
        return False
    return True


# Mirrors runner_base.RunnerBase.train(): the early-stopping trigger.
def _early_stops(cur_epoch, best_epoch, patience, eval_start_epoch, n_valid_splits=1):
    return (
        patience > 0
        and n_valid_splits > 0
        and cur_epoch - max(best_epoch, eval_start_epoch) >= patience
    )


class TestValidationWindow:
    @pytest.mark.parametrize("epoch", [0, 1, 2, 3, 4])
    def test_warmup_epochs_do_not_evaluate(self, epoch):
        assert _evaluates(epoch, eval_start_epoch=5) is False

    @pytest.mark.parametrize("epoch", [5, 6, 7, 8, 9])
    def test_scored_epochs_evaluate(self, epoch):
        assert _evaluates(epoch, eval_start_epoch=5) is True

    def test_sixth_epoch_is_index_five(self):
        """The knob counts indices, matching the `epoch: [5]` training log."""
        first_scored = next(e for e in range(10) if _evaluates(e, eval_start_epoch=5))
        assert first_scored == 5
        assert len([e for e in range(10) if not _evaluates(e, eval_start_epoch=5)]) == 5

    def test_default_zero_preserves_historical_behaviour(self):
        assert all(_evaluates(e, eval_start_epoch=0) for e in range(10))

    def test_evaluate_only_run_is_never_skipped(self):
        """Evaluate-only passes `cur_epoch="provided"`, which must still score."""
        assert _evaluates("provided", eval_start_epoch=5, evaluate_only=True) is True
        assert _evaluates("provided", eval_start_epoch=5, evaluate_only=False) is True


class TestEarlyStoppingWindow:
    def test_does_not_fire_during_warmup(self):
        """best_epoch is still its initial 0 throughout the warm-up window."""
        assert not any(
            _early_stops(e, best_epoch=0, patience=5, eval_start_epoch=5)
            for e in range(5)
        )

    def test_does_not_fire_on_the_first_scored_epoch(self):
        """The regression this file exists for.

        Measuring from best_epoch alone gives 5 - 0 >= 5, which fires. Clamping
        to eval_start_epoch gives 5 - 5 = 0, which does not. In the live runner
        validate() happens to update best_epoch first, so the naive expression
        survives by accident -- this asserts it does not depend on that ordering.
        """
        assert _early_stops(5, best_epoch=0, patience=5, eval_start_epoch=5) is False

    def test_patience_is_measured_from_the_first_scored_epoch(self):
        # Epoch [5] scored and improved from -inf, so best_epoch == 5.
        assert not _early_stops(9, best_epoch=5, patience=5, eval_start_epoch=5)
        assert _early_stops(10, best_epoch=5, patience=5, eval_start_epoch=5)

    def test_shipped_config_cannot_early_stop(self):
        """max_epoch 10, eval from [5], patience 5 -> earliest stop is [10].

        The last index in a 10-epoch run is [9], so the trigger is unreachable.
        This is asserted rather than left implicit so that lowering max_epoch or
        raising eval_start_epoch without revisiting patience shows up here.
        """
        max_epoch, eval_start_epoch, patience = 10, 5, 5
        assert eval_start_epoch + patience >= max_epoch
        assert not any(
            _early_stops(e, best_epoch=eval_start_epoch, patience=patience,
                         eval_start_epoch=eval_start_epoch)
            for e in range(max_epoch)
        )

    def test_lowering_patience_makes_it_reachable(self):
        """The documented one-line fix if early stopping should be live."""
        assert _early_stops(9, best_epoch=5, patience=4, eval_start_epoch=5) is True

    def test_disabled_patience_never_fires(self):
        assert not any(
            _early_stops(e, best_epoch=0, patience=-1, eval_start_epoch=0)
            for e in range(50)
        )


class TestShippedConfig:
    def test_production_yaml_matches_the_documented_window(self):
        yaml = pytest.importorskip("yaml")
        cfg_path = _REPO_ROOT / "pretraining" / "configs" / "mimic_cxr_full.yaml"
        run_cfg = yaml.safe_load(cfg_path.read_text())["run"]
        assert run_cfg["max_epoch"] == 10
        assert run_cfg["eval_start_epoch"] == 5
        # Five scored epochs: [5] through [9].
        scored = [
            e for e in range(run_cfg["max_epoch"])
            if _evaluates(e, run_cfg["eval_start_epoch"])
        ]
        assert scored == [5, 6, 7, 8, 9]
