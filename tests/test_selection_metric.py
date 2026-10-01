"""Stage-1 checkpoint selection: macro_recall in 1b/1c, loss in 1a (2026-10-01).

The runner looks ``selection_metric`` up in the validation stats, which carry
every ``classification_metrics.AGGREGATE_METRICS`` entry once MHCAC runs. Phase
1a runs no classifier, so naming a classification metric there would leave the
runner with nothing to select on.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from training.evaluation.classification_metrics import AGGREGATE_METRICS  # noqa: E402


@pytest.fixture(scope="module")
def run_cfg():
    yaml = pytest.importorskip("yaml")
    cfg = yaml.safe_load((REPO / "pretraining/configs/mimic_cxr_full.yaml").read_text())
    return cfg["run"]


def test_classification_phases_select_on_macro_recall(run_cfg):
    assert "macro_recall" in AGGREGATE_METRICS
    assert run_cfg["selection_metric"] == "macro_recall"
    for phase in ("phase1b", "phase1c"):
        assert run_cfg["phases"][phase]["run"]["selection_metric"] == "macro_recall", phase


def test_phase1a_still_selects_on_loss(run_cfg):
    assert run_cfg["phases"]["phase1a"]["run"]["selection_metric"] == "loss"


def test_no_explicit_selection_mode_that_could_invert_it(run_cfg):
    # RunnerBase infers "max" for macro_recall; an explicit "min" would keep the worst epoch.
    assert "selection_mode" not in run_cfg
    for block in run_cfg["phases"].values():
        assert "selection_mode" not in block.get("run", {})
