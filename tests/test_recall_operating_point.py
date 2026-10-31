"""Pins the recall-leaning cutpoint objective (docs/handoff/PLAN-2026-11-01-recall-operating-point.md)."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from training.evaluation.threshold_calibration import (
    decide_with_cutpoints,
    fit_severity_cutpoints,
    present_fbeta,
    severity_scores,
)

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("ops", _ROOT / "scripts" / "compare_stage1_operating_points.py")
ops = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ops)

NAMES = ("Atelectasis", "Cardiomegaly")


def _pred(n=400, seed=0):
    rng = np.random.default_rng(seed)
    labels = rng.choice([0, 1, 2], size=(n, 2), p=[0.6, 0.3, 0.1])
    pos = np.clip(0.25 + 0.35 * (labels == 1) + 0.15 * (labels == 2) + rng.normal(0, 0.2, labels.shape), 0.01, 0.98)
    unc = np.full(labels.shape, 0.01)
    probs = np.stack([1 - pos - unc, pos, unc], -1)
    labels[:5, 0] = -1  # missing cells are skipped
    return SimpleNamespace(probabilities=probs, labels=labels, pathology_names=NAMES)


def test_present_fbeta_by_hand():
    present = np.array([1, 1, 1, 0, 0], bool)
    called = np.array([1, 1, 0, 1, 0], bool)  # tp 2, fn 1, fp 1
    assert present_fbeta(present, called, 1.0) == pytest.approx(2 * 2 / (2 * 2 + 1 + 1))
    assert present_fbeta(present, called, 2.0) == pytest.approx(5 * 2 / (5 * 2 + 4 * 1 + 1))
    assert present_fbeta(np.zeros(3, bool), np.zeros(3, bool), 2.0) == 0.0


def test_larger_beta_never_raises_t1_or_lowers_present_recall():
    pred = _pred()
    s = severity_scores(pred.probabilities)
    prev_t1 = {n: np.inf for n in NAMES}
    for beta in (1.0, 1.5, 2.0, 3.0, 4.0):
        cut = fit_severity_cutpoints(pred, objective="present_fbeta", beta=beta)
        for j, n in enumerate(NAMES):
            t1, t2 = cut[n]
            assert t1 <= t2 and t1 <= prev_t1[n] + 1e-12
            prev_t1[n] = t1
            m = pred.labels[:, j] >= 0
            d = decide_with_cutpoints(s[m, j], t1, t2)
            assert set(np.unique(d)) <= {0, 1, 2}


def test_t2_split_does_not_change_present_calls():
    pred = _pred(seed=1)
    cut = fit_severity_cutpoints(pred, objective="present_fbeta", beta=2.0)
    s = severity_scores(pred.probabilities)[:, 1]
    t1, t2 = cut["Cardiomegaly"]
    assert np.array_equal(decide_with_cutpoints(s, t1, t2) != 0, s >= t1)


def test_default_objective_unchanged():
    pred = _pred(seed=2)
    assert fit_severity_cutpoints(pred) == fit_severity_cutpoints(pred, objective="weighted_f1", beta=9.0)


def test_crossfit_never_scores_where_it_fitted(monkeypatch):
    pred = _pred(n=60, seed=3)
    pred.probabilities = pred.probabilities.copy()
    pred.probabilities[:, 0, 2] = np.arange(60) * 1e-6  # row identity in an unused cell
    seen = []

    def spy(rule, fit_on, score_on):
        seen.append((set(np.rint(fit_on.probabilities[:, 0, 2] * 1e6).astype(int)),
                     set(np.rint(score_on.probabilities[:, 0, 2] * 1e6).astype(int))))
        return np.zeros(score_on.labels.shape, int)

    monkeypatch.setattr(ops, "rule_decisions", spy)
    ops.crossfit(pred, ["headline"], repeats=3)
    assert len(seen) == 6 and all(not (a & b) and len(a | b) == 60 for a, b in seen)


def test_summarize_counts_present_recall_and_precision():
    labels = np.array([[1, 0], [2, 0], [0, 1], [0, -1]])
    decisions = np.array([[1, 0], [0, 1], [2, 1], [0, 1]])
    m = ops.summarize(decisions, labels, NAMES)
    # present cells: (0,0) (1,0) (2,1) -> called: (0,0) (2,1); calls on valid cells: (0,0) (1,1) (2,0) (2,1)
    assert m["present_recall"] == pytest.approx(2 / 3) and m["present_precision"] == pytest.approx(2 / 4)
