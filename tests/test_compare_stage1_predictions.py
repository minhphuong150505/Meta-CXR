"""CPU tests for scripts/compare_stage1_predictions.py (multi-view ablation)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from training.evaluation.schemas import ClassificationPredictions

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "compare_stage1_predictions", REPO_ROOT / "scripts/compare_stage1_predictions.py"
)
compare = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compare)

NAMES = ("Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion")


def _preds(rng, labels, num_views, noise):
    n, p = labels.shape
    logits = rng.normal(size=(n, p, 3)) * noise
    logits[np.arange(n)[:, None], np.arange(p)[None, :], labels] += 2.0
    probs = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
    return ClassificationPredictions(
        labels=labels, probabilities=probs, pathology_names=NAMES,
        sample_keys=np.array([f"k{i}" for i in range(n)]), num_views=num_views,
    )


def _write(tmp_path, name, preds):
    path = tmp_path / name
    preds.save(path)
    return path


def test_identical_files_give_zero_delta_and_view_groups(tmp_path):
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 3, size=(60, len(NAMES)))
    views = np.array([1] * 30 + [2] * 20 + [3] * 10)
    preds = _preds(rng, labels, views, noise=1.0)
    a = _write(tmp_path, "a.npz", preds)
    out = tmp_path / "out.json"
    assert compare.main(["--baseline", str(a), "--candidate", str(a),
                         "--output", str(out), "--resamples", "20"]) == 0
    res = json.loads(out.read_text())
    assert {g: res["groups"][g]["n"] for g in res["groups"]} == {
        "all": 60, "views_1": 30, "views_2": 20, "views_3plus": 10, "views_2plus": 30,
    }
    for group in res["groups"].values():
        assert group["changed_argmax_fraction"] == 0.0
        for metric in group["metrics"].values():
            assert metric["delta"] == 0.0


def test_misaligned_studies_are_refused(tmp_path):
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 3, size=(10, len(NAMES)))
    a = _preds(rng, labels, np.ones(10, int), 1.0)
    b = _preds(rng, labels, np.ones(10, int), 1.0)
    b.sample_keys = b.sample_keys[::-1]
    pa, pb = _write(tmp_path, "a.npz", a), _write(tmp_path, "b.npz", b)
    assert compare.main(["--baseline", str(pa), "--candidate", str(pb),
                         "--output", str(tmp_path / "o.json")]) == 2


def test_missing_num_views_is_refused(tmp_path):
    rng = np.random.default_rng(2)
    labels = rng.integers(0, 3, size=(10, len(NAMES)))
    a = _write(tmp_path, "a.npz", _preds(rng, labels, None, 1.0))
    assert compare.main(["--baseline", str(a), "--candidate", str(a),
                         "--output", str(tmp_path / "o.json")]) == 2
