"""Pins the residual-lateral tooling (docs/handoff/PLAN-2026-11-04-residual-lateral.md).

Fake data only: no MIMIC rows, no GPU.
"""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from model.lavis.data.mimic_cxr_utils import build_study_index
from training.evaluation.schemas import ClassificationPredictions, build_sample_keys
from training.evaluation.threshold_calibration import CutpointFile
from training.evaluation.view_keys import (
    F1,
    LATERAL_PRIORITY,
    OTHER,
    PL,
    ViewMap,
    align,
    build_view_map,
    check_num_views,
    lateral_pass_positions,
)

pytest.importorskip("scipy")

_ROOT = Path(__file__).resolve().parents[1]
NAMES = ("Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum",
         "Fracture", "Lung Lesion", "Lung Opacity", "No Finding", "Pleural Effusion", "Pleural Other",
         "Pneumonia", "Pneumothorax", "Support Devices")


def _module(name):
    spec = importlib.util.spec_from_file_location(name, _ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rs = _module("residual_stack")


@pytest.fixture(autouse=True)
def _coarse_cutpoint_grid(monkeypatch):
    monkeypatch.setattr(rs, "CUTPOINT_GRID", 11)  # the 121-point grid is O(grid^2) per finding


# Fake annotation rows: (subject, study, view). Studies in first-row order:
#   s1 PA + LATERAL -> PL       s2 PA alone -> F1        s3 AP + LL -> PL
#   s4 LATERAL alone -> other   s5 PA + AP + LATERAL -> other (aux is AP)
#   s6 LATERAL listed first, then PA -> PL (lateral row precedes the frontal row)
ROWS = [
    ("p1", "s1", "PA"), ("p1", "s1", "LATERAL"),
    ("p2", "s2", "PA"),
    ("p3", "s3", "LL"), ("p3", "s3", "AP"),
    ("p4", "s4", "LATERAL"),
    ("p5", "s5", "PA"), ("p5", "s5", "AP"), ("p5", "s5", "LATERAL"),
    ("p6", "s6", "LATERAL"), ("p6", "s6", "PA"),
]


def _rows():
    return [{"subject_id": a, "study_id": b, "ViewPosition": v} for a, b, v in ROWS]


def test_view_map_groups_and_keys():
    vm = build_view_map(_rows())
    assert list(vm.groups) == [PL, F1, PL, OTHER, OTHER, PL]
    assert vm.counts() == {PL: 3, F1: 1, OTHER: 2}
    assert list(vm.frontal_rows) == [0, 2, 4, 5, 6, 10]
    assert list(vm.lateral_rows) == [1, -1, 3, -1, -1, 9]
    # keys are the eval hook's digest of the anchor's annotation row
    assert vm.frontal_keys[0] == build_sample_keys(["0"])[0]
    assert vm.lateral_keys[2] == build_sample_keys(["3"])[0]
    assert vm.lateral_keys[1] == ""
    assert list(vm.num_views) == [2, 1, 2, 1, 2, 2]


def test_lateral_pass_positions_match_a_lateral_anchor_index():
    vm = build_view_map(_rows())
    lateral = build_study_index(_rows(), anchor_priority=LATERAL_PRIORITY, max_aux_views=0)
    assert list(lateral_pass_positions(vm, lateral)) == [0, 2, 5]
    assert list(lateral_pass_positions(vm, lateral, order=[1, 2, 3, 5])) == [2, 5]
    # a frontal-anchor index is refused: its anchors are not the laterals
    frontal = build_study_index(_rows(), max_aux_views=0)
    with pytest.raises(ValueError):
        lateral_pass_positions(vm, frontal)
    with pytest.raises(ValueError):
        lateral_pass_positions(vm, lateral[:-1])


def test_view_map_roundtrip_and_num_views_check(tmp_path):
    vm = build_view_map(_rows())
    back = ViewMap.load(vm.save(tmp_path / "vm.npz"))
    assert np.array_equal(back.frontal_keys, vm.frontal_keys) and np.array_equal(back.groups, vm.groups)
    check_num_views(back, vm.num_views, vm.frontal_keys)
    with pytest.raises(ValueError):
        check_num_views(back, np.ones(len(vm), int), vm.frontal_keys)


def test_align():
    keys = np.array(["a", "b", "c"])
    assert list(align(keys, np.array(["c", "a"]))) == [2, 0]
    with pytest.raises(ValueError):
        align(keys, np.array(["d"]))
    with pytest.raises(ValueError):
        align(np.array(["a", "a"]), np.array(["a"]))


def test_lateral_selection_in_feature_script(tmp_path):
    feat = _module("encoder_probe_features")
    ann = pd.DataFrame(_rows())
    ds = SimpleNamespace(annotation=ann,
                         studies=build_study_index(_rows(), anchor_priority=LATERAL_PRIORITY, max_aux_views=0))
    keep, fkeys, lkeys = feat.lateral_selection(ds, np.arange(6), tmp_path)
    assert list(keep) == [0, 2, 5]
    assert list(fkeys) == list(build_sample_keys(["0", "4", "10"]))
    assert list(lkeys) == list(build_sample_keys(["1", "3", "9"]))
    assert (tmp_path / "view_map.npz").is_file()


# ----------------------------------------------------------------------------- R1
def _dirichlet(rng, n, f=14):
    return rng.dirichlet(np.ones(3), size=(n, f))


def test_r1_zero_parameters_is_off_bit_for_bit():
    p = _dirichlet(np.random.default_rng(0), 50)
    e = _dirichlet(np.random.default_rng(1), 50)
    out = rs.apply_r1(p[:, 0], [e[:, 0]], np.zeros(1), np.zeros(3))
    assert np.array_equal(out, p[:, 0])


def test_crossfit_folds_deterministic_and_order_free():
    keys = np.array([f"k{i:03d}" for i in range(101)])
    f1 = rs.crossfit_folds(keys)
    assert np.array_equal(f1, rs.crossfit_folds(keys))
    perm = np.random.default_rng(3).permutation(101)
    assert np.array_equal(rs.crossfit_folds(keys[perm]), f1[perm])  # fold follows the key, not the row
    assert abs(int(f1.sum()) - 50) <= 1


def _synthetic(n=600, informative=True, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, size=(n, 14))
    y[rng.random((n, 14)) < 0.1] = -1
    p_off = _dirichlet(rng, n)
    if informative:
        logits = rng.normal(size=(n, 14, 3))
        yy = np.where(y < 0, 0, y)
        np.put_along_axis(logits, yy[..., None], np.take_along_axis(logits, yy[..., None], -1) + 2.0, -1)
        expert = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
    else:
        expert = _dirichlet(rng, n)
    return p_off, expert, y


def test_r1_learns_an_informative_expert_and_ignores_noise():
    p_off, expert, y = _synthetic(informative=True)
    m = y[:, 0] >= 0
    a, b = rs.fit_r1_finding(rs._logp(p_off[m, 0]), [rs._logp(expert[m, 0])], y[m, 0])
    assert a[0] > 0.5 and b[0] == 0.0
    p_off, noise, y = _synthetic(informative=False, seed=1)
    m = y[:, 0] >= 0
    a, _ = rs.fit_r1_finding(rs._logp(p_off[m, 0]), [rs._logp(noise[m, 0])], y[m, 0])
    assert a[0] < 0.1


def test_large_lambda_returns_to_off():
    p_off, expert, y = _synthetic()
    fold = rs.crossfit_folds(np.array([f"k{i:04d}" for i in range(len(y))]))
    out, params = rs.r1_crossfit(p_off, [expert], y, fold, lam=1e6)
    for A, B in params.values():
        assert np.abs(A).max() < 1e-4 and np.abs(B).max() < 1e-4
    assert np.allclose(out, p_off, atol=1e-4)


def test_permuted_expert_learns_nothing():
    p_off, expert, y = _synthetic(n=1500)
    keys = np.array([f"k{i:04d}" for i in range(len(y))])
    fold = rs.crossfit_folds(keys)
    perm = rs.permute_within_folds(expert, fold)
    _, params = rs.r1_crossfit(p_off, [perm], y, fold)
    assert np.mean([A.mean() for A, _ in params.values()]) < 0.05
    _, params = rs.r1_crossfit(p_off, [expert], y, fold)
    assert np.mean([A.mean() for A, _ in params.values()]) > 0.5


def test_holm_and_nb():
    adj = rs.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == pytest.approx(0.03) and adj["c"] == pytest.approx(0.06) and adj["b"] == pytest.approx(0.06)
    assert rs.nb([[0.1], [0.2, 0.4]]) == pytest.approx(2 * np.sqrt(np.var([0.2, 0.4], ddof=1)))


def test_run_end_to_end_with_an_informative_lateral():
    p_off, expert, y = _synthetic(n=400)
    c = _dirichlet(np.random.default_rng(9), 400)
    keys = np.array([f"k{i:04d}" for i in range(400)])
    noisy = [np.clip(expert + 0.01 * s, 1e-6, None) for s in range(3)]
    noisy = [x / x.sum(-1, keepdims=True) for x in noisy]
    res = rs.run(p_off, y, NAMES, keys, {"L": noisy, "C": [c, c, c]}, boot_n=30)
    assert res["diffs"]["D1"]["ap_pos_T"]["delta"] > 0
    assert res["diffs"]["D2"]["ap_pos_T"]["delta"] > 0
    assert set(res["diffs"]) == {"D1", "D2", "D3", "Dperm", "L_minus_C_alone"}
    assert res["verdict"].startswith(("QUA", "PARTIAL", "KHONG"))
    assert "_probs" in res and len(res["_probs"]["OFF+L"]) == 3


def test_main_keeps_non_pl_studies_bit_identical(tmp_path):
    vm = build_view_map(_rows())
    n = len(vm)
    rng = np.random.default_rng(5)
    y = rng.integers(0, 3, size=(n, 14))
    p = _dirichlet(rng, n)
    ClassificationPredictions(labels=y, probabilities=p, pathology_names=NAMES,
                              sample_keys=vm.frontal_keys).save(tmp_path / "off.npz")
    vm.save(tmp_path / "vm.npz")
    CutpointFile({nm: (0.3, 0.6) for nm in NAMES}, {}).save(tmp_path / "cut.json")
    pl = np.flatnonzero(vm.groups == PL)
    paths = []
    for s in range(2):
        e = _dirichlet(np.random.default_rng(10 + s), len(pl))
        lab = y[pl].copy()
        np.savez(tmp_path / f"L{s}.npz", keys=vm.frontal_keys[pl], probabilities=e, labels=lab)
        paths.append(str(tmp_path / f"L{s}.npz"))
    out = tmp_path / "res.json"
    rc = rs.main(["--off", str(tmp_path / "off.npz"), "--view-map", str(tmp_path / "vm.npz"),
                  "--off-cutpoints", str(tmp_path / "cut.json"), "--expert", "L=" + ",".join(paths),
                  "--expect-counts", "PL=3,F1=1,other=2", "--boot", "5", "--name", "fake", "--out", str(out)])
    assert rc == 0
    res = json.loads(out.read_text())
    assert res["group_counts"] == {"PL": 3, "F1": 1, "other": 2}
    assert "OFF+L" in res["whole_val"]
    # wrong expected counts abort
    with pytest.raises(SystemExit):
        rs.main(["--off", str(tmp_path / "off.npz"), "--view-map", str(tmp_path / "vm.npz"),
                 "--off-cutpoints", str(tmp_path / "cut.json"), "--expert", "L=" + paths[0],
                 "--name", "x", "--out", str(tmp_path / "x.json")])


# ----------------------------------------------------------------------------- expert fit helpers
def test_expert_fit_helpers():
    ef = _module("lateral_expert_fit")
    y = np.array([[0, -100], [1, 2], [-1, -1], [2, 0]] * 10)
    fit, hold = ef.holdout_split(y)
    labelled = np.flatnonzero((y >= 0).any(1))
    assert sorted(np.r_[fit, hold].tolist()) == labelled.tolist() and len(hold) == len(labelled) // 10
    assert (ef.normalise_missing(np.array([[-1, 2]])) == np.array([[-100, 2]])).all()
    ef.check_labels(y, y.copy(), "same")
    with pytest.raises(SystemExit):
        ef.check_labels(y, y[::-1], "differ")
