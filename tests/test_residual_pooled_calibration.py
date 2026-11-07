"""CPU tests for scripts/residual_pooled_calibration.py (per-group Platt for pooled ranking metrics)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("residual_pooled_calibration",
                                               _ROOT / "scripts" / "residual_pooled_calibration.py")
pc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pc)


def _data(n=600, f=3, seed=0, shift=0.0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, (n, f))
    y[rng.random((n, f)) < 0.1] = -1
    z = rng.normal(size=(n, f, 3))
    yy = np.where(y < 0, 0, y)
    np.put_along_axis(z, yy[..., None], np.take_along_axis(z, yy[..., None], -1) + 1.5, -1)
    z[..., 1] += shift
    p = np.exp(z) / np.exp(z).sum(-1, keepdims=True)
    return p, y


def test_rank_swap_keeps_order_and_values():
    order = np.array([0.3, 0.1, 0.9, 0.5])
    vals = np.array([10.0, 40.0, 20.0, 30.0])
    out = pc.rank_swap(order, vals)
    assert sorted(out) == sorted(vals)
    assert np.array_equal(np.argsort(out), np.argsort(order))


def test_platt_is_monotone_and_identity_on_one_class():
    rng = np.random.default_rng(1)
    x = rng.random(300)
    y = rng.random(300) < x
    a, b = pc.platt_fit(x, y)
    assert a >= pc.A_MIN
    out = pc.platt_apply(x, a, b)
    assert np.array_equal(np.argsort(out, kind="stable"), np.argsort(x, kind="stable"))
    assert pc.platt_fit(x, np.zeros(300, bool)) == (1.0, 0.0)


def test_calibration_keeps_within_group_metrics_and_fixes_a_pure_scale_shift():
    # nonPL and PL with the same ranking quality, but PL's Positive scores are shifted up:
    # pooled AUROC suffers, per-group calibration (fit on train) restores it.
    p_a, y_a = _data(800, seed=2)
    p_b, y_b = _data(800, seed=3, shift=2.5)
    p = np.concatenate([p_a, p_b])
    y = np.concatenate([y_a, y_b])
    g = np.array(["nonPL"] * 800 + ["PL"] * 800)
    tp_a, ty_a = _data(3000, seed=4)
    tp_b, ty_b = _data(3000, seed=5, shift=2.5)
    tp = np.concatenate([tp_a, tp_b])
    ty = np.concatenate([ty_a, ty_b])
    tg = np.array(["nonPL"] * 3000 + ["PL"] * 3000)
    raw = pc.channel_scores(p)
    cal = pc.apply_group_calibration(raw, g, pc.fit_group_calibration(pc.channel_scores(tp), ty, tg))
    assert pc.within_group_max_change(raw, cal, y, g) <= 1e-12
    before = pc.summarise(pc.pooled_metrics(raw, y))["auroc_pos_mean"]
    after = pc.summarise(pc.pooled_metrics(cal, y))["auroc_pos_mean"]
    assert after > before + 0.01


def test_run_end_to_end_and_scale_arm_isolates_the_shift():
    p, y = _data(400, seed=6)
    tp, ty = _data(1500, seed=7)
    g = np.array(["PL"] * 200 + ["nonPL"] * 200)
    tg = np.array(["PL"] * 750 + ["nonPL"] * 750)

    def shifted(q, rows):
        z = np.log(q).copy()
        z[rows, :, 1] += 1.0  # pure scale shift on PL, ranking inside PL nearly kept
        out = np.exp(z) / np.exp(z).sum(-1, keepdims=True)
        out[~rows] = q[~rows]
        return out

    r1 = [shifted(p, g == "PL")] * 2
    r1_tr = [shifted(tp, tg == "PL")] * 2
    res = pc.run(p, r1, y, g, tp, r1_tr, ty, tg, boot_n=20)
    assert res["verdict"] in {"TANG", "HET GIAM", "VAN GIAM"}
    assert set(res["diffs"]) >= {"R1-OFF", "scale-OFF", "rank-OFF", "R1_cal-OFF"}
    assert res["within_group_max_change"] <= 1e-12
    bad = [r.copy() for r in r1]
    bad[0][np.flatnonzero(g == "nonPL")[0], 0] = [0.2, 0.3, 0.5]
    with pytest.raises(ValueError, match="outside PL"):
        pc.run(p, bad, y, g, tp, r1_tr, ty, tg, boot_n=5)


def test_cli_refuses_test_without_allow_test(tmp_path):
    with pytest.raises(SystemExit, match="allow-test"):
        pc.main(["--off", "x", "--view-map", "x", "--expert", "L=x", "--train-off", "x", "--train-expert", "L=x",
                 "--scored-split", "test", "--out", str(tmp_path / "o.json")])
    with pytest.raises(SystemExit, match="go together"):
        pc.main(["--off", "x", "--view-map", "x", "--expert", "L=x", "--train-off", "x", "--train-expert", "L=x",
                 "--r1-fit-off", "x", "--out", str(tmp_path / "o.json")])
