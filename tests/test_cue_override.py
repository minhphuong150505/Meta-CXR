"""CPU tests for training/dataio/cue_override.py (R1 cues of PL studies from each record's own logits)."""

import importlib.util
from hashlib import blake2b
from pathlib import Path

import numpy as np
import pytest

from training.dataio import cue_override as co

_ROOT = Path(__file__).resolve().parents[1]
FINDINGS = ["No Finding", "Cardiomegaly", "Edema"]   # record (MHCAC) order
NAMES = ["Edema", "Cardiomegaly", "No Finding"]      # npz order


def _key(stem):
    return blake2b(stem.encode(), digest_size=12).hexdigest()


def _groups_from(p_rec, cut_npz_order):
    """Cues the Stage-2 builder would give: cutpoints on p_pos/(p_pos+p_neg), MHCAC order, No Finding skipped."""
    out = {"positive": [], "negative": [], "uncertain": []}
    perm = [NAMES.index(n) for n in FINDINGS]
    for j, name in enumerate(FINDINGS):
        if name == "No Finding":
            continue
        p = p_rec[j]
        t1, t2 = cut_npz_order[perm[j]]
        s = p[1] / (p[1] + p[0])
        out["positive" if s >= t2 else "uncertain" if s >= t1 else "negative"].append(name)
    return out


def _override(pl_row, l_probs, A, B, off_cut, r1_cut, stems):
    return {"keys": np.array([_key(s) for s in stems]), "pl_row": np.array(pl_row), "names": NAMES,
            "l_probs": l_probs, "A": A, "B": B, "off_cut": off_cut, "r1_cut": r1_cut,
            "index": {_key(s): i for i, s in enumerate(stems)}}


def _record(stem, logits, off_cut):
    p = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
    return {"image_path": f"/x/{stem}.jpg", "class_logits": logits, "pred_groups": _groups_from(p, off_cut)}


def test_zero_r1_keeps_cues_and_non_pl_is_untouched():
    rng = np.random.default_rng(0)
    off_cut = np.array([[0.3, 0.6]] * 3)
    recs = [_record(f"s{i}", rng.normal(size=(3, 3)), off_cut) for i in range(4)]
    before = [dict(r["pred_groups"]) for r in recs]
    ov = _override([0, -1, 1, -1], rng.dirichlet(np.ones(3), (1, 2, 3)), np.zeros((1, 3)), np.zeros((1, 3, 3)),
                   off_cut, off_cut, [f"s{i}" for i in range(4)])
    counts = co.apply_r1_cues(recs, ov, FINDINGS)
    assert [r["pred_groups"] for r in recs] == before
    assert counts == {"pl_records": 2, "changed_records": 0, "changed_cells": 0, "off_agreement": 1.0}


def test_r1_matches_residual_stack_and_changes_pl_cues():
    spec = importlib.util.spec_from_file_location("residual_stack", _ROOT / "scripts" / "residual_stack.py")
    rs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rs)
    rng = np.random.default_rng(1)
    p = rng.dirichlet(np.ones(3), 3)
    lp = rng.dirichlet(np.ones(3), (2, 3))
    A, B = np.array([[1.5, 0.2, 0.7], [0.3, 0.9, 0.0]]), rng.normal(size=(2, 3, 3)) * [0, 1, 1]
    got = co.r1_probs(p, lp, A, B)
    want = np.mean([[rs.apply_r1(p[j][None], [lp[s, j][None]], A[s, j:j + 1], B[s, j])[0] for j in range(3)]
                    for s in range(2)], 0)
    assert np.allclose(got, want)
    off_cut = np.array([[0.3, 0.6]] * 3)
    rec = _record("a", rng.normal(size=(3, 3)), off_cut)
    strong = np.zeros((1, 1, 3, 3)) + [0.01, 0.98, 0.01]  # the expert says Positive everywhere
    ov = _override([0], strong, np.full((1, 3), 3.0), np.zeros((1, 3, 3)), off_cut, off_cut, ["a"])
    co.apply_r1_cues([rec], ov, FINDINGS)
    assert rec["pred_groups"]["positive"] == ["Cardiomegaly", "Edema"]


def test_refuses_mismatch_and_missing_keys():
    off_cut = np.array([[0.3, 0.6]] * 3)
    rec = _record("a", np.array([[3.0, 0, 0], [0, 3.0, 0], [0, 0, 3.0]]), off_cut)
    rec["pred_groups"] = {"positive": [], "negative": [], "uncertain": ["Cardiomegaly", "Edema"]}
    ov = _override([0], np.full((1, 1, 3, 3), 1 / 3), np.zeros((1, 3)), np.zeros((1, 3, 3)), off_cut, off_cut, ["a"])
    with pytest.raises(SystemExit, match="agree with OFF"):
        co.apply_r1_cues([rec], ov, FINDINGS)
    with pytest.raises(SystemExit, match="not in the override"):
        co.apply_r1_cues([_record("zz", np.zeros((3, 3)), off_cut)], ov, FINDINGS)
