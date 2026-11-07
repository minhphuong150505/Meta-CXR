"""CPU tests for scripts/r1_stage2_cues.py (R1 cue-override npz for Stage 2)."""

import importlib.util
from hashlib import blake2b
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")
_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("r1_stage2_cues", _ROOT / "scripts" / "r1_stage2_cues.py")
rc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rc)

NAMES = ("Atelectasis", "Cardiomegaly", "Edema")


def test_gen_key_matches_generation_sample_key():
    gen = pytest.importorskip("scripts.generate_stage2_reports")
    assert rc.gen_key("SYNTHETIC-x") == blake2b(b"SYNTHETIC-x", digest_size=12).hexdigest()
    assert rc.gen_key("SYNTHETIC-x") == gen.sample_key_for("/a/b/SYNTHETIC-x.jpg")


def test_npz_roundtrip_through_cue_override(tmp_path, monkeypatch):
    from training.dataio import cue_override as co

    rng = np.random.default_rng(0)
    n = 200
    p = rng.dirichlet(np.ones(3), (n, 3))
    y = rng.integers(0, 3, (n, 3))
    pl = np.arange(0, n, 2)
    l_pl = [rng.dirichlet(np.ones(3), (len(pl), 3)) for _ in range(2)]
    params = [(np.full((3, 1), 0.5), np.zeros((3, 3))), (np.full((3, 1), 1.0), np.zeros((3, 3)))]
    r1_cut, probs = rc.fit_r1_cutpoints(p[pl], y[pl], l_pl, params, NAMES, grid=11)
    keys = np.array([rc.gen_key(f"d{i}") for i in range(n)])
    off_cut = {k: (0.3, 0.6) for k in NAMES}
    rc.write_npz(tmp_path / "o.npz", keys, pl, NAMES, l_pl, params, off_cut, r1_cut)
    ov = co.load_override(tmp_path / "o.npz")
    assert ov["l_probs"].shape == (2, len(pl), 3, 3) and (ov["pl_row"] >= 0).sum() == len(pl)
    j = ov["pl_row"][pl[3]]
    assert np.allclose(co.r1_probs(p[pl[3]], ov["l_probs"][:, j], ov["A"], ov["B"]), probs[3], atol=1e-5)
