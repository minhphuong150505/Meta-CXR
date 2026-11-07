"""CPU tests for scripts/r1_stage2_cues.py (OFF + R1 cues of PL studies for Stage 2)."""

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


def test_pl_cues_change_only_pl_rows():
    rng = np.random.default_rng(0)
    n = 300
    p = rng.dirichlet(np.ones(3), size=(n, 3))
    y = rng.integers(0, 3, (n, 3))
    pl = np.arange(0, n, 2)
    r1 = rng.dirichlet(np.ones(3), size=(len(pl), 3))
    off_cut = {name: (0.3, 0.6) for name in NAMES}
    off_codes, new_codes, cut = rc.pl_cues(p, y, pl, r1, NAMES, off_cut, grid=11)
    outside = np.setdiff1d(np.arange(n), pl)
    assert np.array_equal(new_codes[outside], off_codes[outside])
    assert set(cut) == set(NAMES) and set(np.unique(off_codes)) <= {0, 1, 2}
