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


def test_test_only_copies_the_recipe_and_takes_test_experts(tmp_path, monkeypatch):
    """``--test-only`` (2026-11-27): A/B/cutpoints bit for bit from the recipe, l_probs from the test experts."""
    rng = np.random.default_rng(1)
    n, pl = 40, np.arange(0, 40, 4)
    params = [(rng.normal(size=(3, 1)), rng.normal(size=(3, 3))) for _ in range(2)]
    keys_va = np.array([rc.gen_key(f"v{i}") for i in range(n)])
    l_va = [rng.dirichlet(np.ones(3), (len(pl), 3)) for _ in range(2)]
    rc.write_npz(tmp_path / "r_val.npz", keys_va, pl, NAMES, l_va, params, {k: (0.3, 0.6) for k in NAMES},
                 {k: (0.2, 0.7) for k in NAMES})
    l_te = [rng.dirichlet(np.ones(3), (len(pl), 3)) for _ in range(2)]
    keys_te = np.array([rc.gen_key(f"t{i}") for i in range(n)])

    class _Off:
        pathology_names = NAMES

    monkeypatch.setattr(rc.pc, "load_split", lambda *a: (None, None, pl, None, l_te))
    monkeypatch.setattr(rc, "split_keys", lambda split, config, expected: (keys_te, None))
    import training.evaluation.schemas as sch
    import training.evaluation.view_keys as vk
    monkeypatch.setattr(sch.ClassificationPredictions, "load", staticmethod(lambda p: _Off()))
    monkeypatch.setattr(vk.ViewMap, "load", staticmethod(lambda p: type("V", (), {"frontal_keys": None})()))
    argv = ["--off", "x", "--view-map", "y", "--expert", "L=a,b", "--test-only", "--recipe",
            str(tmp_path / "r_val.npz"), "--out-prefix", str(tmp_path / "r")]
    assert rc.main(argv) == 0
    a, b = np.load(tmp_path / "r_val.npz"), np.load(tmp_path / "r_test.npz")
    for k in ("A", "B", "off_cut", "r1_cut", "names"):
        assert np.array_equal(a[k], b[k])
    assert np.array_equal(b["keys"], keys_te) and (b["pl_row"] >= 0).sum() == len(pl)
    assert np.allclose(b["l_probs"], np.stack(l_te))
    with pytest.raises(SystemExit):  # never overwrites
        rc.main(argv)
    with pytest.raises(SystemExit):  # --test-only needs --recipe
        rc.main(["--off", "x", "--view-map", "y", "--expert", "L=a", "--test-only", "--out-prefix", str(tmp_path / "q")])
