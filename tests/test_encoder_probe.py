"""Pins the encoder-probe helpers (docs/handoff/PLAN-2026-10-31-encoder-probe.md)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]


def _module(name):
    spec = importlib.util.spec_from_file_location(name, _ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fit = _module("encoder_probe_fit")
feat = _module("encoder_probe_features")


def test_ap_auc_perfect_and_reversed():
    y = np.array([True, True, False, False])
    assert fit.ap_auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == (1.0, 1.0)
    ap, auc = fit.ap_auc(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert auc == 0.0 and ap == pytest.approx((1 / 3 + 2 / 4) / 2)


def test_ap_auc_ties_scored_at_end_of_run():
    y = np.array([True, False, True, False])
    ap, auc = fit.ap_auc(y, np.full(4, 0.5))
    assert ap == pytest.approx(0.5) and auc == pytest.approx(0.5)


def test_ap_matches_sklearn_without_ties():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    y = rng.random(500) < 0.3
    s = rng.random(500) + 0.5 * y
    ap, auc = fit.ap_auc(y, s)
    assert ap == pytest.approx(sk.average_precision_score(y, s))
    assert auc == pytest.approx(sk.roc_auc_score(y, s))


def test_single_class_finding_is_nan_and_skipped_by_macro():
    probs = np.full((4, 2, 3), 1 / 3)
    probs[:, 0, 1] = [0.9, 0.8, 0.1, 0.2]
    labels = np.array([[1, 0], [2, 0], [0, 0], [0, -100]])
    ap, _ = fit.per_finding(probs, labels)
    assert ap[0] == 1.0 and np.isnan(ap[1])
    assert fit.macro_ap(probs, labels) == 1.0


def test_presence_score_ignores_uncertain_mass():
    probs = np.array([[[0.2, 0.2, 0.6]]])
    assert fit.presence_scores(probs)[0, 0] == pytest.approx(0.5)


def test_paired_bootstrap_of_identical_arms_is_zero():
    rng = np.random.default_rng(1)
    probs = rng.dirichlet(np.ones(3), size=(60, 14))
    labels = rng.integers(0, 3, size=(60, 14))
    d, lo, hi = fit.paired_bootstrap(probs, probs, labels, n=20)
    assert d == 0.0 and lo == 0.0 and hi == 0.0


def test_native_preprocessing_shapes_and_normalisation():
    img = Image.fromarray(np.full((600, 500), 255, dtype=np.uint8), mode="L")
    r = feat.raddino_preprocess(img)
    assert r.shape == (3, 518, 518)
    assert r.max() == pytest.approx((1 - 0.5307) / 0.2583, rel=1e-5)
    m = feat.medsiglip_preprocess(img)
    assert m.shape == (3, 448, 448) and m.max() == pytest.approx(1.0)


def test_raddino_crop_keeps_the_centre():
    arr = np.zeros((1036, 518), dtype=np.uint8)  # portrait: shortest edge already 518
    arr[518, 259] = 255
    r = feat.raddino_preprocess(Image.fromarray(arr, mode="L"))
    y, x = np.unravel_index(r[0].argmax(), r[0].shape)
    assert abs(y - 259) <= 1 and abs(x - 259) <= 1


def test_grid_tokens_pools_row_major_patches():
    torch = pytest.importorskip("torch")
    tokens = torch.arange(16, dtype=torch.float32).view(1, 16, 1)  # 4x4, value = index
    out = feat.grid_tokens(tokens, 4, 2)
    assert out.shape == (1, 4, 1)
    assert out[0, :, 0].tolist() == [2.5, 4.5, 10.5, 12.5]


def test_attention_probe_shapes():
    torch = pytest.importorskip("torch")
    probe = fit.build_attention_probe([(49, 32), (64, 16)], [32, 24], d=8)
    out = probe([torch.zeros(2, 49, 32), torch.zeros(2, 64, 16)], [torch.zeros(2, 32), torch.zeros(2, 24)])
    assert out.shape == (2, 14, 3)
