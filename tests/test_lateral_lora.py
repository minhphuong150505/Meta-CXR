"""Pins Step 1 of the residual-lateral plan (docs/handoff/PLAN-2026-11-04-residual-lateral.md):
the minimal LoRA, the both-view cache's checks and the LoRA fit's pure helpers.

Fake data only: no MIMIC rows, no GPU.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from model.lavis.data.mimic_cxr_utils import build_study_index
from training.evaluation.schemas import build_sample_keys
from training.evaluation.view_keys import DEFAULT_PRIORITY, PL, build_view_map
from vision_encoders.lora import (
    SWIN_STAGE34_QV,
    LoRALinear,
    apply_lora,
    base_state_dict,
    lora_parameters,
    merge_lora,
    set_lora_train,
)

pytest.importorskip("scipy")

_ROOT = Path(__file__).resolve().parents[1]


def _module(name):
    spec = importlib.util.spec_from_file_location(name, _ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------------- LoRA
class _Self(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.query, self.key, self.value = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)

    def forward(self, x):
        return self.query(x) + self.key(x) * 0.5 + self.value(x)


def _swin_like(d=8, stages=4):
    """Module tree named like HF SwinModel: encoder.layers.<s>.blocks.<b>.attention.self.{query,key,value}."""
    torch.manual_seed(0)
    enc = nn.Module()
    enc.layers = nn.ModuleList()
    for _ in range(stages):
        stage = nn.Module()
        stage.blocks = nn.ModuleList()
        for _ in range(2):
            blk = nn.Module()
            blk.attention = nn.Module()
            blk.attention.self = _Self(d)
            stage.blocks.append(blk)
        enc.layers.append(stage)
    root = nn.Module()
    root.encoder = enc

    def forward(x):
        for stage in enc.layers:
            for blk in stage.blocks:
                x = torch.tanh(blk.attention.self(x))
        return x

    root.forward = forward
    return root


def test_lora_targets_only_stage34_query_value():
    m = _swin_like()
    names = apply_lora(m, SWIN_STAGE34_QV, r=2, alpha=4)
    assert len(names) == 2 * 2 * 2  # 2 stages x 2 blocks x {query, value}
    assert all(n.startswith(("encoder.layers.2.", "encoder.layers.3.")) for n in names)
    assert all(n.endswith((".query", ".value")) for n in names)
    assert isinstance(m.encoder.layers[3].blocks[1].attention.self.value, LoRALinear)
    assert isinstance(m.encoder.layers[3].blocks[1].attention.self.key, nn.Linear)
    assert isinstance(m.encoder.layers[1].blocks[0].attention.self.query, nn.Linear)


def test_lora_at_init_is_bit_identical_and_only_lora_trains():
    m = _swin_like()
    x = torch.randn(5, 3, 8)
    before = m.forward(x)
    apply_lora(m, SWIN_STAGE34_QV, r=2, alpha=4, dropout=0.1)
    m.eval()
    assert torch.equal(m.forward(x), before)
    trainable = [p for p in m.parameters() if p.requires_grad]
    assert {id(p) for p in trainable} == {id(p) for p in lora_parameters(m)}
    assert len(trainable) == 2 * 8


def test_lora_gradient_reaches_B_and_merge_keeps_output():
    m = _swin_like()
    apply_lora(m, SWIN_STAGE34_QV, r=2, alpha=4)
    x = torch.randn(4, 8)
    m.forward(x).sum().backward()
    lm = m.encoder.layers[2].blocks[0].attention.self.query
    assert lm.lora_B.grad is not None and lm.lora_B.grad.abs().sum() > 0
    with torch.no_grad():
        for p in lora_parameters(m):
            p.add_(torch.randn_like(p) * 0.1)
    out = m.forward(x)
    assert merge_lora(m) == 8
    assert torch.allclose(m.forward(x), out, atol=1e-5)


def test_base_state_dict_excludes_lora_and_detects_backbone_change():
    m = _swin_like()
    apply_lora(m, SWIN_STAGE34_QV, r=2, alpha=4)
    a = base_state_dict(m)
    assert not any(k.endswith(("lora_A", "lora_B")) for k in a)
    with torch.no_grad():
        for p in lora_parameters(m):
            p.add_(1.0)
    b = base_state_dict(m)
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_set_lora_train_touches_only_lora_wrappers():
    m = _swin_like()
    apply_lora(m, SWIN_STAGE34_QV, r=2, alpha=4, dropout=0.5)
    m.eval()
    set_lora_train(m, True)
    lm = m.encoder.layers[2].blocks[0].attention.self.query
    assert lm.training and lm.dropout.training
    assert not m.encoder.layers[2].blocks[0].attention.self.key.training


def test_apply_lora_refuses_a_pattern_that_matches_nothing():
    with pytest.raises(ValueError):
        apply_lora(_swin_like(), r"attention\.self\.qeury$", r=2, alpha=4)


# ----------------------------------------------------------------------------- cache checks
cache = _module("lateral_lora_cache")

ROWS = [
    ("p1", "s1", "PA"), ("p1", "s1", "LATERAL"),
    ("p2", "s2", "PA"),
    ("p3", "s3", "LL"), ("p3", "s3", "AP"),
    ("p4", "s4", "LATERAL"),
    ("p6", "s6", "LATERAL"), ("p6", "s6", "PA"),
]


def _rows():
    return [{"subject_id": a, "study_id": b, "ViewPosition": v} for a, b, v in ROWS]


def test_pl_positions_keeps_pl_studies_in_order():
    rows = _rows()
    vm = build_view_map(rows)
    studies = build_study_index(rows, anchor_priority=DEFAULT_PRIORITY, max_aux_views=1)
    keep = cache.pl_positions(vm, studies, np.arange(len(studies)))
    assert list(keep) == [i for i, g in enumerate(vm.groups) if g == PL] == [0, 2, 4]
    assert list(cache.pl_positions(vm, studies, np.array([1, 2, 3]))) == [2]


def test_pl_positions_refuses_a_different_aux_view():
    rows = _rows()
    vm = build_view_map(rows)
    studies = build_study_index(rows, anchor_priority=DEFAULT_PRIORITY, max_aux_views=1)
    studies[0] = {**studies[0], "aux": [5]}
    with pytest.raises(ValueError, match="not the PL lateral"):
        cache.pl_positions(vm, studies, np.arange(len(studies)))


def test_check_batch():
    keys = build_sample_keys(["0", "4"])
    good = {"image_id": torch.tensor([0, 4]), "aux_swin_image": torch.zeros(2, 1, 3, 4, 4),
            "aux_mask": torch.ones(2, 1, dtype=torch.bool), "aux_view_ids": torch.full((2, 1), 2)}
    cache.check_batch(good, keys, build_sample_keys)
    with pytest.raises(RuntimeError, match="keys"):
        cache.check_batch(good, keys[::-1], build_sample_keys)
    with pytest.raises(RuntimeError, match="without its lateral"):
        cache.check_batch({**good, "aux_mask": torch.tensor([[True], [False]])}, keys, build_sample_keys)
    with pytest.raises(RuntimeError, match="not a lateral"):
        cache.check_batch({**good, "aux_view_ids": torch.tensor([[2], [1]])}, keys, build_sample_keys)


# ----------------------------------------------------------------------------- fit helpers
fitmod = _module("lateral_lora_fit")
NAMES = ("Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum",
         "Fracture", "Lung Lesion", "Lung Opacity", "No Finding", "Pleural Effusion", "Pleural Other",
         "Pneumonia", "Pneumothorax", "Support Devices")


def test_should_stop_patience_two():
    assert not fitmod.should_stop([0.1, 0.2])
    assert not fitmod.should_stop([0.1, 0.2, 0.3])
    assert not fitmod.should_stop([0.1, 0.3, 0.2])
    assert fitmod.should_stop([0.1, 0.3, 0.2, 0.25])
    assert fitmod.should_stop([0.3, 0.3, 0.3])  # ties are not improvements


def test_shared_rows():
    ck = np.array(["a", "b", "c", "d"])
    a, b = fitmod.shared_rows(ck, np.array(["c", "x", "a"]), ref_is_pl_only=False)
    assert list(a) == [2, 0] and list(b) == [0, 2]
    with pytest.raises(SystemExit):
        fitmod.shared_rows(ck, np.array(["c", "x"]), ref_is_pl_only=True)


def _fake(n=300, seed=0):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 3, (n, 14))
    labels[rng.random((n, 14)) < 0.2] = -100
    return labels


def _probs(labels, signal, seed):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=labels.shape + (3,))
    onehot = np.eye(3)[np.clip(labels, 0, 2)]
    z = z + signal * onehot
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def test_compare_identical_sets_does_not_pass():
    y = _fake()
    base = [_probs(y, 1.0, s) for s in range(3)]
    res = fitmod.compare(base, [p.copy() for p in base], y, NAMES, boot_n=50)
    assert res["delta"] == 0.0 and not res["passed"]


def test_compare_better_set_passes():
    y = _fake()
    base = [_probs(y, 0.3, s) for s in range(3)]
    new = [_probs(y, 3.0, 10 + s) for s in range(3)]
    res = fitmod.compare(base, new, y, NAMES, boot_n=50)
    assert res["delta"] > 0.1 and res["lo"] > 0 and res["passed"]


def test_load_set_refuses_mismatched_seeds(tmp_path):
    y = _fake(20)
    keys = build_sample_keys([str(i) for i in range(20)])
    np.savez(tmp_path / "a.npz", keys=keys, probabilities=_probs(y, 1, 0), labels=y)
    np.savez(tmp_path / "b.npz", keys=keys[::-1], probabilities=_probs(y, 1, 1), labels=y)
    name, k, probs, lab = fitmod.load_set(f"X={tmp_path / 'a.npz'}")
    assert name == "X" and len(probs) == 1 and np.array_equal(lab, y)
    with pytest.raises(SystemExit):
        fitmod.load_set(f"X={tmp_path / 'a.npz'},{tmp_path / 'b.npz'}")


# ----------------------------------------------------------------------------- B1: folds, weight reload
def test_load_lora_state_dict_roundtrip_and_refusals():
    from vision_encoders.lora import load_lora_state_dict, lora_state_dict

    a, b = _swin_like(), _swin_like()
    apply_lora(a, SWIN_STAGE34_QV, r=2, alpha=4)
    apply_lora(b, SWIN_STAGE34_QV, r=2, alpha=4)
    with torch.no_grad():
        for p in lora_parameters(a):
            p.add_(torch.randn_like(p))
    x = torch.randn(3, 8)
    a.eval(), b.eval()
    assert load_lora_state_dict(b, lora_state_dict(a)) == 16
    assert torch.equal(a.forward(x), b.forward(x))
    sd = lora_state_dict(a)
    with pytest.raises(KeyError):
        load_lora_state_dict(b, {k: v for k, v in list(sd.items())[1:]})
    k0 = next(iter(sd))
    with pytest.raises(ValueError):
        load_lora_state_dict(b, {**sd, k0: torch.zeros(1, 1)})


def test_fold_rows_partition_is_deterministic_and_order_free():
    keys = np.array([f"k{i:05d}" for i in range(2000)])
    tr0, oof0 = fitmod.fold_rows(keys, 2, 0)
    tr1, oof1 = fitmod.fold_rows(keys, 2, 1)
    assert set(oof0).isdisjoint(oof1) and len(oof0) + len(oof1) == len(keys)
    assert np.array_equal(np.sort(tr0), np.sort(oof1)) and np.array_equal(np.sort(tr1), np.sort(oof0))
    assert 0.45 < len(oof0) / len(keys) < 0.55
    perm = np.random.default_rng(0).permutation(len(keys))
    f = fitmod.fold_of(keys, 2)
    assert np.array_equal(fitmod.fold_of(keys[perm], 2), f[perm])
    with pytest.raises(ValueError):
        fitmod.fold_rows(keys, 2, 2)
    with pytest.raises(ValueError):
        fitmod.fold_of(keys, 1)


def test_fit_holdout_rows_stays_inside_the_pool_and_default_is_unchanged():
    from swin_finetune_fit import holdout_split

    y = _fake(500)
    a, b = fitmod.fit_holdout_rows(y, None)
    ra, rb = holdout_split(y)
    assert np.array_equal(a, ra) and np.array_equal(b, rb)
    pool = np.arange(0, 500, 2)
    fit, hold = fitmod.fit_holdout_rows(y, pool)
    assert set(fit) | set(hold) <= set(pool) and set(fit).isdisjoint(hold)


def test_out_stem():
    assert fitmod.out_stem("L_lora", 16, 0, None) == "L_lora_s16"
    assert fitmod.out_stem("C_lora", 17, 2, 1) == "C_lora_s17_f1of2"


def test_fit_refuses_fold_without_n_folds(tmp_path):
    with pytest.raises(SystemExit, match="go together"):
        fitmod.main(["fit", "--arm", "L_lora", "--seed", "16", "--cache-dir", str(tmp_path),
                     "--ref-dir", str(tmp_path), "--out-dir", str(tmp_path / "o"), "--fold", "0"])


def test_check_weight_set():
    m = {"arm": "L_lora", "seed": 16, "fold": 0}
    assert fitmod.check_weight_set([m, {**m, "fold": 1}]) == ("lateral", True)
    with pytest.raises(SystemExit, match="mix"):
        fitmod.check_weight_set([m, {**m, "seed": 17, "fold": 1}])
    with pytest.raises(SystemExit, match="repeat"):
        fitmod.check_weight_set([m, m])
    with pytest.raises(SystemExit):
        fitmod.check_weight_set([])


def test_cache_refuses_test_without_flag(tmp_path):
    with pytest.raises(SystemExit, match="allow-test"):
        cache.main(["--split", "test", "--out-dir", str(tmp_path / "x")])
