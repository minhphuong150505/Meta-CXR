"""Pins the Swin fine-tune probe helpers (docs/handoff/PLAN-2026-10-31-swin-finetune-probe.md)."""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from vision_encoders.swin.medclip_swin import medclip_preprocess, pad_to_square  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]


def _module(name):
    spec = importlib.util.spec_from_file_location(name, _ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cache = _module("swin_finetune_cache")
fit = _module("swin_finetune_fit")


def _radiograph(h=300, w=260, seed=0):
    return Image.fromarray(np.random.default_rng(seed).integers(0, 256, (h, w), dtype=np.uint8), mode="L")


def test_uint8_cache_is_medclip_input_before_normalisation():
    img = _radiograph()
    expected = np.asarray(pad_to_square(img).resize((224, 224), Image.BICUBIC))
    got = cache.swin_image_to_uint8(medclip_preprocess(img).numpy())
    assert got.dtype == np.uint8 and np.array_equal(got, expected)


def test_cache_then_normalise_reproduces_medclip_preprocess():
    img = _radiograph(seed=1)
    u8 = torch.from_numpy(cache.swin_image_to_uint8(medclip_preprocess(img).numpy()))[None]
    assert torch.allclose(fit.to_swin_input(u8)[0], medclip_preprocess(img), atol=1e-6)


def _fake_swin_names():
    names = ["model.embeddings.patch_embeddings.projection.weight", "model.embeddings.norm.bias"]
    for i in range(4):
        names += [f"model.encoder.layers.{i}.blocks.0.attention.self.query.weight",
                  f"model.encoder.layers.{i}.blocks.0.layernorm_before.bias"]
        if i < 3:
            names.append(f"model.encoder.layers.{i}.downsample.reduction.weight")
    return names + ["model.layernorm.weight"]


def test_layerwise_lr_groups_cover_every_parameter_once():
    names = _fake_swin_names()
    params = [(n, torch.nn.Parameter(torch.zeros(4) if n.endswith(("bias", "layernorm.weight")) else torch.zeros(4, 4)))
              for n in names]
    groups = fit.swin_param_groups(params)
    seen = [n for g in groups for n in g["names"]]
    assert sorted(seen) == sorted(names) and len(seen) == len(set(seen))
    lr = {n: g["lr"] for g in groups for n in g["names"]}
    wd = {n: g["weight_decay"] for g in groups for n in g["names"]}
    assert lr["model.layernorm.weight"] == pytest.approx(5e-5)
    assert lr["model.encoder.layers.3.blocks.0.attention.self.query.weight"] == pytest.approx(5e-5)
    assert lr["model.encoder.layers.0.downsample.reduction.weight"] == pytest.approx(5e-5 * 0.7 ** 3)
    assert lr["model.embeddings.patch_embeddings.projection.weight"] == pytest.approx(5e-5 * 0.7 ** 4)
    assert wd["model.embeddings.norm.bias"] == 0.0 and wd["model.layernorm.weight"] == 0.0
    assert wd["model.encoder.layers.2.blocks.0.attention.self.query.weight"] == 0.05


def test_unknown_swin_parameter_is_refused():
    with pytest.raises(ValueError):
        fit.swin_param_groups([("model.pooler.weight", torch.nn.Parameter(torch.zeros(2, 2)))])


def test_identity_affine_leaves_the_image_and_no_matrix_is_a_flip():
    z = torch.zeros(1)
    theta = fit.affine_theta(z, torch.ones(1), z, z)
    assert torch.allclose(theta, torch.tensor([[[1.0, 0, 0], [0, 1.0, 0]]]))
    x = torch.rand(1, 1, 16, 16)
    grid = torch.nn.functional.affine_grid(theta, [1, 1, 16, 16], align_corners=False)
    assert torch.allclose(torch.nn.functional.grid_sample(x, grid, align_corners=False), x, atol=1e-5)
    g = torch.Generator().manual_seed(0)
    a, s = torch.rand(500, generator=g) * 20 - 10, torch.rand(500, generator=g) * 0.2 + 0.9
    th = fit.affine_theta(a, s, torch.zeros(500), torch.zeros(500))
    assert (torch.linalg.det(th[:, :, :2]) > 0).all()


def test_augmentation_stays_in_range_and_only_with_a_generator():
    u8 = torch.randint(0, 256, (4, 224, 224), dtype=torch.uint8)
    assert torch.equal(fit.to_swin_input(u8), fit.to_swin_input(u8))
    aug = fit.augment(u8.float().unsqueeze(1) / 255, torch.Generator().manual_seed(0))
    assert aug.shape == (4, 1, 224, 224) and aug.min() >= 0 and aug.max() <= 1
    assert not torch.allclose(fit.to_swin_input(u8, torch.Generator().manual_seed(0)), fit.to_swin_input(u8))


def test_holdout_split_matches_the_encoder_probe():
    labels = np.random.default_rng(3).integers(-100, 3, (1000, 14))
    labels[:50] = -100
    f, h = fit.holdout_split(labels)
    labelled = np.flatnonzero((labels >= 0).any(1))
    perm = np.random.default_rng(16).permutation(labelled)
    assert np.array_equal(h, np.sort(perm[: len(perm) // 10])) and np.array_equal(f, np.sort(perm[len(perm) // 10:]))


def test_warmup_then_cosine():
    assert fit.warmup_cosine(0, 100, 10) == pytest.approx(0.1)
    assert fit.warmup_cosine(10, 100, 10) == pytest.approx(1.0)
    assert fit.warmup_cosine(55, 100, 10) == pytest.approx(0.5 * (1 + math.cos(math.pi * 0.5)))


def test_reading_rule():
    assert fit.verdict(0.025, 0.001) == "GO_ON"
    assert fit.verdict(0.025, -0.001) == "NOT_ENOUGH"
    assert fit.verdict(0.01, 0.002) == "NOT_ENOUGH"
    assert fit.verdict(0.0, -0.01) == "STOP" and fit.verdict(-0.01, -0.02) == "STOP"
