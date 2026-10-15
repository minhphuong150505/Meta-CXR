"""scripts/merge_qformer_refit.py: the refit replaces the Q-Former side and nothing else."""

from __future__ import annotations

import pytest
import torch

from scripts.merge_qformer_refit import MergeError, merge_states, refit_prefixes

PREFIXES = ("query_tokens", "Qformer", "qformer_input_norm")


def _base():
    torch.manual_seed(0)
    return {
        "query_tokens": torch.randn(1, 4, 8),
        "Qformer.bert.w": torch.randn(8, 8),
        "mhcac.w": torch.randn(8, 8),
        "visual_encoder.encoder.layer4.w": torch.randn(4),   # 1c-unfrozen, absent from the refit
        "visual_encoder.bn.running_mean": torch.randn(4),
    }


def _refit(base):
    return {
        "query_tokens": torch.randn(1, 4, 8),
        "Qformer.bert.w": torch.randn(8, 8),
        "qformer_input_norm.norms.biovil.weight": torch.ones(8),
        "mhcac.w": base["mhcac.w"].clone(),
        "visual_encoder.bn.running_mean": base["visual_encoder.bn.running_mean"].clone(),
    }


def test_merge_takes_the_qformer_side_from_the_refit_and_the_rest_from_the_base():
    base = _base()
    refit = _refit(base)
    merged, summary = merge_states(base, refit, PREFIXES)
    assert torch.equal(merged["Qformer.bert.w"], refit["Qformer.bert.w"])
    assert torch.equal(merged["query_tokens"], refit["query_tokens"])
    assert torch.equal(merged["qformer_input_norm.norms.biovil.weight"], torch.ones(8))
    # the encoder block the refit checkpoint does not hold comes from the base
    assert torch.equal(merged["visual_encoder.encoder.layer4.w"], base["visual_encoder.encoder.layer4.w"])
    assert torch.equal(merged["mhcac.w"], base["mhcac.w"])
    assert summary["replaced"] == 2 and summary["added"] == 1
    assert summary["verified_identical"] == 2 and summary["base_only_kept"] == 1


def test_a_refit_that_moved_mhcac_or_a_buffer_is_refused():
    base = _base()
    for key in ("mhcac.w", "visual_encoder.bn.running_mean"):
        refit = _refit(base)
        refit[key] = refit[key] + 1e-6
        with pytest.raises(MergeError, match="changed 1 tensors"):
            merge_states(base, refit, PREFIXES)


def test_a_refit_missing_a_qformer_tensor_or_adding_a_stray_one_is_refused():
    base = _base()
    refit = _refit(base)
    del refit["query_tokens"]
    with pytest.raises(MergeError, match="lacks 1 tensors"):
        merge_states(base, refit, PREFIXES)
    refit = _refit(base)
    refit["brand_new.w"] = torch.zeros(1)
    with pytest.raises(MergeError, match="does not have outside"):
        merge_states(base, refit, PREFIXES)


def test_prefixes_come_from_the_refit_phase_block():
    config = {"run": {"phase": "phase1q", "phases": {"phase1q": {
        "trainable": ["query_tokens", "Qformer", "qformer_input_norm"],
        "init_from": {"path": "/x", "skip": ["Qformer", "itm_head"]}}}}}
    assert refit_prefixes(config) == ("query_tokens", "Qformer", "qformer_input_norm", "itm_head")
    with pytest.raises(MergeError, match="names no phase"):
        refit_prefixes({"run": {"phase": None}})
