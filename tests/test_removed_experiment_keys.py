"""Config knobs of experiments removed on 2026-11-27 are refused, not ignored."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from pretraining.retired_keys import (
    REMOVED_EXPERIMENT_MODEL_KEYS,
    RetiredConfigKey,
    reject_removed_experiment_keys,
)

REPO_CONFIG = "pretraining/configs/mimic_cxr_full.yaml"


@pytest.mark.parametrize("section,key", REMOVED_EXPERIMENT_MODEL_KEYS)
def test_each_removed_key_is_refused(section, key):
    cfg = {key: 1} if section is None else {section: {key: 1}}
    with pytest.raises(RetiredConfigKey, match="2026-11-27"):
        reject_removed_experiment_keys(OmegaConf.create(cfg))


def test_the_shipped_yaml_and_its_phases_carry_none_of_them():
    config = OmegaConf.load(REPO_CONFIG)
    reject_removed_experiment_keys(config.model)
    for phase in config.run.phases.values():
        if "model" in phase:
            reject_removed_experiment_keys(phase.model)
    assert "phase1q" not in config.run.phases


def test_an_unrelated_config_passes():
    reject_removed_experiment_keys(OmegaConf.create({"view_fusion": {"p_view_drop": 0.15}}))
