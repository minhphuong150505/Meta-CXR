"""Training cues from the true labels mixed with MHCAC's (``--train-cues oracle_mix``)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.dataio.oracle_cues import (  # noqa: E402
    UNKNOWN,
    map_chexpert_rows,
    mix_oracle_cues,
    true_label_codes,
)

FINDINGS = ("No Finding", "Edema", "Cardiomegaly", "Fracture")


def test_mapping_matches_the_stage1_rule():
    from model.lavis.data.chexpert_labels import BLANK_AS_NEGATIVE, IGNORE_LABEL, map_chexpert_labels

    frame = pd.DataFrame({
        "No Finding": [1.0, np.nan, np.nan],
        "Edema": [np.nan, -1.0, np.nan],
        "Cardiomegaly": [0.0, 1.0, np.nan],
        "Fracture": [np.nan, np.nan, np.nan],
    })
    ours = map_chexpert_rows(frame, FINDINGS)
    stage1 = map_chexpert_labels(frame, list(FINDINGS), BLANK_AS_NEGATIVE).to_numpy()
    stage1 = np.where(stage1 == IGNORE_LABEL, UNKNOWN, stage1)
    assert (ours == stage1).all()
    assert (ours[2] == UNKNOWN).all()          # no CheXpert information at all


def test_true_codes_join_on_the_anchor_dicom(tmp_path):
    pd.DataFrame({"subject_id": [1, 2], "study_id": [10, 20], "dicom_id": ["a", "b"]}).to_csv(
        tmp_path / "split.csv", index=False)
    pd.DataFrame({"subject_id": [1], "study_id": [10], "No Finding": [np.nan], "Edema": [1.0],
                  "Cardiomegaly": [np.nan], "Fracture": [-1.0]}).to_csv(
        tmp_path / "chexpert.csv", index=False)
    records = [{"image_path": "/x/a.jpg"}, {"image_path": "/x/b.jpg"}]
    codes = true_label_codes(records, split_csv=tmp_path / "split.csv",
                             chexpert_csv=tmp_path / "chexpert.csv", findings=FINDINGS)
    assert codes[0].tolist() == [0, 1, 0, 2]
    assert (codes[1] == UNKNOWN).all()


def _records():
    return [{"sample_key": f"s{i}",
             "pred_groups": {"positive": ["Cardiomegaly"], "negative": ["Edema", "Fracture"],
                             "uncertain": []}} for i in range(50)]


def test_share_zero_uses_the_truth_and_share_one_keeps_the_model():
    codes = np.tile(np.array([[1, 1, 0, UNKNOWN]]), (50, 1))
    truth = _records()
    counts = mix_oracle_cues(truth, codes, FINDINGS, model_share=0.0, seed=16)
    assert truth[0]["pred_groups"] == {"positive": ["Edema"], "negative": ["Cardiomegaly", "Fracture"],
                                       "uncertain": []}
    assert counts["from_truth"] == 100 and counts["unknown_truth"] == 50   # Fracture unknown
    assert all(r["cue_source"] == "oracle_mix" for r in truth)
    kept = _records()
    mix_oracle_cues(kept, codes, FINDINGS, model_share=1.0, seed=16)
    assert kept[0]["pred_groups"] == _records()[0]["pred_groups"]


def test_mix_is_deterministic_and_near_the_requested_share():
    codes = np.tile(np.array([[1, 1, 0, 1]]), (50, 1))
    a, b = _records(), _records()
    ca = mix_oracle_cues(a, codes, FINDINGS, model_share=0.2, seed=16)
    mix_oracle_cues(b, codes, FINDINGS, model_share=0.2, seed=16)
    assert [r["pred_groups"] for r in a] == [r["pred_groups"] for r in b]
    assert 0.1 < ca["from_model"] / ca["cells"] < 0.3
    assert all("No Finding" not in sum(r["pred_groups"].values(), []) for r in a)


def test_bad_share_is_refused():
    with pytest.raises(ValueError):
        mix_oracle_cues(_records(), np.zeros((50, 4), dtype=int), FINDINGS, model_share=1.5, seed=0)
