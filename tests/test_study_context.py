"""Study context for the Stage-2 prompt: INDICATION/HISTORY + the previous FINDINGS.

Synthetic split CSV, metadata and report files only; no MIMIC data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.dataio.study_context import (  # noqa: E402
    MAX_PRIOR_WORDS,
    attach_study_context,
    format_context_lines,
    prior_findings_table,
)

# subject 10000001: studies 1 (day 1), 2 (day 5, invalid target), 3 (day 9), 4 (day 9, later time)
# subject 10000002: study 5 only
ROWS = [
    (10000001, 1, "d1", "first findings.", True, 20200101, 80000.0),
    (10000001, 2, "d2", "", False, 20200105, 80000.0),
    (10000001, 3, "d3", "third findings.", True, 20200109, 80000.0),
    (10000001, 4, "d4", "fourth findings.", True, 20200109, 120000.0),
    (10000002, 5, "d5", "other patient.", True, 20200102, 90000.0),
]


@pytest.fixture
def data(tmp_path):
    split = pd.DataFrame(
        [{"subject_id": s, "study_id": st, "dicom_id": d, "findings_clean": f, "target_valid": v}
         for s, st, d, f, v, _, _ in ROWS])
    split_csv = tmp_path / "test.csv"
    split.to_csv(split_csv, index=False)
    meta = pd.DataFrame([{"dicom_id": d, "subject_id": s, "study_id": st, "StudyDate": date,
                          "StudyTime": time} for s, st, d, _, _, date, time in ROWS])
    meta_csv = tmp_path / "metadata.csv"
    meta.to_csv(meta_csv, index=False)
    root = tmp_path / "reports"
    for s, st, *_ in ROWS:
        folder = root / f"p{str(s)[:2]}" / f"p{s}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"s{st}.txt").write_text(
            f"INDICATION: reason {st}.\nFINDINGS: target text {st}.\nIMPRESSION: impression {st}.\n")
    return split_csv, meta_csv, root


def test_prior_is_the_latest_strictly_earlier_valid_study_of_the_same_patient(data):
    split_csv, meta_csv, _ = data
    table = prior_findings_table(split_csv, meta_csv)
    assert pd.isna(table.loc[1, "prior_findings"])          # first study: no prior
    assert table.loc[2, "prior_findings"] == "first findings."
    # study 3: study 2 is earlier but has no valid target -> skipped to study 1
    assert table.loc[3, "prior_findings"] == "first findings." and table.loc[3, "prior_days"] == 8
    # study 4, same day as 3 but later: 3 is its prior (same day, 0 days)
    assert table.loc[4, "prior_findings"] == "third findings." and table.loc[4, "prior_days"] == 0
    # another patient's studies never leak in
    assert pd.isna(table.loc[5, "prior_findings"])


def test_a_later_study_never_becomes_a_prior(data):
    split_csv, meta_csv, _ = data
    table = prior_findings_table(split_csv, meta_csv)
    assert "fourth findings." not in set(table["prior_findings"].dropna())


def test_attach_uses_indication_not_the_target(data):
    split_csv, meta_csv, root = data
    records = [{"image_path": f"/x/files/{d}.jpg"} for d in ("d3", "d5")]
    counts = attach_study_context(records, split_csv=split_csv, metadata_csv=meta_csv,
                                  reports_root=root)
    assert counts == {"n": 2, "unmatched": 0, "with_prior": 1, "with_indication": 2}
    ctx = records[0]["context"]
    assert ctx["indication"] == "reason 3." and ctx["prior_findings"] == "first findings."
    assert "target text" not in str(ctx) and "impression" not in str(ctx)
    assert records[1]["context"]["prior_findings"] is None


def test_unmatched_records_fail_instead_of_training_on_empty_context(data):
    split_csv, meta_csv, root = data
    records = [{"image_path": "/x/unknown.jpg"}, {"image_path": "/x/d1.jpg"}]
    with pytest.raises(ValueError, match="no row"):
        attach_study_context(records, split_csv=split_csv, metadata_csv=meta_csv,
                             reports_root=root)


def test_prompt_lines_name_missing_values_explicitly():
    text = format_context_lines({"indication": None, "prior_findings": None, "prior_days": None})
    assert "Indication: not provided" in text and "no earlier study available" in text
    text = format_context_lines({"indication": "cough", "prior_findings": "clear.", "prior_days": 3})
    assert "Indication: cough" in text and "(3 days earlier): clear." in text
    assert "(same day)" in format_context_lines(
        {"indication": "x", "prior_findings": "y", "prior_days": 0})
    with pytest.raises(ValueError):
        format_context_lines(None)


def test_long_priors_are_truncated(data, tmp_path):
    split_csv, meta_csv, root = data
    df = pd.read_csv(split_csv)
    df.loc[df.study_id == 1, "findings_clean"] = " ".join(["word"] * (MAX_PRIOR_WORDS + 50))
    df.to_csv(split_csv, index=False)
    records = [{"image_path": "/x/d3.jpg"}]
    attach_study_context(records, split_csv=split_csv, metadata_csv=meta_csv, reports_root=root)
    assert len(records[0]["context"]["prior_findings"].split()) == MAX_PRIOR_WORDS + 1
