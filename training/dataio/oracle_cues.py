"""Training-time cues from the TRUE CheXpert labels, mixed with the model's own.

Added 2026-10-11 (``run_medgemma_qlora.py --train-cues oracle_mix``). Why: the v3
Stage-2 model IGNORES its cue channel -- replacing its cues with cues from the
true labels changed no metric (1,000 test studies), so a better Stage-1
classifier could never improve its reports. A model trained on cues that are
usually right is expected to learn to rely on them (RaDialog: structured
findings raised CE 26.1 -> 39.4). At inference the cues are MHCAC's.

Per (study, finding) the training cue is the true class, except with
probability ``model_share`` (deterministic in seed, study and finding) it is
the class MHCAC predicted -- so the model also sees the classifier's typical
errors. A finding with no CheXpert information keeps MHCAC's class.

Label mapping is the Stage-1 rule (D-018, ``blank_label_policy: negative``,
``model/lavis/data/chexpert_labels.py``): 1 -> Positive, 0 -> Negative,
-1 -> Uncertain, blank -> Negative when the study has any CheXpert value,
unknown when it has none. Re-implemented here with pandas so this module
imports nothing from LAVIS (tests/test_native_independence.py);
``tests/test_oracle_cues.py`` pins it against ``chexpert_labels``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

GROUP_NAMES = ("negative", "positive", "uncertain")   # class index 0, 1, 2
UNKNOWN = -1


def map_chexpert_rows(frame: pd.DataFrame, findings: Sequence[str]) -> np.ndarray:
    """[N, len(findings)] class codes under the blank-as-negative rule."""
    values = frame[list(findings)].to_numpy(dtype=float)
    has_any = ~np.isnan(values).all(axis=1)
    codes = np.full(values.shape, UNKNOWN, dtype=np.int64)
    codes[values == 1.0] = 1
    codes[values == 0.0] = 0
    codes[values == -1.0] = 2
    blank = np.isnan(values) & has_any[:, None]
    codes[blank] = 0
    return codes


def true_label_codes(
    records: Iterable[dict],
    *,
    split_csv: str | Path,
    chexpert_csv: str | Path,
    findings: Sequence[str],
) -> np.ndarray:
    """[N, len(findings)] true class codes for the records' anchor studies."""
    records = list(records)
    df = pd.read_csv(split_csv, usecols=["subject_id", "study_id", "dicom_id"])
    by_dicom = df.drop_duplicates("dicom_id").set_index("dicom_id")
    chexpert = pd.read_csv(chexpert_csv)
    lut = chexpert.drop_duplicates(["subject_id", "study_id"]).set_index(["subject_id", "study_id"])
    rows = map_chexpert_rows(lut, findings)
    position = {key: i for i, key in enumerate(lut.index)}
    out = np.full((len(records), len(findings)), UNKNOWN, dtype=np.int64)
    for i, record in enumerate(records):
        dicom = Path(str(record.get("image_path", ""))).stem
        if dicom not in by_dicom.index:
            continue
        key = (int(by_dicom.at[dicom, "subject_id"]), int(by_dicom.at[dicom, "study_id"]))
        if key in position:
            out[i] = rows[position[key]]
    return out


def _uniform(seed: int, study: str, finding: str) -> float:
    digest = hashlib.blake2b(f"{seed}|{study}|{finding}".encode(), digest_size=8).hexdigest()
    return int(digest, 16) / 2**64


def model_classes(record: dict) -> dict[str, int]:
    """finding -> class index of the record's current (MHCAC) cue groups."""
    groups = record.get("pred_groups") or {}
    return {f: GROUP_NAMES.index(name) for name in GROUP_NAMES for f in groups.get(name, [])}


def mix_oracle_cues(
    records: list[dict],
    codes: np.ndarray,
    findings: Sequence[str],
    *,
    model_share: float,
    seed: int,
    skip: Sequence[str] = ("No Finding",),
) -> dict[str, int]:
    """Replace each record's ``pred_groups`` by the oracle mix; return counts only."""
    if not 0.0 <= model_share <= 1.0:
        raise ValueError(f"model_share must be in [0, 1], got {model_share}")
    if codes.shape != (len(records), len(findings)):
        raise ValueError(f"codes {codes.shape} do not match {len(records)} records x {len(findings)}")
    counts = {"records": len(records), "cells": 0, "from_truth": 0, "from_model": 0,
              "changed": 0, "unknown_truth": 0}
    for record, row in zip(records, codes):
        current = model_classes(record)
        key = str(record.get("sample_key", record.get("image_path", "")))
        groups: dict[str, list[str]] = {name: [] for name in GROUP_NAMES}
        for finding, code in zip(findings, row):
            if finding in skip:
                continue
            counts["cells"] += 1
            if code == UNKNOWN:
                counts["unknown_truth"] += 1
            use_model = code == UNKNOWN or _uniform(seed, key, finding) < model_share
            cls = current.get(finding) if use_model else int(code)
            counts["from_model" if use_model else "from_truth"] += 1
            if cls is None:
                continue
            counts["changed"] += int(cls != current.get(finding))
            groups[GROUP_NAMES[cls]].append(finding)
        record["pred_groups"] = groups
        record["cue_state"] = "predicted"
        record["cue_source"] = "oracle_mix"
    return counts
