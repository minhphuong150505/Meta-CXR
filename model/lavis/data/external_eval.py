"""External evaluation sets served through ``MIMIC_CXR_Dataset`` (paper Table 4).

Only pandas, so the CPU test suite can pin the conversion without importing
torchvision, transformers or LAVIS.

The paper's Table 4 scores the Stage-1 classifier, trained on MIMIC-CXR only,
on the CheXpert validation set: 200 studies / 234 images labelled 0/1 by a
radiologist consensus. Rather than a second dataset class, this module turns
CheXpert's ``valid.csv`` into the two tables ``MIMIC_CXR_Dataset`` already
reads -- the per-image split rows and the per-study label rows -- so the image
preprocessing, view selection and collation are exactly the ones every
MIMIC-CXR number in this repo went through.

Enabled by ``datasets.mimic_cxr.external_eval`` in the run config; absent means
MIMIC-CXR, unchanged::

    datasets:
      mimic_cxr:
        external_eval:
          kind: chexpert_val
          valid_csv: <dir>/CheXpert-v1.0-small/valid.csv
          vis_root: <dir>          # the directory CONTAINING CheXpert-v1.0-small/
          splits: [test]           # which eval split names it replaces

CheXpert carries no report, so every row has an empty FINDINGS target and
``target_valid=False``: the vision-language losses see no valid pair and return
zero, and only the classifier is scored.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

CHEXPERT_VAL = "chexpert_val"
EXTERNAL_KINDS = (CHEXPERT_VAL,)
EVAL_SPLITS = ("val", "test")
_REQUIRED_KEYS = ("kind", "valid_csv", "vis_root", "splits")

# CheXpert's own path layout:
#   CheXpert-v1.0-small/valid/patient64541/study1/view1_frontal.jpg
# Anchored on "/valid/" so a train.csv (uncertain -1 labels, labeler-derived)
# can never be scored as if it were the radiologist-labelled validation set.
_VALID_PATH = re.compile(
    r"(?:^|/)valid/patient(?P<patient>\d+)/study(?P<study>\d+)/"
    r"(?P<view>view\d+_(?:frontal|lateral))\.(?:jpg|jpeg|png)$"
)


@dataclass(frozen=True)
class ExternalEval:
    kind: str
    valid_csv: Path
    vis_root: Path
    splits: tuple[str, ...]


def resolve_external_eval(raw, split: str) -> ExternalEval | None:
    """Validate ``datasets.mimic_cxr.external_eval``; ``None`` = MIMIC-CXR for ``split``.

    The train split is never replaced: an external set is evaluation-only.
    """
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise TypeError(f"external_eval must be a mapping, got {type(raw).__name__}")
    raw = dict(raw)
    unknown = sorted(set(raw) - set(_REQUIRED_KEYS))
    missing = [key for key in _REQUIRED_KEYS if raw.get(key) in (None, "")]
    if unknown or missing:
        raise ValueError(
            f"external_eval: unknown keys {unknown}, missing keys {missing}; "
            f"expected exactly {list(_REQUIRED_KEYS)}"
        )
    if raw["kind"] not in EXTERNAL_KINDS:
        raise ValueError(f"external_eval.kind {raw['kind']!r}; expected one of {EXTERNAL_KINDS}")
    splits = raw["splits"]
    if isinstance(splits, str):
        splits = [splits]
    splits = tuple(str(s) for s in splits)
    bad = sorted(set(splits) - set(EVAL_SPLITS))
    if not splits or bad:
        raise ValueError(
            f"external_eval.splits must be a non-empty subset of {EVAL_SPLITS} "
            f"(an external set never replaces train); got {list(splits)}"
        )
    if split not in splits:
        return None
    return ExternalEval(
        kind=raw["kind"],
        valid_csv=Path(str(raw["valid_csv"])),
        vis_root=Path(str(raw["vis_root"])),
        splits=splits,
    )


def chexpert_val_tables(
    valid: pd.DataFrame, label_cols: Sequence[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert CheXpert ``valid.csv`` rows into MIMIC-format ``(reports, labels)``.

    ``reports`` has one row per image with the columns ``MIMIC_CXR_Dataset``
    requires (``image_path`` kept relative, as CheXpert writes it); ``labels``
    has one row per study, the 14 columns as 0.0/1.0. Refuses anything that is
    not the radiologist-labelled validation set: paths outside ``valid/``, and
    any label other than 0/1 (no -1 uncertain, no blank).
    """
    label_cols = list(label_cols)
    required = ["Path", "Frontal/Lateral", "AP/PA", *label_cols]
    missing = [c for c in required if c not in valid.columns]
    if missing:
        raise ValueError(f"CheXpert valid.csv is missing columns: {missing}")
    if valid.empty:
        raise ValueError("CheXpert valid.csv has no rows")

    paths = valid["Path"].astype(str).str.replace("\\", "/", regex=False)
    parsed = paths.str.extract(_VALID_PATH)
    unmatched = parsed["patient"].isna()
    if unmatched.any():
        raise ValueError(
            f"{int(unmatched.sum())} rows are not CheXpert validation images "
            f"(expected .../valid/patientN/studyM/viewK_frontal|lateral.jpg); "
            f"first: {paths[unmatched].iloc[0]!r}"
        )
    if paths.str.startswith("/").any() or paths.str.contains(r"(?:^|/)\.\.(?:/|$)").any():
        raise ValueError("CheXpert paths must be relative and stay inside vis_root")

    values = valid[label_cols].apply(pd.to_numeric, errors="coerce")
    bad = ~values.isin([0.0, 1.0])
    if bad.to_numpy().any():
        col = label_cols[int(bad.any(axis=0).to_numpy().argmax())]
        raise ValueError(
            f"CheXpert validation labels must be 0/1 (radiologist consensus); "
            f"column {col!r} holds {sorted(map(str, valid.loc[bad[col], col].unique()))[:5]}"
        )

    frontal = valid["Frontal/Lateral"].astype(str).str.strip().str.lower()
    if not frontal.isin(["frontal", "lateral"]).all():
        raise ValueError("Frontal/Lateral must be Frontal or Lateral on every row")
    ap_pa = valid["AP/PA"].fillna("").astype(str).str.strip().str.upper()
    view_position = ap_pa.where(frontal.eq("frontal"), "LATERAL")
    # A frontal image without a usable AP/PA tag is UNKNOWN to view fusion,
    # exactly as an untagged MIMIC row is.
    view_position = view_position.where(
        ~frontal.eq("frontal") | ap_pa.isin(["AP", "PA"]), "UNKNOWN"
    )

    subject = parsed["patient"].astype(int)
    study = parsed["study"].astype(int)
    reports = pd.DataFrame({
        "subject_id": subject,
        "study_id": study,
        "dicom_id": "chexpert_p" + parsed["patient"] + "_s" + parsed["study"] + "_" + parsed["view"],
        "image_path": paths,
        "ViewPosition": view_position,
        "findings_clean": "",
        "target_valid": False,
        "extraction_method": "chexpert_val_no_report",
        "has_chexpert_label": True,
    })
    if reports["dicom_id"].duplicated().any():
        raise ValueError("CheXpert valid.csv lists the same image twice")

    per_image = pd.concat([reports[["subject_id", "study_id"]], values], axis=1)
    conflicting = per_image.groupby(["subject_id", "study_id"])[label_cols].nunique().gt(1)
    if conflicting.to_numpy().any():
        raise ValueError(
            f"{int(conflicting.any(axis=1).sum())} CheXpert studies carry different "
            "labels on different views; labels must be study-level"
        )
    labels = (
        per_image.groupby(["subject_id", "study_id"], sort=False)[label_cols]
        .first()
        .reset_index()
    )
    return reports.reset_index(drop=True), labels


def load_chexpert_val(
    external: ExternalEval, label_cols: Sequence[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read ``external.valid_csv`` and convert it; every image must exist under ``vis_root``."""
    if not external.valid_csv.is_file():
        raise FileNotFoundError(f"CheXpert valid.csv not found: {external.valid_csv}")
    reports, labels = chexpert_val_tables(pd.read_csv(external.valid_csv), label_cols)
    absent = [p for p in reports["image_path"] if not (external.vis_root / p).is_file()]
    if absent:
        raise FileNotFoundError(
            f"{len(absent)} of {len(reports)} CheXpert images are missing under "
            f"vis_root={external.vis_root} (it must CONTAIN CheXpert-v1.0-small/); "
            f"first: {absent[0]}"
        )
    return reports, labels
