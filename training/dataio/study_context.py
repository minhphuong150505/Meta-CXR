"""Clinical context for a study: its INDICATION/HISTORY and the previous FINDINGS.

Stage-2 prompt input added 2026-10-11 (``--prompt-style paper_context``). The
evidence for it, measured on this project's test split: copying the patient's
previous FINDINGS alone (no model) already matches the META-CXR paper's
BLEU-1/BLEU-4 and beats the v3 model by 0.12 lexicon CE; 90% of test FINDINGS
use temporal/comparison language; 91.7% of test studies carry INDICATION or
HISTORY. See ``docs/TONG_HOP_2026-10-11.md``.

Leakage rules, pinned by ``tests/test_study_context.py``:

* the previous study is the latest study of the SAME patient strictly EARLIER
  in time (``StudyDate`` + ``StudyTime``), never the study itself, never a
  later one; only studies whose FINDINGS target is valid are candidates;
* the indication comes from the current report's INDICATION/HISTORY sections,
  never from FINDINGS or IMPRESSION.

pandas only (plus the stdlib report parser), so it is testable on a CPU box.
Everything it returns is report text: keep it in memory or in a sensitive
cache, never in a log, a summary or git.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd

try:  # repo root on sys.path (python -m ...)
    from preporcessing.mimic_report_parser import clean_report_text, clinical_context
except ImportError:  # pragma: no cover - script layout
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "preporcessing"))
    from mimic_report_parser import clean_report_text, clinical_context  # type: ignore

CONTEXT_KEY = "context"
MAX_INDICATION_WORDS = 60
MAX_PRIOR_WORDS = 150


def _truncate(text: str, max_words: int) -> str:
    words = text.split()
    return " ".join(words[:max_words]) + (" ..." if len(words) > max_words else "")


def _valid(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(("true", "1", "1.0"))


def study_times(metadata_csv: str | Path) -> pd.Series:
    """study_id -> earliest acquisition time of the study (Timestamp)."""
    meta = pd.read_csv(metadata_csv, usecols=["study_id", "StudyDate", "StudyTime"])
    date = pd.to_datetime(meta["StudyDate"].astype("int64").astype(str), format="%Y%m%d")
    seconds = meta["StudyTime"].astype(float).fillna(0.0)
    hh, rest = (seconds // 10000), seconds % 10000
    mm, ss = rest // 100, rest % 100
    when = date + pd.to_timedelta(hh * 3600 + mm * 60 + ss, unit="s")
    return pd.Series(when.values, index=meta["study_id"].values).groupby(level=0).min()


def prior_findings_table(split_csv: str | Path, metadata_csv: str | Path) -> pd.DataFrame:
    """Per study of the split: the previous valid study's FINDINGS and the gap in days.

    Index ``study_id``; columns ``prior_findings`` (str or NaN), ``prior_days``.
    """
    df = pd.read_csv(split_csv, usecols=["subject_id", "study_id", "findings_clean", "target_valid"])
    times = study_times(metadata_csv)
    studies = df.drop_duplicates("study_id").copy()
    studies["t"] = studies["study_id"].map(times)
    if studies["t"].isna().any():
        missing = int(studies["t"].isna().sum())
        raise ValueError(f"{missing} studies of {Path(split_csv).name} have no StudyDate")
    candidates = studies[_valid(studies["target_valid"])
                         & studies["findings_clean"].fillna("").str.strip().ne("")]
    left = studies[["subject_id", "study_id", "t"]].sort_values("t")
    right = (candidates[["subject_id", "study_id", "t", "findings_clean"]]
             .rename(columns={"study_id": "prior_study_id", "t": "prior_t"})
             .sort_values("prior_t"))
    merged = pd.merge_asof(left, right, left_on="t", right_on="prior_t", by="subject_id",
                           allow_exact_matches=False, direction="backward")
    # allow_exact_matches=False: the previous study is STRICTLY earlier, so a
    # study can never be its own prior, and a later study can never leak in.
    merged["prior_days"] = (merged["t"] - merged["prior_t"]).dt.days
    out = merged.set_index("study_id")[["findings_clean", "prior_days"]]
    return out.rename(columns={"findings_clean": "prior_findings"})


def read_indication(reports_root: str | Path, subject_id: int, study_id: int) -> str:
    """INDICATION/HISTORY of one raw MIMIC-CXR report, cleaned; "" if none."""
    subject = str(int(subject_id))
    path = Path(reports_root) / f"p{subject[:2]}" / f"p{subject}" / f"s{int(study_id)}.txt"
    if not path.is_file():
        return ""
    return clinical_context(path.read_text(encoding="utf-8", errors="replace"))


def _dicom_of(record: dict) -> str:
    return Path(str(record.get("image_path", ""))).stem


def attach_study_context(
    records: Iterable[dict],
    *,
    split_csv: str | Path,
    metadata_csv: str | Path,
    reports_root: str | Path,
    max_unmatched_fraction: float = 0.01,
) -> dict[str, Any]:
    """Add ``record["context"]`` to every record; return counts only.

    Records are matched to the split CSV by the anchor image's DICOM id (the
    file stem of ``image_path``). Fails when more than ``max_unmatched_fraction``
    of the records cannot be matched, rather than training on empty context.
    """
    records = list(records)
    df = pd.read_csv(split_csv, usecols=["subject_id", "study_id", "dicom_id"])
    by_dicom = df.drop_duplicates("dicom_id").set_index("dicom_id")
    priors = prior_findings_table(split_csv, metadata_csv)
    indication_cache: dict[int, str] = {}
    counts = {"n": len(records), "unmatched": 0, "with_prior": 0, "with_indication": 0}
    for record in records:
        dicom = _dicom_of(record)
        if dicom not in by_dicom.index:
            counts["unmatched"] += 1
            record[CONTEXT_KEY] = {"indication": None, "prior_findings": None, "prior_days": None}
            continue
        subject_id, study_id = int(by_dicom.at[dicom, "subject_id"]), int(by_dicom.at[dicom, "study_id"])
        if study_id not in indication_cache:
            indication_cache[study_id] = read_indication(reports_root, subject_id, study_id)
        indication = indication_cache[study_id] or None
        prior, days = None, None
        if study_id in priors.index:
            row = priors.loc[study_id]
            text = row["prior_findings"]
            if isinstance(text, str) and text.strip():
                prior = clean_report_text(text)
                days = int(row["prior_days"])
        record[CONTEXT_KEY] = {
            "indication": _truncate(indication, MAX_INDICATION_WORDS) if indication else None,
            "prior_findings": _truncate(prior, MAX_PRIOR_WORDS) if prior else None,
            "prior_days": days,
        }
        counts["with_prior"] += prior is not None
        counts["with_indication"] += indication is not None
    if counts["n"] and counts["unmatched"] / counts["n"] > max_unmatched_fraction:
        raise ValueError(
            f"{counts['unmatched']} of {counts['n']} records have no row in "
            f"{Path(split_csv).name}; the context would be silently empty"
        )
    return counts


def format_context_lines(context: dict | None) -> str:
    """The two prompt lines. A missing value is said explicitly, never left out."""
    if context is None:
        raise ValueError("record carries no study context; run attach_study_context first")
    indication = context.get("indication") or "not provided"
    prior = context.get("prior_findings")
    days = context.get("prior_days")
    if prior:
        when = "same day" if days == 0 else f"{days} day{'s' if days != 1 else ''} earlier"
        prior_line = f"Previous study findings ({when}): {prior}"
    else:
        prior_line = "Previous study findings: no earlier study available."
    return f"Indication: {indication}\n{prior_line}"

