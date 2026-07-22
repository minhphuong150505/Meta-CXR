"""Split-manifest loading for the native MedGemma pipeline.

Reads the preprocessed split CSVs directly and deliberately imports neither
LAVIS nor the Stage-1 model stack: ``medgemma_direct`` must be runnable with no
Stage-1 checkpoint, no Stage-1 config, no Q-Former and no MHCAC.

Only pandas is required, so the split/section/leakage invariants stay testable
on a CPU box.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

try:
    from stage2_utils import stable_fingerprint
except ImportError:  # ``python -m training...``
    from training.stage2_utils import stable_fingerprint


FINDINGS_ONLY = "findings_only"
IMPRESSION_ONLY = "impression_only"
FINDINGS_AND_IMPRESSION = "findings_and_impression"
SECTION_MODES = (FINDINGS_ONLY, IMPRESSION_ONLY, FINDINGS_AND_IMPRESSION)
DEFAULT_SECTION_MODE = FINDINGS_AND_IMPRESSION

FINDINGS_HEADER = "FINDINGS:"
IMPRESSION_HEADER = "IMPRESSION:"

# Emitted by preporcessing/preprocess_mimic_cxr.py. ``impression_clean`` and
# ``impression_valid`` were added when IMPRESSION became a supported target; a
# manifest built before that lacks them and cannot serve impression modes.
REQUIRED_COLUMNS = (
    "subject_id",
    "study_id",
    "dicom_id",
    "image_path",
    "findings_clean",
    "target_valid",
)
IMPRESSION_COLUMNS = ("impression_clean", "impression_valid")

DEFAULT_ANCHOR_PRIORITY = ("PA", "AP", "LATERAL", "LL", "UNKNOWN")

IDENTIFIER_COLUMNS = ("subject_id", "study_id", "dicom_id")


class ManifestError(RuntimeError):
    """Raised when a manifest cannot safely serve the requested target."""


def format_report(findings: str, impression: str, section_mode: str) -> str:
    """Render a target/prediction string for the requested section mode.

    Single-section modes emit bare text, matching the existing FINDINGS-only
    adapters. ``findings_and_impression`` emits explicit headers so the two
    sections can be scored separately and neither can be silently substituted
    for the other.
    """
    findings = (findings or "").strip()
    impression = (impression or "").strip()
    if section_mode == FINDINGS_ONLY:
        return findings
    if section_mode == IMPRESSION_ONLY:
        return impression
    if section_mode == FINDINGS_AND_IMPRESSION:
        return f"{FINDINGS_HEADER} {findings}\n\n{IMPRESSION_HEADER} {impression}"
    raise ValueError(f"unknown section mode {section_mode!r}")


def split_generated_report(text: str) -> tuple[str, str]:
    """Split a structured generation back into (findings, impression).

    An unheadered generation is treated as FINDINGS with an empty IMPRESSION
    rather than being duplicated into both, so an omitted IMPRESSION is scored
    as a miss instead of being silently filled in.
    """
    if not isinstance(text, str) or not text.strip():
        return "", ""
    upper = text.upper()
    impression_at = upper.find(IMPRESSION_HEADER)
    if impression_at < 0:
        head = text
        impression = ""
    else:
        head = text[:impression_at]
        impression = text[impression_at + len(IMPRESSION_HEADER):].strip()
    findings_at = head.upper().find(FINDINGS_HEADER)
    if findings_at >= 0:
        head = head[findings_at + len(FINDINGS_HEADER):]
    return head.strip(), impression


def row_target(row: Mapping[str, Any], section_mode: str) -> str:
    """Return the training target for one row, or "" when it is unusable.

    ``findings_and_impression`` requires *both* sections to be valid. Accepting
    a row with only one would train the model to emit an empty section.
    """
    findings_ok = bool(row.get("target_valid"))
    impression_ok = bool(row.get("impression_valid"))
    findings = str(row.get("findings_clean") or "").strip()
    impression = str(row.get("impression_clean") or "").strip()
    if section_mode == FINDINGS_ONLY:
        return findings if findings_ok and findings else ""
    if section_mode == IMPRESSION_ONLY:
        return impression if impression_ok and impression else ""
    if section_mode == FINDINGS_AND_IMPRESSION:
        if not (findings_ok and impression_ok and findings and impression):
            return ""
        return format_report(findings, impression, section_mode)
    raise ValueError(f"unknown section mode {section_mode!r}")


def assert_columns(frame: pd.DataFrame, section_mode: str, source: str) -> None:
    missing = [name for name in REQUIRED_COLUMNS if name not in frame.columns]
    if section_mode in (IMPRESSION_ONLY, FINDINGS_AND_IMPRESSION):
        missing += [name for name in IMPRESSION_COLUMNS if name not in frame.columns]
    if missing:
        raise ManifestError(
            f"{source} is missing column(s) {', '.join(sorted(set(missing)))}. "
            f"Section mode {section_mode!r} needs a manifest rebuilt with the "
            "current preporcessing/preprocess_mimic_cxr.py."
        )


def select_anchor_rows(
    frame: pd.DataFrame, anchor_priority: Iterable[str] = DEFAULT_ANCHOR_PRIORITY
) -> pd.DataFrame:
    """Reduce image-level rows to one anchor view per study.

    Sampling one row per study stops multi-view studies from being over-weighted
    and stops a single report being duplicated once per image.
    """
    if "ViewPosition" not in frame.columns:
        return frame.drop_duplicates(["subject_id", "study_id"], keep="first")
    order = {name.upper(): rank for rank, name in enumerate(anchor_priority)}
    fallback = len(order)
    view = frame["ViewPosition"].astype("string").fillna("UNKNOWN").str.upper()
    ranked = frame.assign(_view_rank=view.map(lambda v: order.get(v, fallback)))
    ranked = ranked.sort_values(
        ["subject_id", "study_id", "_view_rank", "dicom_id"], kind="stable"
    )
    anchors = ranked.drop_duplicates(["subject_id", "study_id"], keep="first")
    return anchors.drop(columns="_view_rank")


def assert_no_leakage(frames: Mapping[str, pd.DataFrame]) -> None:
    """Fail fast when a subject, study or image appears in two splits."""
    seen: dict[str, dict[Any, str]] = {column: {} for column in IDENTIFIER_COLUMNS}
    # subject_id alone is the strongest constraint; study_id and dicom_id are
    # checked as composites so numerically equal ids under different subjects
    # are not reported as false leaks.
    keyed = {
        "subject_id": lambda row: row[0],
        "study_id": lambda row: (row[0], row[1]),
        "dicom_id": lambda row: row[2],
    }
    violations: list[str] = []
    for split_name, frame in frames.items():
        if frame.empty:
            continue
        rows = frame[list(IDENTIFIER_COLUMNS)].itertuples(index=False, name=None)
        local: dict[str, set] = {column: set() for column in IDENTIFIER_COLUMNS}
        for row in rows:
            for column, make_key in keyed.items():
                local[column].add(make_key(row))
        for column in IDENTIFIER_COLUMNS:
            for key in local[column]:
                previous = seen[column].get(key)
                if previous is not None and previous != split_name:
                    violations.append(
                        f"{column} appears in both {previous!r} and {split_name!r}"
                    )
                    break
                seen[column][key] = split_name
    if violations:
        raise ManifestError(
            "split leakage detected; refusing to build records: "
            + "; ".join(sorted(set(violations)))
        )


def deterministic_subset(records: list[dict], limit: int, seed: int) -> list[dict]:
    if limit <= 0 or limit >= len(records):
        return records
    rng = random.Random(seed)
    return [records[index] for index in sorted(rng.sample(range(len(records)), limit))]


def build_records(
    frame: pd.DataFrame,
    *,
    split: str,
    section_mode: str,
    vis_root: str | Path,
    cohort_id: str,
    limit: int = 0,
    seed: int = 16,
    anchor_priority: Iterable[str] = DEFAULT_ANCHOR_PRIORITY,
    require_image: bool = False,
) -> list[dict]:
    """Build native-mode records from one split frame.

    Records carry no raw MIMIC identifiers: ``sample_key`` is a salted digest so
    evaluation artifacts stay publishable while remaining joinable locally.
    """
    assert_columns(frame, section_mode, f"{split} manifest")
    anchors = select_anchor_rows(frame, anchor_priority)
    records: list[dict] = []
    skipped_target = 0
    skipped_missing_image = 0
    for row in anchors.to_dict("records"):
        target = row_target(row, section_mode)
        if not target:
            skipped_target += 1
            continue
        image_path = os.path.join(str(vis_root), str(row["image_path"]))
        if require_image and not Path(image_path).is_file():
            skipped_missing_image += 1
            continue
        anchor_view = row.get("ViewPosition")
        anchor_view = str(anchor_view).strip().upper() if anchor_view not in (None, "") else None
        records.append(
            {
                "index": len(records),
                "sample_key": stable_fingerprint(
                    {"cohort": cohort_id, "dicom": str(row["dicom_id"])}, length=24
                ),
                "ref": target,
                "image_path": image_path,
                # View metadata for the v2 prompt builder. Absent ViewPosition
                # stays None rather than being invented; auxiliary views are not
                # threaded here (native anchor-only), so the list is empty.
                "anchor_view": anchor_view,
                "auxiliary_views": [],
            }
        )
    print(
        f"[manifest:{split}] {len(records)} records "
        f"(section_mode={section_mode}, skipped_target={skipped_target}, "
        f"skipped_missing_image={skipped_missing_image})",
        flush=True,
    )
    if not records:
        raise ManifestError(
            f"{split} manifest produced no usable records for section mode "
            f"{section_mode!r}; every row failed its validity flags."
        )
    return deterministic_subset(records, limit, seed)
