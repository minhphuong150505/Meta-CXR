#!/usr/bin/env python3
"""Preprocess the full MIMIC-CXR (p10-p19) metadata + reports into train/val/test CSVs.

Script form of `kltn-data-preprocessing.ipynb`, with the Kaggle/gdown plumbing and the
exploratory plotting cells removed. Reads the three `.csv.gz` metadata files and the
per-study report .txt tree, emits the split CSVs consumed by
`model/lavis/data/ReportDataset.py::MIMIC_CXR_Dataset`.

Only CSVs and reports are touched — images are never read.

Two deliberate deviations from the notebook, both documented in
`plan/full-dataset-preprocessing.md`:
  * FINDINGS/IMPRESSION are extracted at study level (~227k rows) and merged onto the
    image level (~377k rows) afterwards, instead of merging the raw report text first.
    Identical output, several GB less peak RAM.
  * `image_path` is written RELATIVE (`files/p1X/pXXXXXXXX/sYYYYYYY/<dicom>.jpg`) because
    `ReportDataset._row_visual` re-anchors it with `os.path.join(vis_root, rel)`.
"""

from __future__ import annotations

import argparse
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# split.csv uses "validate"; the pipeline expects val.csv
SPLIT_TO_FILENAME = {"train": "train", "validate": "val", "test": "test"}
FRONTAL_VIEWS = ["PA", "AP"]
MIN_IMAGE_SIZE = 100


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", required=True,
                   help="dir holding mimic-cxr-2.0.0-{chexpert,metadata,split}.csv.gz")
    p.add_argument("--reports-root", required=True,
                   help="root of the report tree, i.e. the dir containing p10/ .. p19/")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--views", choices=["all", "frontal"], default="all",
                   help="'all' keeps every ViewPosition (NaN -> UNKNOWN); "
                        "'frontal' keeps only PA/AP")
    p.add_argument("--workers", type=int, default=16, help="threads for reading reports")
    p.add_argument("--min-words", type=int, default=3,
                   help="drop rows whose findings have fewer words than this")
    p.add_argument("--upper-quantile", type=float, default=0.99,
                   help="drop rows above this quantile of findings length")
    p.add_argument("--limit-studies", type=int, default=None,
                   help="debug: only process this many studies")
    return p.parse_args()


def clean_chexpert(chexpert_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Notebook cell 12. Keeps label-less studies, flags them via has_chexpert_label."""
    chexpert_df = chexpert_df.copy()
    chexpert_df["subject_id"] = chexpert_df["subject_id"].astype(int)
    chexpert_df["study_id"] = chexpert_df["study_id"].astype(int)

    dup_count = int(chexpert_df["study_id"].duplicated().sum())
    print(f"[chexpert] duplicated study_id: {dup_count}")
    if dup_count:
        chexpert_df = chexpert_df.drop_duplicates(subset=["study_id"], keep="first")

    label_cols = [c for c in chexpert_df.columns if c not in ("subject_id", "study_id")]
    print(f"[chexpert] label columns: {len(label_cols)}")

    no_label_mask = chexpert_df[label_cols].isnull().all(axis=1)
    print(f"[chexpert] studies with no label at all (kept, flagged): {int(no_label_mask.sum())}")

    out = chexpert_df.copy()
    out["has_chexpert_label"] = ~no_label_mask

    # 3-class mapping: 0=negative, 1=positive, 2=uncertain, NaN->0.
    # For has_chexpert_label=False rows the 0s are placeholders, NOT real negatives.
    for col in label_cols:
        mapped = pd.Series(0, index=out.index, dtype="int8")
        mapped[out[col] == 1.0] = 1
        mapped[out[col] == -1.0] = 2
        out[col] = mapped

    return out, label_cols


def clean_metadata(metadata_df: pd.DataFrame, views: str) -> pd.DataFrame:
    """Notebook cell 14, minus the boxplots."""
    metadata_df = metadata_df.copy()
    for col in ("subject_id", "study_id"):
        if col in metadata_df.columns:
            metadata_df[col] = metadata_df[col].astype(int)
    metadata_df["dicom_id"] = metadata_df["dicom_id"].astype(str)

    before = metadata_df.shape
    metadata_df = metadata_df.drop_duplicates().reset_index(drop=True)
    print(f"[metadata] {before} -> {metadata_df.shape} after dropping fully-identical rows")

    if metadata_df["ViewPosition"].dtype == object:
        metadata_df["ViewPosition"] = metadata_df["ViewPosition"].str.strip()
    metadata_df["ViewPosition"] = metadata_df["ViewPosition"].replace("", np.nan)

    # Drop only implausibly small images; high-res is normal for CXR.
    too_small = (metadata_df["Rows"] < MIN_IMAGE_SIZE) | (metadata_df["Columns"] < MIN_IMAGE_SIZE)
    print(f"[metadata] images smaller than {MIN_IMAGE_SIZE}px: {int(too_small.sum())}")
    metadata_df = metadata_df[~too_small].reset_index(drop=True)

    if views == "frontal":
        metadata_df = metadata_df[metadata_df["ViewPosition"].isin(FRONTAL_VIEWS)].reset_index(drop=True)
        print(f"[metadata] frontal-only filter -> {metadata_df.shape}")
    else:
        metadata_df["ViewPosition"] = metadata_df["ViewPosition"].fillna("UNKNOWN")

    print("[metadata] ViewPosition distribution:")
    print(metadata_df["ViewPosition"].value_counts().to_string())
    return metadata_df


def clean_split(split_df: pd.DataFrame) -> pd.DataFrame:
    """Notebook cell 19."""
    split_df = split_df.copy()
    split_df["subject_id"] = split_df["subject_id"].astype(int)
    split_df["study_id"] = split_df["study_id"].astype(int)
    split_df["dicom_id"] = split_df["dicom_id"].astype(str)

    before = split_df.shape
    split_df = split_df.drop_duplicates().reset_index(drop=True)
    print(f"[split] {before} -> {split_df.shape} after dropping duplicates")

    split_df["split"] = split_df["split"].str.strip().str.lower()

    leakage = int((split_df.groupby("subject_id")["split"].nunique() > 1).sum())
    print(f"[split] subjects appearing in more than one split (expect 0): {leakage}")
    if leakage:
        print("[split] WARNING: patient-level leakage detected — investigate before training")

    print("[split] distribution:")
    print(split_df["split"].value_counts().to_string())
    return split_df


def extract_sections(report_text: str) -> tuple[str, str]:
    """Notebook cell 32."""
    if not isinstance(report_text, str):
        return "", ""
    findings_match = re.search(r"FINDINGS:(.*?)(?=IMPRESSION:|$)", report_text,
                              re.DOTALL | re.IGNORECASE)
    impression_match = re.search(r"IMPRESSION:(.*?)(?=\n\n|\Z)", report_text,
                                 re.DOTALL | re.IGNORECASE)
    findings = findings_match.group(1).strip() if findings_match else ""
    impression = impression_match.group(1).strip() if impression_match else ""
    return findings, impression


def get_target_text(report_text: str) -> tuple[str, str, str]:
    """Notebook cell 32: FINDINGS with two fallbacks when the tag is absent."""
    findings, impression = extract_sections(report_text)
    if findings:
        return findings, impression, "FINDINGS_TAG"
    if not isinstance(report_text, str):
        return "", impression, "EMPTY"
    fallback = re.search(r"(?:reviewed in comparison to[^.]*\.\s*)(.*)", report_text,
                         re.DOTALL | re.IGNORECASE)
    if fallback:
        return fallback.group(1).strip(), impression, "FALLBACK_COMPARISON"
    return report_text.strip(), impression, "FALLBACK_RAW"


def clean_report_text(text: str) -> str:
    """Notebook cell 34."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"_{2,}", "", text)
    text = re.sub(r"\[\*\*.*?\*\*\]", "", text)
    text = re.sub(r"\s+([.,;:])", r"\1", text)
    return text.strip()


def report_path(reports_root: Path, subject_id: int, study_id: int) -> Path:
    sid = str(subject_id)
    return reports_root / f"p{sid[:2]}" / f"p{sid}" / f"s{study_id}.txt"


def build_study_text(studies: pd.DataFrame, reports_root: Path, workers: int) -> pd.DataFrame:
    """Read + parse every report once, at study level (notebook cells 29/32/34 fused).

    Returns one row per (subject_id, study_id) with the cleaned text columns only —
    report_raw is dropped before returning so it never reaches the image-level frame.
    """
    records = studies.to_dict("records")
    print(f"[reports] reading {len(records)} studies with {workers} threads")

    def read_one(rec: dict) -> str | None:
        path = report_path(reports_root, rec["subject_id"], rec["study_id"])
        try:
            return path.read_text(encoding="utf-8")
        except (FileNotFoundError, UnicodeDecodeError):
            return None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        raw = list(tqdm(ex.map(read_one, records), total=len(records), desc="reports"))

    n_missing = sum(r is None for r in raw)
    fail_rate = n_missing / len(raw) * 100 if raw else 0.0
    print(f"[reports] read ok: {len(raw) - n_missing}, missing: {n_missing} ({fail_rate:.2f}%)")
    if fail_rate > 5:
        raise SystemExit(
            f"[reports] {fail_rate:.2f}% of reports unreadable — check --reports-root "
            f"structure before trusting this output"
        )

    parsed = [get_target_text(r) for r in tqdm(raw, desc="sections")]
    del raw

    out = pd.DataFrame(records)
    out["findings_clean"] = [clean_report_text(p[0]) for p in parsed]
    out["impression_clean"] = [clean_report_text(p[1]) for p in parsed]
    out["extraction_method"] = [p[2] for p in parsed]
    del parsed

    print("[reports] extraction method distribution:")
    print(out["extraction_method"].value_counts().to_string())
    return out


def build_image_path(subject_id: int, study_id: int, dicom_id: str) -> str:
    """Relative path; ReportDataset joins it onto vis_root."""
    sid = str(subject_id)
    return f"files/p{sid[:2]}/p{sid}/s{study_id}/{dicom_id}.jpg"


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    reports_root = Path(args.reports_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== load ===")
    chexpert_df = pd.read_csv(raw_dir / "mimic-cxr-2.0.0-chexpert.csv.gz")
    split_df = pd.read_csv(raw_dir / "mimic-cxr-2.0.0-split.csv.gz")
    metadata_df = pd.read_csv(raw_dir / "mimic-cxr-2.0.0-metadata.csv.gz")
    print(f"chexpert={chexpert_df.shape} split={split_df.shape} metadata={metadata_df.shape}")

    print("\n=== clean chexpert ===")
    chexpert_clean, label_cols = clean_chexpert(chexpert_df)
    del chexpert_df

    print("\n=== clean metadata ===")
    metadata_clean = clean_metadata(metadata_df, args.views)
    del metadata_df

    print("\n=== clean split ===")
    split_clean = clean_split(split_df)
    del split_df

    print("\n=== merge ===")
    # metadata carries many DICOM columns we do not need downstream; keep it narrow
    meta_cols = ["dicom_id", "subject_id", "study_id", "ViewPosition"]
    merged = split_clean.merge(metadata_clean[meta_cols],
                               on=["dicom_id", "subject_id", "study_id"], how="inner")
    print(f"after split+metadata: {merged.shape}")
    merged = merged.merge(chexpert_clean[["subject_id", "study_id", "has_chexpert_label"]],
                          on=["subject_id", "study_id"], how="inner")
    print(f"after chexpert: {merged.shape}")
    print(f"unique studies: {merged['study_id'].nunique()} (paper reports 227,835)")
    del metadata_clean, split_clean, chexpert_clean

    print("\n=== reports ===")
    studies = merged[["subject_id", "study_id"]].drop_duplicates().reset_index(drop=True)
    if args.limit_studies:
        studies = studies.head(args.limit_studies)
        print(f"[debug] limited to {len(studies)} studies")
    study_text = build_study_text(studies, reports_root, args.workers)

    merged = merged.merge(study_text[["subject_id", "study_id", "findings_clean",
                                      "impression_clean"]],
                          on=["subject_id", "study_id"], how="inner")
    print(f"after attaching report text: {merged.shape}")
    del study_text

    print("\n=== length filter ===")
    merged["findings_word_count"] = merged["findings_clean"].str.split().str.len().fillna(0).astype(int)
    print(merged["findings_word_count"].describe().to_string())
    before = merged.shape[0]
    merged = merged[merged["findings_word_count"] >= args.min_words]
    upper = merged["findings_word_count"].quantile(args.upper_quantile)
    merged = merged[merged["findings_word_count"] <= upper].reset_index(drop=True)
    print(f"{before} -> {merged.shape[0]} rows "
          f"(min_words={args.min_words}, upper q{args.upper_quantile}={upper:.0f})")

    print("\n=== image_path ===")
    merged["image_path"] = [
        build_image_path(s, st, d)
        for s, st, d in zip(merged["subject_id"], merged["study_id"], merged["dicom_id"])
    ]
    print(f"example: {merged['image_path'].iloc[0]}")

    print("\n=== save ===")
    final_cols = ["subject_id", "study_id", "dicom_id", "split", "ViewPosition",
                  "image_path", "findings_clean", "impression_clean",
                  "findings_word_count", "has_chexpert_label"]
    processed = merged[final_cols]

    for split_name, out_name in SPLIT_TO_FILENAME.items():
        subset = processed[processed["split"] == split_name]
        subset.to_csv(out_dir / f"{out_name}.csv", index=False)
        subset.to_parquet(out_dir / f"{out_name}.parquet", index=False)
        print(f"{split_name:9s} -> {out_name}.csv  rows={len(subset):>7d}  "
              f"studies={subset['study_id'].nunique():>7d}")

    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
