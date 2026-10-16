"""Aggregate statistics for one processed split.

Emits counts only -- never subject_id, study_id, dicom_id, image paths or report
text -- so the output is safe to keep alongside the code. The inputs are not:
this must run on the host that mounts the private data.

Run it for ``test`` first and diff against the committed test_split_stats.json.
If the script cannot reproduce a split whose numbers are already known, its
numbers for the other splits are not trustworthy either.

    python split_stats.py --split val --output-dir <dir>
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Order matters: it is the column order of the label CSV and the label_index
# used by the bbox mask cache. Must stay identical to chexpert_cols in
# blip2_qformer.py.
CHEXPERT_LABELS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]


def token_stats(series: pd.Series) -> dict:
    """Length distribution over studies that actually have the section.

    Zero-token rows are studies where the section is absent, not short. Keeping
    them drags p10 to 0 and describes coverage instead of length -- coverage is
    already reported separately as target_valid / impression_valid.
    """
    values = series.dropna()
    values = values[values > 0]
    if values.empty:
        return {"n": 0}
    return {
        "n": int(values.shape[0]),
        "median": float(values.median()),
        "p10": float(values.quantile(0.10)),
        "p90": float(values.quantile(0.90)),
        "max": float(values.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=("train", "val", "test"))
    parser.add_argument(
        "--processed-dir",
        default="/mnt/drive1tb/mimic-cxr-jpg-full/processed/full_allviews_v2",
    )
    parser.add_argument(
        "--chexpert-csv",
        default="/mnt/drive1tb/mimic-cxr-jpg-full/mimic-cxr-2.0.0-chexpert.csv.gz",
    )
    parser.add_argument("--mask-cache-dir", default="/home/phuong/explanation_masks_v2")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    split = args.split
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(f"{args.processed_dir}/{split}.csv")
    # One row per study for everything report- and label-shaped; the image rows
    # of a study all carry the same report and the same CheXpert record.
    studies = df.drop_duplicates("study_id")

    views = df["ViewPosition"].fillna("UNKNOWN").replace("", "UNKNOWN")
    per_study = df.groupby("study_id").size()

    stats = {
        "scale": {
            "images": int(len(df)),
            "studies": int(df["study_id"].nunique()),
            "patients": int(df["subject_id"].nunique()),
        },
        "views": {k: int(v) for k, v in views.value_counts().items()},
        "images_per_study": {
            str(k): int(v) for k, v in per_study.value_counts().sort_index().items()
        },
        "text": {
            "target_valid": int(studies["target_valid"].sum()),
            "impression_valid": int(studies["impression_valid"].sum()),
            "has_chexpert_label": int(studies["has_chexpert_label"].sum()),
            "findings_token_count": token_stats(studies["findings_token_count"]),
            "impression_token_count": token_stats(studies["impression_token_count"]),
        },
        "extraction_method": {
            k: int(v) for k, v in studies["extraction_method"].value_counts().items()
        },
    }

    # ---- label matrix, at study level -------------------------------------
    chex = pd.read_csv(args.chexpert_csv)
    merged = studies[["subject_id", "study_id"]].merge(
        chex, on=["subject_id", "study_id"], how="left"
    )
    n_studies = len(merged)
    rows = []
    totals = {"positive": 0, "negative": 0, "uncertain": 0, "blank": 0}
    for label in CHEXPERT_LABELS:
        col = merged[label] if label in merged.columns else pd.Series([np.nan] * n_studies)
        pos = int((col == 1.0).sum())
        neg = int((col == 0.0).sum())
        unc = int((col == -1.0).sum())
        blank = int(n_studies - pos - neg - unc)
        mentioned = pos + neg + unc
        rows.append(
            {
                "pathology": label,
                "positive": pos,
                "negative": neg,
                "uncertain": unc,
                "blank": blank,
                "mentioned": mentioned,
                "mentioned_pct": round(100.0 * mentioned / n_studies, 2),
                # Positive share among the cells that state a polarity. Blank is
                # an absence of evidence, not a negative, so it is excluded --
                # including it would understate every prevalence by ~4x here.
                "prevalence_P_over_PN": (
                    round(pos / (pos + neg), 4) if (pos + neg) else None
                ),
            }
        )
        totals["positive"] += pos
        totals["negative"] += neg
        totals["uncertain"] += unc
        totals["blank"] += blank

    label_df = pd.DataFrame(rows)
    label_df.to_csv(out_dir / f"{split}_label_distribution.csv", index=False)

    cells = n_studies * len(CHEXPERT_LABELS)
    stats["label_matrix_totals"] = {
        "cells": cells,
        "mentioned": cells - totals["blank"],
        "blank": totals["blank"],
        "blank_pct": round(100.0 * totals["blank"] / cells, 2),
        "labels_per_study": round((cells - totals["blank"]) / n_studies, 2),
        "positive": totals["positive"],
        "negative": totals["negative"],
        "uncertain": totals["uncertain"],
    }

    # ---- explanation-mask coverage ----------------------------------------
    cache = Path(args.mask_cache_dir)
    pooled_path = cache / f"index_{split}.json"
    bbox_path = cache / f"index_bbox_{split}.json"
    if pooled_path.exists():
        pooled = json.loads(pooled_path.read_text())
        bbox = json.loads(bbox_path.read_text()) if bbox_path.exists() else {}
        # mask_source 0 = CheXmask lung union, 1 = MS-CXR bbox union.
        sources = [int(v.get("mask_source", 0)) for v in pooled.values()]
        stats["explanation_masks"] = {
            "studies_with_mask": len(pooled),
            "lung_chexmask": sum(1 for s in sources if s == 0),
            "bbox_ms_cxr": sum(1 for s in sources if s == 1),
            "bbox_label_pairs": len(bbox),
            "bbox_studies": len({k.rsplit(":", 1)[0] for k in bbox}),
        }
    else:
        stats["explanation_masks"] = {"note": f"no mask cache at {pooled_path}"}

    stats["provenance"] = {
        "manifest": f"processed/full_allviews_v2/{split}.csv",
        "chexpert": Path(args.chexpert_csv).name,
        "mask_cache": str(cache),
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "note": (
            "Aggregate counts only. No subject_id / study_id / dicom_id / "
            "report text."
        ),
    }

    out_json = out_dir / f"{split}_split_stats.json"
    out_json.write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))
    print(f"\nwrote {out_json}")
    print(f"wrote {out_dir / f'{split}_label_distribution.csv'}")


if __name__ == "__main__":
    main()
