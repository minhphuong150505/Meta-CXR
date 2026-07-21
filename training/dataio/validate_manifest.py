#!/usr/bin/env python3
"""Standalone fail-fast validation of the preprocessed split manifests.

The build-time leakage check inside ``preporcessing/preprocess_mimic_cxr.py``
only runs when the splits are generated. This script re-validates the CSVs that
training actually consumes, so a stale or hand-edited manifest cannot reach a
training run unnoticed.

    python -m training.dataio.validate_manifest --section-mode findings_and_impression

Exits non-zero on any leakage, missing column, or empty split.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

try:
    from dataio.manifest import (
        DEFAULT_SECTION_MODE,
        SECTION_MODES,
        ManifestError,
        assert_columns,
        assert_no_leakage,
        row_target,
        select_anchor_rows,
    )
except ImportError:  # ``python -m training.dataio.validate_manifest``
    from training.dataio.manifest import (
        DEFAULT_SECTION_MODE,
        SECTION_MODES,
        ManifestError,
        assert_columns,
        assert_no_leakage,
        row_target,
        select_anchor_rows,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path)
    parser.add_argument("--val-csv", type=Path)
    parser.add_argument("--test-csv", type=Path)
    parser.add_argument(
        "--section-mode", choices=SECTION_MODES, default=DEFAULT_SECTION_MODE
    )
    parser.add_argument(
        "--vis-root",
        type=Path,
        help="If given, sample-check that image files resolve under this root.",
    )
    parser.add_argument(
        "--image-sample",
        type=int,
        default=200,
        help="Rows per split to stat when --vis-root is given (0 checks all).",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    if args.train_csv and args.val_csv and args.test_csv:
        return {"train": args.train_csv, "val": args.val_csv, "test": args.test_csv}
    # Fall back to the configured paths only when the caller did not pass any.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from local_config import (  # noqa: E402
        PROCESSED_TEST_CSV,
        PROCESSED_TRAIN_CSV,
        PROCESSED_VAL_CSV,
    )

    return {
        "train": Path(args.train_csv or PROCESSED_TRAIN_CSV),
        "val": Path(args.val_csv or PROCESSED_VAL_CSV),
        "test": Path(args.test_csv or PROCESSED_TEST_CSV),
    }


def main() -> int:
    args = parse_args()
    paths = resolve_paths(args)
    frames: dict[str, pd.DataFrame] = {}
    problems: list[str] = []

    for split, path in paths.items():
        if str(path).startswith("gs://"):
            problems.append(
                f"{split}: {path} is a gs:// URI; download it locally before validating"
            )
            continue
        if not Path(path).is_file():
            problems.append(f"{split}: manifest not found at {path}")
            continue
        frames[split] = pd.read_csv(path)
        print(f"[read] {split:5s} rows={len(frames[split]):>7d}  {path}")

    if problems:
        for problem in problems:
            print(f"[FAIL] {problem}", file=sys.stderr)
        return 1

    for split, frame in frames.items():
        try:
            assert_columns(frame, args.section_mode, f"{split} manifest")
        except ManifestError as error:
            problems.append(str(error))

    if problems:
        for problem in problems:
            print(f"[FAIL] {problem}", file=sys.stderr)
        return 1

    try:
        assert_no_leakage(frames)
        print("[ok] no subject_id / study_id / dicom_id overlap across splits")
    except ManifestError as error:
        problems.append(str(error))

    for split, frame in frames.items():
        anchors = select_anchor_rows(frame)
        usable = sum(1 for row in anchors.to_dict("records") if row_target(row, args.section_mode))
        print(
            f"[{split:5s}] studies={len(anchors):>7d}  "
            f"usable_for_{args.section_mode}={usable:>7d}"
        )
        if usable == 0:
            problems.append(
                f"{split}: no rows satisfy section mode {args.section_mode!r}"
            )
        if args.vis_root:
            sample = anchors if args.image_sample <= 0 else anchors.head(args.image_sample)
            missing = [
                rel
                for rel in sample["image_path"].astype(str)
                if not (Path(args.vis_root) / rel).is_file()
            ]
            if missing:
                problems.append(
                    f"{split}: {len(missing)}/{len(sample)} sampled images not found "
                    f"under {args.vis_root} (e.g. {missing[0]})"
                )
            else:
                print(f"[{split:5s}] all {len(sample)} sampled images resolve")

    if problems:
        for problem in problems:
            print(f"[FAIL] {problem}", file=sys.stderr)
        return 1
    print("[PASS] manifests are split-disjoint and serve the requested section mode")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
