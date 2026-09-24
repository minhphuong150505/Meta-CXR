"""Count Stage-1 CheXpert labels under each blank-label policy; derive class weights.

CPU only, no images. Reads the CheXpert export and the processed split CSVs,
applies exactly the mapping ReportDataset applies
(``model/lavis/data/chexpert_labels.py``), and prints, per split and per policy:

* the number of 0 / 1 / 2 / IGNORE_LABEL cells per label, at STUDY level;
* how many studies are ``classification_valid``;
* where every IGNORE_LABEL cell under ``negative`` comes from -- it must be only
  studies with no CheXpert information plus the ``excluded_labels`` columns.

For the train split it also prints the classification weights for the
``negative`` policy with the formula mimic_cxr_full.yaml already uses:
``[1.0, n_neg/n_pos, n_neg/n_unc]``, kappa 1, each capped at 10.

Counts only -- nothing here prints an identifier or report text, so the output
is safe to paste into a handoff.

Example (on the training host)::

    python scripts/count_chexpert_blank_policy.py \\
        --chexpert-csv <data>/mimic-cxr-2.0.0-chexpert.csv.gz \\
        --split-dir <data>/processed/full_allviews_v2 \\
        --cfg-path pretraining/configs/mimic_cxr_full.yaml \\
        --json-out /home/phuong/blank_policy_counts.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]

CHEXPERT_COLS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
    "Support Devices",
]
SPLITS = ("train", "val", "test")
CAP = 10.0


def _load_labels_module():
    # By file path: importing through model.lavis would run model/lavis/__init__.py
    # and load the whole training stack for a counting script.
    path = _REPO / "model/lavis/data/chexpert_labels.py"
    spec = importlib.util.spec_from_file_location("_chexpert_labels", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _excluded_from_cfg(cfg_path: Path) -> list[str]:
    import yaml

    cfg = yaml.safe_load(cfg_path.read_text())
    mhcac = (cfg.get("model", {}) or {}).get("mhcac", {}) or {}
    return list(mhcac.get("excluded_labels", ["No Finding"]) or [])


def _coerce_bool(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.to_numpy()
    lowered = series.astype(str).str.strip().str.lower()
    mapping = {"true": True, "1": True, "false": False, "0": False}
    unknown = sorted(set(lowered) - set(mapping))
    if unknown:
        raise ValueError(f"has_chexpert_label holds non-boolean values: {unknown}")
    return lowered.map(mapping).to_numpy(dtype=bool)


def _study_rows(split_csv: Path) -> pd.DataFrame:
    frame = pd.read_csv(split_csv, usecols=lambda c: c in {
        "subject_id", "study_id", "has_chexpert_label"})
    frame["subject_id"] = frame["subject_id"].astype(int)
    frame["study_id"] = frame["study_id"].astype(int)
    return frame.drop_duplicates(["subject_id", "study_id"]).reset_index(drop=True)


def count_split(labels_mod, chexpert, studies, policy, excluded):
    prepared = labels_mod.prepare_chexpert_labels(
        chexpert, CHEXPERT_COLS, blank_policy=policy, excluded_labels=excluded
    )
    processed = (
        _coerce_bool(studies["has_chexpert_label"])
        if "has_chexpert_label" in studies else None
    )
    merged = labels_mod.attach_chexpert_labels(
        studies[["subject_id", "study_id"]], prepared, CHEXPERT_COLS,
        processed_has_label=processed,
    )
    ignore = labels_mod.IGNORE_LABEL
    per_label = {}
    for col in CHEXPERT_COLS:
        values = merged[col].to_numpy()
        per_label[col] = {
            "neg": int((values == 0).sum()),
            "pos": int((values == 1).sum()),
            "unc": int((values == 2).sum()),
            "ignore": int((values == ignore).sum()),
            "blank_raw": int(
                (merged["_chexpert_merge"].eq("both")
                 & merged["_has_chexpert_label_raw"].eq(True)
                 & merged[labels_mod.mention_column(col)].eq(0)).sum()
            ),
        }
    no_info = ~(
        merged["_chexpert_merge"].eq("both")
        & merged["_has_chexpert_label_raw"].eq(True)
    )
    return merged, per_label, no_info


def ignore_provenance(merged, per_label, no_info, excluded) -> dict:
    """Every IGNORE cell must be a no-information study or an excluded column."""
    kept = [c for c in CHEXPERT_COLS if c not in excluded]
    n_no_info = int(no_info.sum())
    expected = n_no_info * len(kept) + len(merged) * (len(CHEXPERT_COLS) - len(kept))
    observed = sum(v["ignore"] for v in per_label.values())
    stray = 0
    for col in kept:
        stray += int(((merged[col] < 0) & ~no_info).sum())
    return {
        "studies": int(len(merged)),
        "no_chexpert_record": int((~merged["_chexpert_merge"].eq("both")).sum()),
        "record_all_blank": int(
            (merged["_chexpert_merge"].eq("both")
             & ~merged["_has_chexpert_label_raw"].eq(True)).sum()
        ),
        "ignore_cells_observed": observed,
        "ignore_cells_expected": expected,
        "ignore_cells_in_kept_columns_of_informative_studies": stray,
        "ok": observed == expected and stray == 0,
    }


def class_weights(per_label) -> list[dict]:
    rows = []
    for col in CHEXPERT_COLS:
        c = per_label[col]
        raw_pos = c["neg"] / c["pos"] if c["pos"] else None
        raw_unc = c["neg"] / c["unc"] if c["unc"] else None
        rows.append({
            "label": col,
            "w_pos": round(min(raw_pos, CAP), 3) if raw_pos is not None else 1.0,
            "w_unc": round(min(raw_unc, CAP), 3) if raw_unc is not None else 1.0,
            "raw_pos": raw_pos,
            "raw_unc": raw_unc,
        })
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--chexpert-csv", required=True)
    p.add_argument("--split-dir", required=True,
                   help="directory holding the processed train.csv / val.csv / test.csv")
    p.add_argument("--cfg-path", default=str(_REPO / "pretraining/configs/mimic_cxr_full.yaml"),
                   help="read model.mhcac.excluded_labels from here")
    p.add_argument("--json-out", default=None, help="also write every count as JSON")
    args = p.parse_args()

    labels_mod = _load_labels_module()
    excluded = _excluded_from_cfg(Path(args.cfg_path))
    chexpert = pd.read_csv(args.chexpert_csv)
    chexpert["subject_id"] = chexpert["subject_id"].astype(int)
    chexpert["study_id"] = chexpert["study_id"].astype(int)
    print(f"excluded_labels (from {args.cfg_path}): {excluded}")

    report: dict = {"excluded_labels": excluded, "splits": {}}
    for split in SPLITS:
        studies = _study_rows(Path(args.split_dir) / f"{split}.csv")
        report["splits"][split] = {}
        for policy in labels_mod.BLANK_LABEL_POLICIES:
            merged, per_label, no_info = count_split(
                labels_mod, chexpert, studies, policy, excluded
            )
            entry = {
                "classification_valid": int(merged["classification_valid"].sum()),
                "mention_valid": int(merged["mention_valid"].sum()),
                "per_label": per_label,
                "ignore_provenance": ignore_provenance(merged, per_label, no_info, excluded),
            }
            report["splits"][split][policy] = entry
            print(f"\n=== {split} / blank_label_policy={policy} ===")
            print(f"studies {len(merged):,}  classification_valid "
                  f"{entry['classification_valid']:,}  mention_valid {entry['mention_valid']:,}")
            print(f"{'label':28s} {'neg(0)':>9s} {'pos(1)':>9s} {'unc(2)':>9s} "
                  f"{'-100':>9s} {'blank(raw)':>11s}")
            for col, c in per_label.items():
                print(f"{col:28s} {c['neg']:9,d} {c['pos']:9,d} {c['unc']:9,d} "
                      f"{c['ignore']:9,d} {c['blank_raw']:11,d}")
            prov = entry["ignore_provenance"]
            if policy == labels_mod.BLANK_AS_NEGATIVE:
                print(f"-100 provenance: {prov}")
            else:
                # Under `ignore` every blank is -100 by design, so the check
                # that -100 comes only from no-information studies does not apply.
                print(f"-100 provenance: n/a under {policy} "
                      f"(no_chexpert_record {prov['no_chexpert_record']}, "
                      f"record_all_blank {prov['record_all_blank']})")

    train_neg = report["splits"]["train"][labels_mod.BLANK_AS_NEGATIVE]["per_label"]
    weights = class_weights(train_neg)
    report["class_weights_negative"] = weights
    print("\n=== class_weights for blank_label_policy=negative (train, study level) ===")
    print("[w_negative, w_positive, w_uncertain] = [1.0, n_neg/n_pos, n_neg/n_unc], "
          f"kappa 1, cap {CAP:g}")
    for row in weights:
        c = train_neg[row["label"]]
        capped = []
        if row["raw_pos"] is not None and row["raw_pos"] > CAP:
            capped.append(f"pos raw {row['raw_pos']:.2f} CAPPED")
        if row["raw_unc"] is not None and row["raw_unc"] > CAP:
            capped.append(f"unc raw {row['raw_unc']:.2f} CAPPED")
        if row["raw_pos"] is None:
            capped.append("NO POSITIVES")
        note = ("  " + "; ".join(capped)) if capped else ""
        print(f"  - [1.0, {row['w_pos']:6.3f}, {row['w_unc']:6.3f}]  # {row['label']:27s}"
              f" pos {c['pos']:,} / neg {c['neg']:,} / unc {c['unc']:,}{note}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2, default=float))
        print(f"\nwrote {args.json_out}")

    bad = [
        (split, policy)
        for split, policies in report["splits"].items()
        for policy, entry in policies.items()
        if policy == labels_mod.BLANK_AS_NEGATIVE and not entry["ignore_provenance"]["ok"]
    ]
    if bad:
        raise SystemExit(f"IGNORE_LABEL provenance check FAILED for {bad}")


if __name__ == "__main__":
    main()
