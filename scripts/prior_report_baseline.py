#!/usr/bin/env python3
"""Copy-the-prior-report baseline: what is the patient's previous FINDINGS worth?

MIMIC-CXR findings are written against the previous study ("unchanged",
"compared with ..."), and a single-image model never sees it. For every test
study of a cached Stage-2 record file, find the same patient's most recent
EARLIER study (metadata ``StudyDate``/``StudyTime``) with valid FINDINGS and use
that text as the prediction. Writes, into a private ``--out-dir``:

* ``prior_subset.jsonl``          -- prior copy, only the studies that have one;
* ``<name>_subset.jsonl``         -- the ``--fallback`` model's predictions on the
                                     same studies (for a paired comparison);
* ``prior_or_<name>.jsonl``       -- every study: prior copy if any, else the model.

Report text -- private, training host only. Prints counts only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test-cache", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True, help="split CSV of the test split")
    ap.add_argument("--metadata", type=Path, required=True, help="mimic-cxr-2.0.0-metadata.csv.gz")
    ap.add_argument("--fallback", required=True, help="name=<predictions.jsonl>")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)

    import torch

    records = torch.load(args.test_cache, map_location="cpu", weights_only=False, mmap=True)["records"]
    df = pd.read_csv(args.manifest, usecols=["subject_id", "study_id", "image_path",
                                             "findings_clean", "target_valid"])
    meta = pd.read_csv(args.metadata, usecols=["study_id", "StudyDate", "StudyTime"])
    when = (meta.assign(t=meta["StudyDate"].astype("int64") * 1_000_000
                        + meta["StudyTime"].astype(float).astype("int64"))
            .groupby("study_id")["t"].min())
    df["_key"] = df["image_path"].map(lambda v: str(v).rsplit("/", 1)[-1])
    df["t"] = df["study_id"].map(when)
    key_to_study = df.drop_duplicates("_key").set_index("_key")[["subject_id", "study_id", "t"]]
    studies = (df[df["target_valid"].astype(str).str.lower().isin(("true", "1"))]
               .drop_duplicates("study_id")[["subject_id", "study_id", "t", "findings_clean"]])
    by_subject = {s: g.sort_values("t") for s, g in studies.groupby("subject_id")}

    name, path = args.fallback.split("=", 1)
    model = {}
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        model[d["sample_key"]] = d["pred"]

    args.out_dir.mkdir(parents=True, exist_ok=False)
    f_prior = open(args.out_dir / "prior_subset.jsonl", "w", encoding="utf-8")
    f_model = open(args.out_dir / f"{name}_subset.jsonl", "w", encoding="utf-8")
    f_mix = open(args.out_dir / f"prior_or_{name}.jsonl", "w", encoding="utf-8")
    covered = unmatched = 0
    for r in records:
        key = str(r["image_path"]).rsplit("/", 1)[-1]
        if key not in key_to_study.index:
            unmatched += 1
            prior = None
        else:
            subject, study, t = key_to_study.loc[key]
            g = by_subject.get(subject)
            prior = None
            if g is not None:
                earlier = g[(g["t"] < t) & (g["study_id"] != study)]
                if len(earlier):
                    prior = str(earlier.iloc[-1]["findings_clean"])
        pred_model = model[r["sample_key"]]
        if prior:
            covered += 1
            f_prior.write(json.dumps({"sample_key": r["sample_key"], "pred": prior, "ref": r["ref"]}) + "\n")
            f_model.write(json.dumps({"sample_key": r["sample_key"], "pred": pred_model, "ref": r["ref"]}) + "\n")
        f_mix.write(json.dumps({"sample_key": r["sample_key"], "pred": prior or pred_model,
                                "ref": r["ref"]}) + "\n")
    for f in (f_prior, f_model, f_mix):
        f.close()
    print(f"[prior] {covered}/{len(records)} test studies have an earlier study with FINDINGS "
          f"({covered / len(records):.1%}); {unmatched} records not found in the manifest",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
