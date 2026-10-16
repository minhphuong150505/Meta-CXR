"""Aggregate-only dataset statistics for the thesis. Writes counts, no identifiers, no text."""
import json, sys
import numpy as np, pandas as pd
ROOT = "/mnt/drive1tb/mimic-cxr-jpg-full"
PROC = f"{ROOT}/processed/full_allviews_v2"
OUT = sys.argv[1]
LABELS = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema",
          "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other",
          "Fracture", "Support Devices"]
VIEW_RANK = {"PA": 0, "AP": 1, "LATERAL": 2, "LL": 2}
res = {"labels": LABELS}

def q(s):
    s = pd.Series(s).dropna()
    if s.empty: return None
    return {"n": int(len(s)), "mean": float(s.mean()), "std": float(s.std()), "min": float(s.min()),
            "p5": float(s.quantile(.05)), "p25": float(s.quantile(.25)), "median": float(s.median()),
            "p75": float(s.quantile(.75)), "p95": float(s.quantile(.95)), "max": float(s.max())}

# ---------------- raw sources
meta = pd.read_csv(f"{ROOT}/mimic-cxr-2.0.0-metadata.csv.gz",
                   usecols=["dicom_id", "subject_id", "study_id", "ViewPosition", "Rows", "Columns"])
split = pd.read_csv(f"{ROOT}/mimic-cxr-2.0.0-split.csv.gz")
chex = pd.read_csv(f"{ROOT}/mimic-cxr-2.0.0-chexpert.csv.gz")
res["raw"] = {
    "metadata_images": int(len(meta)), "metadata_studies": int(meta[["subject_id", "study_id"]].drop_duplicates().shape[0]),
    "metadata_subjects": int(meta["subject_id"].nunique()),
    "split_rows": int(len(split)), "split_counts_images": {k: int(v) for k, v in split["split"].value_counts().items()},
    "chexpert_studies": int(len(chex)),
    "chexpert_all_blank": int(chex[LABELS].isna().all(axis=1).sum()),
    "images_below_100px": int(((meta["Rows"] < 100) | (meta["Columns"] < 100)).sum()),
    "view_counts_all_images": {str(k): int(v) for k, v in meta["ViewPosition"].fillna("UNKNOWN").replace("", "UNKNOWN").value_counts().items()},
    "image_rows": q(meta["Rows"]), "image_columns": q(meta["Columns"]),
    "chexpert_blank_fraction": float(chex[LABELS].isna().to_numpy().mean()),
}

# ---------------- processed manifests
cols = ["subject_id", "study_id", "dicom_id", "split", "ViewPosition", "extraction_method", "target_valid",
        "impression_valid", "target_filter_reason", "findings_token_count", "findings_word_count",
        "impression_token_count", "has_chexpert_label"]
frames = {s: pd.read_csv(f"{PROC}/{s}.csv", usecols=cols) for s in ("train", "val", "test")}
subjects = {s: set(f["subject_id"]) for s, f in frames.items()}
studies = {s: set(zip(f["subject_id"], f["study_id"])) for s, f in frames.items()}
res["leakage"] = {
    "subject_train_val": len(subjects["train"] & subjects["val"]), "subject_train_test": len(subjects["train"] & subjects["test"]),
    "subject_val_test": len(subjects["val"] & subjects["test"]),
    "study_train_val": len(studies["train"] & studies["val"]), "study_train_test": len(studies["train"] & studies["test"]),
    "study_val_test": len(studies["val"] & studies["test"]),
    "duplicate_dicom": int(pd.concat(frames.values())["dicom_id"].duplicated().sum()),
}
chex_idx = chex.set_index(["subject_id", "study_id"])
res["splits"] = {}
for s, f in frames.items():
    f = f.copy()
    f["V"] = f["ViewPosition"].fillna("UNKNOWN").astype(str).str.strip().str.upper()
    f["rank"] = f["V"].map(VIEW_RANK).fillna(3)
    g = f.groupby(["subject_id", "study_id"], sort=False)
    per_study = g.agg(n_images=("dicom_id", "size"), n_viewtypes=("rank", "nunique"),
                      anchor_rank=("rank", "min"), target_valid=("target_valid", "first"),
                      impression_valid=("impression_valid", "first"), reason=("target_filter_reason", "first"),
                      ftok=("findings_token_count", "first"), itok=("impression_token_count", "first"),
                      method=("extraction_method", "first"), has_label=("has_chexpert_label", "first"),
                      has_frontal=("rank", lambda r: bool((r <= 1).any())))
    anchor_names = {0: "PA", 1: "AP", 2: "LATERAL", 3: "UNKNOWN"}
    anchor_view = per_study["anchor_rank"].map(anchor_names)
    # aux exists when another view TYPE exists (repeated anchor projections do not count)
    has_aux = per_study["n_viewtypes"] > 1
    subj_studies = per_study.reset_index().groupby("subject_id").size()
    d = {
        "images": int(len(f)), "studies": int(len(per_study)), "subjects": int(f["subject_id"].nunique()),
        "view_counts_images": {k: int(v) for k, v in f["V"].value_counts().items()},
        "anchor_view_counts": {k: int(v) for k, v in anchor_view.value_counts().items()},
        "studies_multi_image": int((per_study["n_images"] > 1).sum()),
        "studies_with_aux_view": int(has_aux.sum()),
        "images_per_study": q(per_study["n_images"]), "images_per_study_max": int(per_study["n_images"].max()),
        "studies_per_subject": q(subj_studies),
        "studies_with_frontal": int(per_study["has_frontal"].sum()),
        "filter_reason": {k: int(v) for k, v in per_study["reason"].value_counts().items()},
        "impression_valid": int(per_study["impression_valid"].sum()),
        "findings_and_impression_valid": int((per_study["target_valid"] & per_study["impression_valid"]).sum()),
        "extraction_method": {str(k): int(v) for k, v in per_study["method"].value_counts().items()},
        "findings_tokens_valid": q(per_study.loc[per_study["target_valid"], "ftok"]),
        "impression_tokens_valid": q(per_study.loc[per_study["impression_valid"], "itok"]),
        "findings_valid_le15_tokens": int((per_study["target_valid"] & (per_study["ftok"] <= 15)).sum()),
        "has_chexpert_record": int(per_study["has_label"].sum()),
        "frontal_and_findings_valid": int((per_study["has_frontal"] & per_study["target_valid"]).sum()),
    }
    lab = chex_idx.reindex(per_study.index)[LABELS]
    d["studies_missing_chexpert_row"] = int(lab.isna().all(axis=1).sum())
    per = {}
    for L in LABELS:
        v = lab[L]
        per[L] = {"P": int((v == 1).sum()), "N": int((v == 0).sum()), "U": int((v == -1).sum()), "blank": int(v.isna().sum())}
    d["label_counts"] = per
    d["labels_per_study_nonblank"] = q(lab.notna().sum(axis=1))
    d["positives_per_study"] = q((lab == 1).sum(axis=1))
    d["blank_fraction"] = float(lab.isna().to_numpy().mean())
    d["studies_only_no_finding"] = int(((lab["No Finding"] == 1) & lab.drop(columns="No Finding").isna().all(axis=1)).sum())
    res["splits"][s] = d
json.dump(res, open(OUT, "w"), indent=1)
print("written", OUT)
