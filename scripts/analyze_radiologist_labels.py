#!/usr/bin/env python3
"""How noisy are the CheXpert-labeler labels, and how does Stage 1 score against a radiologist?

MIMIC-CXR-JPG 2.1.0 ships ``mimic-cxr-2.1.0-test-set-labeled.csv``: 687 test
studies labelled manually (14 findings; "Airspace Opacity" is the labeler's
"Lung Opacity"). Every Stage-1 number in this repo is scored against the
automatic CheXpert-labeler output, which is also what the model is trained on.
This script measures, on those 687 studies:

1. labeler vs radiologist, per finding, three classes under the Stage-1 rule
   (blank = Negative, D-018): agreement and Cohen's kappa, plus
   positive-vs-rest precision / recall of the labeler;
2. Stage-1 predictions (one or more ``.npz``) scored against BOTH label sets on
   the same studies: one-vs-rest AUROC per class and the paper's weighted F1
   (argmax), with a paired bootstrap of (radiologist - labeler).

The ``.npz`` ``sample_keys`` are salted digests of the batch's ``image_id``,
which for MIMIC is the anchor image's row position in the split annotation
(``ReportDataset``), not the study id. ``--stage1-config`` rebuilds the test
dataset (CSV only, no image is read) to map each key to its study. Aggregates only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.evaluation.classification_metrics import evaluate_classification  # noqa: E402
from training.evaluation.schemas import ClassificationPredictions, build_sample_keys  # noqa: E402

RENAME = {"Airspace Opacity": "Lung Opacity"}


def three_class(frame: pd.DataFrame, names) -> np.ndarray:
    """Stage-1 rule: 1 -> 1, 0 -> 0, -1 -> 2, blank -> 0 when the row has any value."""
    values = frame[list(names)].to_numpy(dtype=float)
    has_any = ~np.isnan(values).all(axis=1)
    out = np.full(values.shape, -1, dtype=np.int64)
    out[values == 1.0] = 1
    out[values == 0.0] = 0
    out[values == -1.0] = 2
    out[np.isnan(values) & has_any[:, None]] = 0
    return out


def kappa(a: np.ndarray, b: np.ndarray) -> float:
    classes = np.union1d(a, b)
    po = float((a == b).mean())
    pe = float(sum((a == c).mean() * (b == c).mean() for c in classes))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--radiologist", type=Path, required=True)
    ap.add_argument("--chexpert", type=Path, required=True)
    ap.add_argument("--predictions", action="append", default=[], help="NAME=test.npz")
    ap.add_argument("--stage1-config", type=Path, default=None,
                    help="YAML of the run that wrote the .npz; maps keys to studies")
    ap.add_argument("--split", default="test")
    ap.add_argument("--resamples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=16)
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args(argv)

    rad = pd.read_csv(args.radiologist).rename(columns=RENAME)
    lab = pd.read_csv(args.chexpert)
    preds = {spec.split("=", 1)[0]: ClassificationPredictions.load(spec.split("=", 1)[1])
             for spec in args.predictions}
    names = list(next(iter(preds.values())).pathology_names) if preds else [
        c for c in rad.columns if c != "study_id"]
    rad = rad.drop_duplicates("study_id").set_index("study_id")
    lab = lab.drop_duplicates("study_id").set_index("study_id")
    common = [s for s in rad.index if s in lab.index]
    y_rad = three_class(rad.loc[common], names)
    y_lab = three_class(lab.loc[common], names)
    report: dict = {"n_studies": len(common), "labeler_vs_radiologist": {}, "models": {}}
    print(f"[rad] {len(common)} radiologist-labelled studies with a labeler record", flush=True)
    print("[rad] finding: agreement kappa | labeler positive precision recall (radiologist = truth)")
    for j, name in enumerate(names):
        a, b = y_lab[:, j], y_rad[:, j]
        ok = (a >= 0) & (b >= 0)
        tp = int(((a == 1) & (b == 1) & ok).sum())
        prec = tp / max(int(((a == 1) & ok).sum()), 1)
        rec = tp / max(int(((b == 1) & ok).sum()), 1)
        row = {"agreement": float((a[ok] == b[ok]).mean()), "kappa": kappa(a[ok], b[ok]),
               "labeler_pos_precision": prec, "labeler_pos_recall": rec,
               "radiologist_positives": int(((b == 1) & ok).sum())}
        report["labeler_vs_radiologist"][name] = row
        print(f"    {name:<28} {row['agreement']:.3f} {row['kappa']:+.3f} | "
              f"{prec:.3f} {rec:.3f} (n_pos={row['radiologist_positives']})", flush=True)

    study_of_key: dict[str, int] = {}
    if preds and args.stage1_config is not None:
        from training.run_context import Stage1Context
        from training.stage1 import lavis_loader as L

        cfg = L.build_cfg(Stage1Context(run_name="rad", config_path=args.stage1_config))
        dataset = L.MIMIC_CXR_Dataset(vis_processor=None, text_processor=None,
                                      vis_root=L.VIS_ROOT, split=args.split, cfg=cfg)
        anchors = [str(entry["anchor"]) for entry in dataset.studies]
        for key, entry in zip(build_sample_keys(anchors).tolist(), dataset.studies):
            study_of_key[key] = int(entry["study_key"][1])
    keys = build_sample_keys([str(s) for s in common])
    rng = np.random.default_rng(args.seed)
    for model, p in preds.items():
        if study_of_key:
            by_study = {study_of_key[k]: i for i, k in enumerate(p.sample_keys.tolist())
                        if k in study_of_key}
            pos = {keys[i]: by_study[s] for i, s in enumerate(common) if s in by_study}
        else:
            pos = {k: i for i, k in enumerate(p.sample_keys.tolist())}
        hit = [i for i, k in enumerate(keys) if k in pos]
        if not hit:
            print(f"[rad] {model}: no study matched the prediction file", flush=True)
            continue
        rows = np.array([pos[keys[i]] for i in hit])
        probs = p.probabilities[rows]

        def score(labels, idx):
            sub = ClassificationPredictions(labels=labels[idx], probabilities=probs[idx],
                                            pathology_names=p.pathology_names,
                                            sample_keys=p.sample_keys[rows][idx])
            return evaluate_classification(sub).aggregates

        all_idx = np.arange(len(hit))
        full_r, full_l = score(y_rad[hit], all_idx), score(y_lab[hit], all_idx)
        metrics = ("auroc_mean", "auroc_positive_mean", "weighted_f1", "mean_weighted_f1_5",
                   "macro_recall")
        boot = {m: [] for m in metrics}
        for _ in range(args.resamples):
            idx = rng.integers(0, len(hit), len(hit))
            r, lb = score(y_rad[hit], idx), score(y_lab[hit], idx)
            for m in metrics:
                boot[m].append(r[m] - lb[m])
        entry = {"n": len(hit)}
        for metric in metrics:
            lo, hi = np.nanpercentile(boot[metric], [2.5, 97.5])
            entry[metric] = {"radiologist": full_r[metric], "labeler": full_l[metric],
                             "delta": full_r[metric] - full_l[metric], "ci95": [float(lo), float(hi)]}
            print(f"[rad] {model} n={len(hit)} {metric}: radiologist {full_r[metric]:.4f} "
                  f"labeler {full_l[metric]:.4f} delta {entry[metric]['delta']:+.4f} "
                  f"[{lo:+.4f}, {hi:+.4f}]", flush=True)
        report["models"][model] = entry
    if args.report:
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
