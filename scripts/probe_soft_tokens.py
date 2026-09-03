"""Ask whether the 32 Q-Former soft tokens carry medical signal at all.

WHY THIS EXISTS. Stage 1 ships `lambda_itc/itm/lm = 0.0`, so the Q-Former's
image path never received a gradient: cross-attention, query tokens and the
query FFN are bit-identical to the BLIP-2 initialisation, which was fitted to
EVA-ViT features on natural images. Here they read BioViL-T + PubMedCLIP tokens
on chest radiographs -- dimensionally compatible (both 1408) and semantically
foreign. So the soft tokens are a FIXED readout of well-trained Stage-1
features, and whether that readout preserved anything usable is an empirical
question, not something the architecture guarantees.

Arm C costs ~70 GPU-hours. This costs minutes and answers the question first.

WHAT IT MEASURES. A linear probe is the right instrument because `img_proj` --
the only trained layer between the soft tokens and MedGemma -- is itself a
single linear map. If a linear probe cannot recover the labels, `img_proj`
cannot either, and the extra channel is decoration.

  probe AUROC ~ 0.50   the readout collapsed; do NOT spend 70 h on arm C
  probe AUROC 0.60+    real signal survives; img_proj has something to learn
  probe AUROC ~ 0.76   near MHCAC's own test macro AUROC (0.7643)

Two shape diagnostics run beside it, because a degenerate readout has a
signature this repo has seen before: PubMedCLIP's raw patch tokens had a fixed
DC direction with mean pairwise cosine 0.674 (against BioViL's 0.0017) and
acted as a constant bias. The same failure here would be soft tokens that are
nearly identical ACROSS studies (no per-image information) or nearly identical
WITHIN a study (32 tokens doing the work of one).

PRIVACY. Prints aggregate numbers only -- never report text, never an
identifier, never an image path. Safe to paste into a summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))

CHEXPERT_14 = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity",
    "No Finding", "Pleural Effusion", "Pleural Other", "Pneumonia",
    "Pneumothorax", "Support Devices",
]
# Excluded from the macro exactly as the Stage-1 evaluator excludes them:
# they are meta labels, not findings.
META_LABELS = {"No Finding", "Support Devices"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True,
                    help="Split CSV (val.csv). Carries image_path and study_id, "
                         "but NOT the labels -- see --chexpert.")
    ap.add_argument("--chexpert", type=Path, required=True,
                    help="mimic-cxr-2.0.0-chexpert.csv.gz. The split CSVs hold "
                         "only has_chexpert_label, so the 14 columns come from "
                         "here and are joined on study_id.")
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=1500,
                    help="Studies to encode. 0 = the whole split.")
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="Scratch dir; the Stage-1 soft-token cache lands under it.")
    ap.add_argument("--checkpoint-root", default="checkpoints")
    ap.add_argument("--stage1-run", default="mimic_cxr_full_blip2")
    ap.add_argument("--stage1-config", type=Path, default=None)
    ap.add_argument("--stage1-checkpoint", type=Path, default=None)
    ap.add_argument("--threshold-path", type=Path, default=None)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--report", type=Path, default=None,
                    help="Write the aggregate result here as JSON.")
    args = ap.parse_args()

    import pandas as pd
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    import train_eval_figure9_llm_variants_200 as fig9
    from run_context import Stage1Context

    context = Stage1Context(
        run_name=args.stage1_run,
        config_path=args.stage1_config
        or (fig9.PROJECT_DIR / "pretraining/configs/mimic_cxr_full.yaml"),
        checkpoint_path=args.stage1_checkpoint,
        thresholds=fig9.load_thresholds(args.threshold_path),
    )
    fig9.set_seed(fig9.SEED)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[probe] encoding {args.limit or 'all'} {args.split} studies "
          f"through Stage 1 (batch 1 -- this is the slow part)", flush=True)
    records = fig9.build_stage1_records(
        context, Path(args.checkpoint_root), args.output_dir,
        args.split, args.limit or None, args.num_workers,
    )
    if not records:
        print("[probe] no records; nothing to probe", file=sys.stderr)
        return 1

    embs = torch.stack([r["qformer_embs"] for r in records]).float()  # [N, 32, 768]
    n, n_tok, dim = embs.shape
    print(f"[probe] soft tokens: {tuple(embs.shape)}", flush=True)

    # ---- shape diagnostics ------------------------------------------------
    # Both are cosines between L2-normalised vectors, so ~0 means "unrelated"
    # and ~1 means "the same vector wearing different hats".
    pooled = embs.mean(dim=1)                                   # [N, 768]
    pn = torch.nn.functional.normalize(pooled, dim=-1)
    # Off-diagonal mean of the N x N cosine matrix, computed without
    # materialising it for large N.
    sims = pn @ pn.T
    off = (sims.sum() - torch.diagonal(sims).sum()) / (n * (n - 1))
    across_studies = float(off)

    tn = torch.nn.functional.normalize(embs, dim=-1)             # [N, 32, 768]
    within = tn @ tn.transpose(1, 2)                             # [N, 32, 32]
    within_studies = float(
        (within.sum(dim=(1, 2)) - torch.diagonal(within, dim1=1, dim2=2).sum(dim=1))
        .div(n_tok * (n_tok - 1)).mean()
    )
    print(f"[probe] mean cosine ACROSS studies (pooled): {across_studies:+.4f}")
    print(f"[probe] mean cosine WITHIN a study (32 tok): {within_studies:+.4f}")
    print("[probe]   ~1.0 on either line means the readout is degenerate; "
          "PubMedCLIP's broken stream measured 0.674, healthy BioViL 0.0017")

    # ---- labels -----------------------------------------------------------
    # Two hops, because the split CSV deliberately carries no label columns:
    #   record.image_path -> val.csv -> study_id -> chexpert.csv.gz -> 14 labels
    df = pd.read_csv(args.manifest, usecols=["image_path", "study_id"])
    lab = pd.read_csv(args.chexpert)
    present = [c for c in CHEXPERT_14 if c in lab.columns]
    if len(present) != len(CHEXPERT_14):
        print(f"[probe] chexpert file has {len(present)}/14 label columns",
              file=sys.stderr)
        return 1
    merged = df.merge(lab.drop(columns=["subject_id"], errors="ignore"),
                      on="study_id", how="left")
    # The records carry an ABSOLUTE path (vis_root already joined on by the
    # dataset) while the CSV keeps the relative one, so a direct join returns
    # zero rows without erroring. Key on the file's basename instead: it is the
    # dicom_id, which is unique across MIMIC-CXR.
    merged["_key"] = merged["image_path"].map(lambda v: str(v).rsplit("/", 1)[-1])
    lut = merged.drop_duplicates("_key").set_index("_key")

    paths = [str(r.get("image_path", "")).rsplit("/", 1)[-1] for r in records]
    hit = [p in lut.index for p in paths]
    print(f"[probe] label join: {sum(hit)}/{len(paths)} studies matched")
    if sum(hit) < 100:
        print("[probe] too few joined rows to probe", file=sys.stderr)
        return 1

    keep = np.array(hit)
    X = pooled.numpy()[keep]
    sub = lut.loc[[p for p, h in zip(paths, hit) if h]]

    # `study_presence` framing: blank / negative / uncertain all mean "not
    # present". This is the framing the project reports F1 and AUROC under, and
    # the only one under which the number is comparable to MHCAC's 0.7643.
    results: dict[str, float] = {}
    shuffled: dict[str, float] = {}
    rng = np.random.default_rng(fig9.SEED)
    for label in present:
        y = (pd.to_numeric(sub[label], errors="coerce").fillna(0.0).to_numpy() == 1.0)
        y = y.astype(int)
        if y.sum() < 20 or (len(y) - y.sum()) < 20:
            continue  # too few of one class for a stable per-label AUROC
        for name, target in (("real", y), ("shuffled", rng.permutation(y))):
            oof = np.zeros(len(target), dtype=float)
            skf = StratifiedKFold(n_splits=args.folds, shuffle=True,
                                  random_state=fig9.SEED)
            for tr, te in skf.split(X, target):
                sc = StandardScaler().fit(X[tr])
                # Strong L2: 768 features against ~1.5k rows overfits without it,
                # and an overfitted probe reports the labels back to itself.
                clf = LogisticRegression(C=0.01, max_iter=2000, solver="lbfgs")
                clf.fit(sc.transform(X[tr]), target[tr])
                oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
            (results if name == "real" else shuffled)[label] = float(
                roc_auc_score(target, oof)
            )

    if not results:
        print("[probe] no label had enough of both classes", file=sys.stderr)
        return 1

    findings = {k: v for k, v in results.items() if k not in META_LABELS}
    macro = float(np.mean(list(findings.values()))) if findings else float("nan")
    macro_shuf = float(np.mean([shuffled[k] for k in findings])) if findings else float("nan")

    print("\n[probe] per-label AUROC (5-fold CV, study_presence framing)")
    for label in sorted(results, key=results.get, reverse=True):
        tag = "  (meta, excluded from macro)" if label in META_LABELS else ""
        print(f"    {label:<28} {results[label]:.4f}   "
              f"shuffled {shuffled[label]:.4f}{tag}")
    print(f"\n[probe] MACRO AUROC over {len(findings)} findings: {macro:.4f}")
    print(f"[probe] same probe on SHUFFLED labels:      {macro_shuf:.4f}  "
          f"(must be ~0.50, or the probe leaks)")
    print(f"[probe] MHCAC's own test macro AUROC:        0.7643  (reference)")

    verdict = ("GO -- soft tokens carry signal" if macro >= 0.60 else
               "MARGINAL -- weak signal, arm C is a gamble" if macro >= 0.55 else
               "NO-GO -- readout collapsed, do not spend 70 h")
    print(f"\n[probe] VERDICT: {verdict}")

    if args.report:
        args.report.write_text(json.dumps({
            "n_studies": int(keep.sum()),
            "n_encoded": n,
            "split": args.split,
            "stage1_run": args.stage1_run,
            "cosine_across_studies": across_studies,
            "cosine_within_study": within_studies,
            "auroc_per_label": results,
            "auroc_per_label_shuffled": shuffled,
            "macro_auroc_findings": macro,
            "macro_auroc_findings_shuffled": macro_shuf,
            "mhcac_reference_macro_auroc": 0.7643,
            "verdict": verdict,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[probe] wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
