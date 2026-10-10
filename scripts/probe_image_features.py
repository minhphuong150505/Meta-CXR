#!/usr/bin/env python3
"""Same-protocol linear probe of several image representations against MHCAC.

Compares, on the SAME test studies with the SAME labels (D-018 rule), how much
three-class CheXpert information a linear read-out recovers from:

* ``softtok`` -- the mean of the 32 Q-Former soft tokens (cached Stage-2 records);
* ``npz:<name>`` -- an external embedding, e.g. MedGemma's own image features
  from ``scripts/embed_medgemma_images.py``;

and reports MHCAC's own predictions on those studies as the reference. The
probe is fit on the train subset present in every source, the L2 strength is
picked on a 10% holdout of TRAIN (the test split is scored once), so the
sources differ only in their features.

Prints aggregates only.

    python scripts/probe_image_features.py --train-cache <...train...pt> --test-cache <...test...pt> \\
        --manifest-dir <split dir> --chexpert <chexpert.csv.gz> \\
        --embedding medgemma=<train.npz>,<test.npz> --report <private>/probe.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (_REPO_ROOT, _REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-cache", type=Path, required=True)
    ap.add_argument("--test-cache", type=Path, required=True)
    ap.add_argument("--manifest-dir", type=Path, required=True)
    ap.add_argument("--chexpert", type=Path, required=True)
    ap.add_argument("--train-limit", type=int, default=60000)
    ap.add_argument("--embedding", action="append", default=[])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=16)
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args(argv)

    import pandas as pd
    import torch
    from probe_soft_tokens_cached import auroc_table, fit_probe, labels_for, summarise
    from retrieval_report_baselines import load

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chexpert = pd.read_csv(args.chexpert)
    _tr, tr_pool, _l, tr_keys = load(args.train_cache, args.train_limit, args.seed)
    _te, te_pool, te_logits, te_keys = load(args.test_cache, None, args.seed)
    y_tr = labels_for(tr_keys, args.manifest_dir / "train.csv", chexpert)
    y_te = labels_for(te_keys, args.manifest_dir / "test.csv", chexpert)

    sources = {"softtok": (dict(zip(tr_keys, tr_pool)), dict(zip(te_keys, te_pool)))}
    for spec in args.embedding:
        name, files = spec.split("=", 1)
        f_tr, f_te = files.split(",")
        a, b = np.load(f_tr), np.load(f_te)
        sources[f"npz_{name}"] = (dict(zip(a["keys"].tolist(), a["emb"].astype(np.float32))),
                                  dict(zip(b["keys"].tolist(), b["emb"].astype(np.float32))))
    common_tr = [i for i, k in enumerate(tr_keys)
                 if (y_tr[i] >= 0).any() and all(k in s[0] for s in sources.values())]
    common_te = [i for i, k in enumerate(te_keys)
                 if (y_te[i] >= 0).any() and all(k in s[1] for s in sources.values())]
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(common_tr))
    hold = set(perm[: len(perm) // 10].tolist())
    fit_idx = [common_tr[j] for j in range(len(common_tr)) if j not in hold]
    sel_idx = [common_tr[j] for j in range(len(common_tr)) if j in hold]
    print(f"[probe] train fit {len(fit_idx)}, train holdout {len(sel_idx)}, test {len(common_te)}",
          flush=True)

    yt = y_te[common_te]
    probs = np.exp(te_logits[common_te] - te_logits[common_te].max(-1, keepdims=True))
    probs = probs / probs.sum(-1, keepdims=True)
    result = {"mhcac": summarise(auroc_table(probs, yt)), "n_test": len(common_te),
              "n_train_fit": len(fit_idx)}
    print(f"[probe] MHCAC: auroc_mean {result['mhcac']['mean']:.4f} "
          f"(neg {result['mhcac']['negative']:.4f} pos {result['mhcac']['positive']:.4f} "
          f"unc {result['mhcac']['uncertain']:.4f})", flush=True)
    per_finding = {"mhcac": auroc_table(probs, yt)}
    for name, (dtr, dte) in sources.items():
        def feats(idx, d, keys):
            return torch.as_tensor(np.stack([d[keys[i]] for i in idx]), dtype=torch.float16)
        val, l2, te_probs = fit_probe(feats(fit_idx, dtr, tr_keys), y_tr[fit_idx],
                                      feats(sel_idx, dtr, tr_keys), y_tr[sel_idx],
                                      feats(common_te, dte, te_keys), device,
                                      epochs=args.epochs, seed=args.seed)
        table = auroc_table(te_probs, yt)
        per_finding[name] = table
        s = summarise(table)
        result[name] = {"l2": l2, "holdout_auroc_mean": val, "test": s}
        print(f"[probe] {name}: test auroc_mean {s['mean']:.4f} (neg {s['negative']:.4f} "
              f"pos {s['positive']:.4f} unc {s['uncertain']:.4f}), l2 {l2:g}", flush=True)
    names = list(per_finding)
    print("\n[probe] positive-class AUROC per finding: " + " / ".join(names))
    for f in per_finding["mhcac"]:
        cells = [per_finding[n].get(f, {}).get("positive") for n in names]
        if all(c is not None for c in cells):
            print(f"    {f:<28} " + " / ".join(f"{c:.4f}" for c in cells))
    result["per_finding"] = per_finding
    if args.report:
        args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
