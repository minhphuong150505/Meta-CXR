#!/usr/bin/env python3
"""Linear probe of the 32 Q-Former soft tokens, from cached Stage-2 records.

Question: do the soft tokens -- the only image channel the paper-mode Stage 2
reads -- still carry what Stage 1 knows? Compared against MHCAC's own
predictions on the SAME test studies with the SAME labels, so the gap is the
information lost between MHCAC's input features and the soft tokens (or never
put into them).

Unlike ``scripts/probe_soft_tokens.py`` (5-fold CV on ~1.5k val studies, mean
pooled, encodes through Stage 1), this reads the ``.sensitive_stage1_cache``
records a Stage-2 run already wrote (``qformer_embs`` [32, 768] and
``class_logits`` [14, 3] per study), fits on TRAIN, picks the L2 strength on
VAL, and scores TEST once. Two feature sets:

* ``pooled``: mean over the 32 tokens, 768-d;
* ``tokens``: all 32 tokens concatenated, 24,576-d -- the most a linear read
  of the soft tokens can see (``img_proj`` is linear per token).

Labels: the Stage-1 rule (D-018, ``blank_label_policy: negative``): blank ->
Negative, -1 -> Uncertain; a study with no CheXpert information is skipped.
AUROC is one-vs-rest per (finding, class), cells with < 20 positives or
negatives skipped, averaged like ``auroc_mean``.

Prints aggregates only -- no report text, no identifiers, no paths.

    python scripts/probe_soft_tokens_cached.py \\
        --cache-dir <stage2 run>/.sensitive_stage1_cache \\
        --manifest-dir <processed split dir> --chexpert <mimic-cxr-2.0.0-chexpert.csv.gz> \\
        --report <private dir>/soft_token_probe.json
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

#: The order of ``class_logits`` (``fig9.ABNORMALITIES_14``); copied so this
#: script does not import the Stage-2 stack.
FINDINGS = (
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices",
)
CLASS_NAMES = ("negative", "positive", "uncertain")
MIN_CELL = 20
L2_GRID = (1e-5, 1e-4, 1e-3, 1e-2)


def _cache(cache_dir: Path, split: str) -> Path:
    hits = sorted(glob.glob(str(cache_dir / f"*_{split}_qformer_all_*.pt")))
    if len(hits) != 1:
        raise SystemExit(f"expected one {split} cache in {cache_dir}, found {len(hits)}; "
                         f"pass --{split}-cache")
    return Path(hits[0])


def load_split(path: Path, limit: int | None, seed: int):
    import torch

    records = torch.load(path, map_location="cpu", weights_only=False, mmap=True)["records"]
    order = np.arange(len(records))
    if limit and limit < len(order):
        order = np.sort(np.random.default_rng(seed).choice(order, limit, replace=False))
    embs = torch.stack([records[i]["qformer_embs"] for i in order]).to(torch.float16)
    logits = torch.stack([records[i]["class_logits"] for i in order]).float()
    keys = [str(records[i]["image_path"]).rsplit("/", 1)[-1] for i in order]
    return embs, logits, keys


def labels_for(keys, manifest: Path, chexpert):
    import pandas as pd

    from model.lavis.data.chexpert_labels import BLANK_AS_NEGATIVE, prepare_chexpert_labels

    df = pd.read_csv(manifest, usecols=["image_path", "subject_id", "study_id"])
    df["_key"] = df["image_path"].map(lambda v: str(v).rsplit("/", 1)[-1])
    df = df.drop_duplicates("_key").set_index("_key")
    prepared = prepare_chexpert_labels(chexpert, list(FINDINGS), blank_policy=BLANK_AS_NEGATIVE)
    lut = prepared.set_index(["subject_id", "study_id"])[list(FINDINGS)]
    y = np.full((len(keys), len(FINDINGS)), -1, dtype=np.int64)
    for i, k in enumerate(keys):
        if k not in df.index:
            continue
        sid = (int(df.at[k, "subject_id"]), int(df.at[k, "study_id"]))
        if sid in lut.index:
            row = lut.loc[sid].to_numpy().astype(np.int64)
            y[i] = np.where(row >= 0, row, -1)
    return y


def auroc_table(probs: np.ndarray, y: np.ndarray) -> dict[str, dict[str, float]]:
    from sklearn.metrics import roc_auc_score

    out: dict[str, dict[str, float]] = {}
    for f, name in enumerate(FINDINGS):
        valid = y[:, f] >= 0
        yt, pf = y[valid, f], probs[valid, f]
        cells = {}
        for c, cls in enumerate(CLASS_NAMES):
            pos = int((yt == c).sum())
            if pos >= MIN_CELL and len(yt) - pos >= MIN_CELL:
                cells[cls] = float(roc_auc_score(yt == c, pf[:, c]))
        if cells:
            out[name] = cells
    return out


def summarise(table) -> dict[str, float]:
    per = {cls: float(np.mean([v[cls] for v in table.values() if cls in v]))
           for cls in CLASS_NAMES}
    per["mean"] = float(np.mean([x for v in table.values() for x in v.values()]))
    return per


def fit_probe(xtr, ytr, xva, yva, xte, device, *, epochs: int, seed: int):
    """Multinomial logistic regression per finding, 14 heads on shared features."""
    import torch

    torch.manual_seed(seed)
    mean = xtr.float().mean(0, keepdim=True)
    std = xtr.float().std(0, keepdim=True).clamp_min(1e-4)

    def prep(x):
        return ((x.float() - mean) / std).to(device)

    ytr_t = torch.as_tensor(ytr, device=device)
    best = None
    for l2 in L2_GRID:
        lin = torch.nn.Linear(xtr.shape[1], len(FINDINGS) * 3).to(device)
        opt = torch.optim.Adam(lin.parameters(), lr=1e-3)
        n = xtr.shape[0]
        for _ in range(epochs):
            perm = torch.randperm(n)
            for s in range(0, n, 1024):
                idx = perm[s:s + 1024]
                out = lin(prep(xtr[idx])).view(-1, len(FINDINGS), 3)
                loss = torch.nn.functional.cross_entropy(
                    out.reshape(-1, 3), ytr_t[idx].reshape(-1), ignore_index=-1)
                loss = loss + l2 * lin.weight.pow(2).sum()
                opt.zero_grad(); loss.backward(); opt.step()

        def predict(x):
            with torch.no_grad():
                chunks = [torch.softmax(lin(prep(x[s:s + 2048])).view(-1, len(FINDINGS), 3), -1)
                          for s in range(0, x.shape[0], 2048)]
            return torch.cat(chunks).cpu().numpy()

        val = summarise(auroc_table(predict(xva), yva))["mean"]
        print(f"[probe]   l2 {l2:g}: val auroc_mean {val:.4f}", flush=True)
        if best is None or val > best[0]:
            best = (val, l2, predict(xte))
    return best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--train-cache", type=Path)
    ap.add_argument("--val-cache", type=Path)
    ap.add_argument("--test-cache", type=Path)
    ap.add_argument("--manifest-dir", type=Path, required=True,
                    help="directory holding the split CSVs {train,val,test}.csv")
    ap.add_argument("--chexpert", type=Path, required=True)
    ap.add_argument("--train-limit", type=int, default=50000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=16)
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args(argv)

    import pandas as pd
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chexpert = pd.read_csv(args.chexpert)
    data = {}
    for split, limit in (("train", args.train_limit), ("val", None), ("test", None)):
        path = getattr(args, f"{split}_cache") or _cache(args.cache_dir, split)
        embs, logits, keys = load_split(path, limit, args.seed)
        y = labels_for(keys, args.manifest_dir / f"{split}.csv", chexpert)
        labelled = (y >= 0).any(1)
        print(f"[probe] {split}: {len(keys)} studies, {int(labelled.sum())} with CheXpert labels",
              flush=True)
        data[split] = (embs[labelled], logits[labelled], y[labelled])

    te_embs, te_logits, te_y = data["test"]
    n, n_tok, dim = te_embs.shape
    # Shape diagnostics on test: ~1 means the readout is degenerate.
    pn = torch.nn.functional.normalize(te_embs.float().mean(1), dim=-1)
    sims = pn @ pn.T
    across = float((sims.sum() - sims.diagonal().sum()) / (n * (n - 1)))
    tn = torch.nn.functional.normalize(te_embs.float(), dim=-1)
    w = tn @ tn.transpose(1, 2)
    within = float(((w.sum((1, 2)) - w.diagonal(dim1=1, dim2=2).sum(1)) / (n_tok * (n_tok - 1))).mean())
    print(f"[probe] cosine across studies {across:+.4f}, within a study {within:+.4f}", flush=True)

    mhcac = summarise(auroc_table(torch.softmax(te_logits, -1).numpy(), te_y))
    result = {"n_train": int(data["train"][2].shape[0]), "n_val": int(data["val"][2].shape[0]),
              "n_test": int(n), "cosine_across_studies": across, "cosine_within_study": within,
              "mhcac_on_same_test": mhcac, "probes": {}}
    print(f"[probe] MHCAC on the same test studies: auroc_mean {mhcac['mean']:.4f} "
          f"(neg {mhcac['negative']:.4f} pos {mhcac['positive']:.4f} unc {mhcac['uncertain']:.4f})",
          flush=True)

    def feats(split, kind):
        e = data[split][0]
        return e.float().mean(1).half() if kind == "pooled" else e.reshape(e.shape[0], -1)

    for kind in ("pooled", "tokens"):
        print(f"[probe] fitting {kind}", flush=True)
        val, l2, te_probs = fit_probe(
            feats("train", kind), data["train"][2], feats("val", kind), data["val"][2],
            feats("test", kind), device, epochs=args.epochs, seed=args.seed)
        table = auroc_table(te_probs, te_y)
        s = summarise(table)
        result["probes"][kind] = {"l2": l2, "val_auroc_mean": val, "test": s,
                                  "per_finding": table}
        print(f"[probe] {kind}: test auroc_mean {s['mean']:.4f} (neg {s['negative']:.4f} "
              f"pos {s['positive']:.4f} unc {s['uncertain']:.4f}), l2 {l2:g}", flush=True)

    # Shuffled-label control on pooled: must be ~0.50 or the probe leaks.
    rng = np.random.default_rng(args.seed)
    ytr_shuf = data["train"][2][rng.permutation(data["train"][2].shape[0])]
    _, _, sh = fit_probe(feats("train", "pooled"), ytr_shuf, feats("val", "pooled"),
                         data["val"][2], feats("test", "pooled"), device,
                         epochs=args.epochs, seed=args.seed)
    result["pooled_shuffled_test_auroc_mean"] = summarise(auroc_table(sh, te_y))["mean"]
    print(f"[probe] shuffled-label control: {result['pooled_shuffled_test_auroc_mean']:.4f}",
          flush=True)

    print("\n[probe] per finding, positive-class AUROC on test: MHCAC / pooled / tokens")
    mh_table = auroc_table(torch.softmax(te_logits, -1).numpy(), te_y)
    for name in FINDINGS:
        cells = [t.get(name, {}).get("positive") for t in
                 (mh_table, result["probes"]["pooled"]["per_finding"],
                  result["probes"]["tokens"]["per_finding"])]
        if all(c is not None for c in cells):
            print(f"    {name:<28} {cells[0]:.4f} / {cells[1]:.4f} / {cells[2]:.4f}")
    if args.report:
        result["mhcac_per_finding"] = mh_table
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[probe] wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
