#!/usr/bin/env python3
"""Nearest-neighbour report retrieval baselines from cached Stage-2 records.

How much report-relevant information does each representation carry? For every
test study, copy the FINDINGS of the nearest train study under a given key and
score the copies exactly like generated reports. Keys:

* ``random``     -- a random train report (floor);
* ``constant``   -- the most frequent train report for every study;
* ``softtok``    -- cosine on the mean of the 32 Q-Former soft tokens;
* ``mhcac``      -- MHCAC's argmax three-class labels of the test study matched
                    to the TRUE labels of train studies (Hamming, random ties);
                    ``mhcac_cutpoints`` the same with the headline cutpoints;
* ``oracle``     -- the test study's TRUE CheXpert labels matched the same way:
                    what label information alone can buy when the text is real;
* ``npz:<name>`` -- cosine on an external embedding (e.g. MedGemma's own image
                    encoder) given as ``--embedding name=<train.npz>,<test.npz>``
                    holding ``keys`` (image basename) and ``emb``.

Writes one JSONL per key (``sample_key`` of the test cache, ``pred``, ``ref``)
into ``--out-dir`` -- report text, so a private directory on the training host
only. Prints counts only.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (_REPO_ROOT, _REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def load(path: Path, limit: int | None, seed: int):
    import torch

    records = torch.load(path, map_location="cpu", weights_only=False, mmap=True)["records"]
    order = np.arange(len(records))
    if limit and limit < len(order):
        order = np.sort(np.random.default_rng(seed).choice(order, limit, replace=False))
    sub = [records[i] for i in order]
    pooled = torch.stack([r["qformer_embs"].float().mean(0) for r in sub]).numpy()
    logits = torch.stack([r["class_logits"].float() for r in sub]).numpy()
    keys = [str(r["image_path"]).rsplit("/", 1)[-1] for r in sub]
    return sub, pooled, logits, keys


def cosine_nn(query: np.ndarray, base: np.ndarray, batch: int = 1024) -> np.ndarray:
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    b = torch.nn.functional.normalize(torch.as_tensor(base, dtype=torch.float32, device=dev), dim=-1)
    out = []
    for s in range(0, len(query), batch):
        q = torch.nn.functional.normalize(
            torch.as_tensor(query[s:s + batch], dtype=torch.float32, device=dev), dim=-1)
        out.append((q @ b.T).argmax(-1).cpu().numpy())
    return np.concatenate(out)


def label_nn(query: np.ndarray, base: np.ndarray, rng) -> np.ndarray:
    """Hamming nearest neighbour on [N, 14] class codes (-1 = unknown), random ties."""
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    b = torch.as_tensor(base, device=dev)
    out = np.empty(len(query), dtype=np.int64)
    for i, q in enumerate(query):
        qt = torch.as_tensor(q, device=dev)
        dist = (b != qt).sum(-1)
        best = torch.nonzero(dist == dist.min(), as_tuple=True)[0].cpu().numpy()
        out[i] = rng.choice(best)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-cache", type=Path, required=True)
    ap.add_argument("--test-cache", type=Path, required=True)
    ap.add_argument("--manifest-dir", type=Path, required=True)
    ap.add_argument("--chexpert", type=Path, required=True)
    ap.add_argument("--train-limit", type=int, default=60000)
    ap.add_argument("--embedding", action="append", default=[],
                    help="name=<train.npz>,<test.npz> with arrays keys, emb")
    ap.add_argument("--cutpoints", type=Path, default=None,
                    help="also match MHCAC's cutpoint decisions (headline rule)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=16)
    args = ap.parse_args(argv)

    import pandas as pd

    from probe_soft_tokens_cached import labels_for

    rng = np.random.default_rng(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    chexpert = pd.read_csv(args.chexpert)
    tr, tr_pool, _tr_logits, tr_keys = load(args.train_cache, args.train_limit, args.seed)
    te, te_pool, te_logits, te_keys = load(args.test_cache, None, args.seed)
    y_tr = labels_for(tr_keys, args.manifest_dir / "train.csv", chexpert)
    y_te = labels_for(te_keys, args.manifest_dir / "test.csv", chexpert)
    have = (y_tr >= 0).any(1)
    print(f"[ret] train database {len(tr)} studies ({int(have.sum())} labelled); test {len(te)}",
          flush=True)
    refs = [r["ref"] for r in tr]

    picks: dict[str, np.ndarray] = {}
    picks["random"] = rng.integers(0, len(tr), len(te))
    most = Counter(refs).most_common(1)[0]
    print(f"[ret] most frequent train report occurs {most[1]} times "
          f"({most[1] / len(tr):.3%} of the database)", flush=True)
    picks["constant"] = np.full(len(te), refs.index(most[0]))
    picks["softtok"] = cosine_nn(te_pool, tr_pool)
    lab_idx = np.nonzero(have)[0]
    pred_codes = te_logits.argmax(-1)
    picks["mhcac"] = lab_idx[label_nn(pred_codes, y_tr[lab_idx], rng)]
    if args.cutpoints is not None:
        from probe_soft_tokens_cached import FINDINGS

        from training.evaluation.threshold_calibration import apply_cutpoints, load_cutpoints

        probs = np.exp(te_logits - te_logits.max(-1, keepdims=True))
        probs = probs / probs.sum(-1, keepdims=True)
        cut_codes = apply_cutpoints(probs, load_cutpoints(args.cutpoints), FINDINGS)
        picks["mhcac_cutpoints"] = lab_idx[label_nn(cut_codes, y_tr[lab_idx], rng)]
    oracle_q = np.where(y_te >= 0, y_te, -2)          # unknown never matches
    picks["oracle"] = lab_idx[label_nn(oracle_q, y_tr[lab_idx], rng)]
    for spec in args.embedding:
        name, files = spec.split("=", 1)
        f_tr, f_te = files.split(",")
        a, b = np.load(f_tr), np.load(f_te)
        pos_tr = {k: i for i, k in enumerate(a["keys"].tolist())}
        pos_te = {k: i for i, k in enumerate(b["keys"].tolist())}
        tr_ok = np.array([k in pos_tr for k in tr_keys])
        te_ok = all(k in pos_te for k in te_keys)
        if not te_ok:
            raise SystemExit(f"embedding {name}: test keys missing")
        base_idx = np.nonzero(tr_ok)[0]
        base = a["emb"][[pos_tr[tr_keys[i]] for i in base_idx]]
        query = b["emb"][[pos_te[k] for k in te_keys]]
        picks[f"npz_{name}"] = base_idx[cosine_nn(query, base)]
        print(f"[ret] embedding {name}: database {len(base_idx)} of {len(tr)}", flush=True)

    for name, idx in picks.items():
        with open(args.out_dir / f"{name}.jsonl", "w", encoding="utf-8") as fh:
            for rec, j in zip(te, idx):
                fh.write(json.dumps({"sample_key": rec["sample_key"], "pred": refs[int(j)],
                                     "ref": rec["ref"]}) + "\n")
        print(f"[ret] wrote {name}: {len(set(int(j) for j in idx))} distinct reports", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
