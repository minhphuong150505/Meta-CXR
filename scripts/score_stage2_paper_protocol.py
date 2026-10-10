#!/usr/bin/env python3
"""Score Stage-2 generations with the META-CXR paper protocol, plus paired comparisons.

Brings into the repo what used to live only on the training host
(``~/paper_nlg_gpu.py``, ``~/pair_cmp.py``), so every Stage-2 number from
2026-10-11 on is reproducible from git:

* paper protocol, corpus level: strip newlines, nltk word_tokenize, lowercase;
  pycocoevalcap BLEU-1..4, METEOR, ROUGE-L, CIDEr; BERTScore F1 with
  ``distilroberta-base`` and ``rescale_with_baseline=True``;
* lexicon CE (``safety.claims.LexiconClaimParser`` on both sides; a RELATIVE
  measure, not CheXbert CE) via ``scripts/analyze_stage2_generations.py``;
* ``--baseline NAME``: every other file is compared with it on the shared
  ``sample_key``s -- per-study BERTScore / ROUGE-L / CIDEr / BLEU-4 means and
  the lexicon-CE macro F1, each with a paired bootstrap CI (2,000 resamples,
  seed 16 by default).

Accepts JSONL rows with ``pred``/``ref`` or ``generated``/``reference``.
Prints aggregates only. Needs nltk, pycocoevalcap (Java for METEOR),
bert_score and a GPU for BERTScore -- run on the training host.

    python scripts/score_stage2_paper_protocol.py \\
        --pred P0=<dir>/generated_val.jsonl --pred P1=<...> --baseline P0 \\
        --output <private dir>/pilot_scores.json
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


def load_rows(path: Path) -> dict[str, tuple[str, str]]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        pred = d.get("pred", d.get("generated", ""))
        ref = d.get("ref", d.get("reference", ""))
        out[d["sample_key"]] = (pred or "", ref or "")
    return out


def prep(text: str) -> str:
    from nltk.tokenize import word_tokenize

    text = text.replace("\n", "").replace("<s>", "").replace("</s>", "")
    return " ".join(word_tokenize(text)).lower()


def per_study_scores(preds: list[str], refs: list[str], device: str) -> dict[str, np.ndarray]:
    from bert_score import score as bscore
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.rouge.rouge import Rouge

    gts = {str(i): [prep(r)] for i, r in enumerate(refs)}
    res = {str(i): [prep(p)] for i, p in enumerate(preds)}
    _, rl = Rouge().compute_score(gts, res)
    _, ci = Cider().compute_score(gts, res)
    _, bl = Bleu(4).compute_score(gts, res, verbose=0)
    _, _, f1 = bscore(preds, refs, model_type="distilroberta-base", lang="en",
                      rescale_with_baseline=True, batch_size=64, device=device)
    return {"BERTScore": f1.numpy(), "ROUGE-L": np.asarray(rl), "CIDEr": np.asarray(ci),
            "BLEU-4": np.asarray(bl[3])}


def corpus_scores(preds: list[str], refs: list[str], device: str) -> dict[str, float]:
    from bert_score import score as bscore
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge

    gts = {str(i): [prep(r)] for i, r in enumerate(refs)}
    res = {str(i): [prep(p)] for i, p in enumerate(preds)}
    out: dict[str, float] = {}
    bleu, _ = Bleu(4).compute_score(gts, res, verbose=0)
    for n, value in enumerate(bleu, 1):
        out[f"BLEU-{n}"] = float(value)
    out["METEOR"] = float(Meteor().compute_score(gts, res)[0])
    out["ROUGE-L"] = float(Rouge().compute_score(gts, res)[0])
    out["CIDEr"] = float(Cider().compute_score(gts, res)[0])
    _, _, f1 = bscore(preds, refs, model_type="distilroberta-base", lang="en",
                      rescale_with_baseline=True, batch_size=64, device=device)
    out["BERTScore"] = float(f1.mean())
    out["words_median"] = float(np.median([len(p.split()) for p in preds]))
    return out


def positive_matrix(texts: list[str]):
    from analyze_stage2_generations import FINDINGS, labels
    from safety.claims import NEGATIVE, LexiconClaimParser

    parser = LexiconClaimParser()
    mat = np.zeros((len(texts), len(FINDINGS)), dtype=bool)
    for i, text in enumerate(texts):
        lab = labels(parser, text)
        for j, f in enumerate(FINDINGS):
            mat[i, j] = lab.get(f) not in (None, NEGATIVE)
    return mat


def macro_f1(pred_pos: np.ndarray, ref_pos: np.ndarray, columns: np.ndarray) -> float:
    tp = (pred_pos & ref_pos).sum(0)
    fp = (pred_pos & ~ref_pos).sum(0)
    fn = (~pred_pos & ref_pos).sum(0)
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    return float(f1[columns].mean())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", action="append", required=True, help="NAME=path.jsonl")
    ap.add_argument("--baseline", default=None, help="NAME to compare every other file with")
    ap.add_argument("--resamples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)

    runs = {}
    for spec in args.pred:
        name, path = spec.split("=", 1)
        runs[name] = load_rows(Path(path))
    if args.baseline is not None and args.baseline not in runs:
        raise SystemExit(f"--baseline {args.baseline} is not one of {sorted(runs)}")

    report: dict = {"protocol": "paper (nltk tokenise, lowercase, pycocoevalcap; BERTScore "
                                "distilroberta-base rescaled) + lexicon CE (relative)",
                    "runs": {}, "paired": {}}
    for name, rows in runs.items():
        keys = sorted(rows)
        preds, refs = [rows[k][0] for k in keys], [rows[k][1] for k in keys]
        scores = corpus_scores(preds, refs, args.device)
        pp, rp = positive_matrix(preds), positive_matrix(refs)
        scored = np.nonzero(rp.sum(0) >= 10)[0]
        scores["lexicon_CE_macro_F1"] = macro_f1(pp, rp, scored)
        scores["n"] = len(keys)
        report["runs"][name] = scores
        print(f"[score] {name}: n={len(keys)} " + " ".join(
            f"{k}={v:.4f}" for k, v in scores.items() if isinstance(v, float)), flush=True)

    if args.baseline is not None:
        base = runs[args.baseline]
        rng = np.random.default_rng(args.seed)
        for name, rows in runs.items():
            if name == args.baseline:
                continue
            keys = sorted(set(base) & set(rows))
            if not keys:
                raise SystemExit(f"{name} and {args.baseline} share no sample_key")
            refs = [base[k][1] for k in keys]
            b_pred = [base[k][0] for k in keys]
            c_pred = [rows[k][0] for k in keys]
            sb = per_study_scores(b_pred, refs, args.device)
            sc = per_study_scores(c_pred, refs, args.device)
            n = len(keys)
            idx = rng.integers(0, n, (args.resamples, n))
            entry = {"n": n, "identical_outputs": int(sum(a == b for a, b in zip(b_pred, c_pred)))}
            for metric in sb:
                d = sc[metric] - sb[metric]
                lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
                entry[metric] = {"delta": float(d.mean()), "ci95": [float(lo), float(hi)]}
            rp = positive_matrix(refs)
            pb, pc = positive_matrix(b_pred), positive_matrix(c_pred)
            scored = np.nonzero(rp.sum(0) >= 10)[0]
            deltas = [macro_f1(pc[i], rp[i], scored) - macro_f1(pb[i], rp[i], scored) for i in idx]
            lo, hi = np.percentile(deltas, [2.5, 97.5])
            entry["lexicon_CE_macro_F1"] = {
                "delta": macro_f1(pc, rp, scored) - macro_f1(pb, rp, scored),
                "ci95": [float(lo), float(hi)]}
            report["paired"][f"{name}_minus_{args.baseline}"] = entry
            print(f"[paired] {name} - {args.baseline} (n={n}): " + "  ".join(
                f"{m} {v['delta']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]"
                for m, v in entry.items() if isinstance(v, dict)), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
