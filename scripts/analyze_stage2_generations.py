#!/usr/bin/env python3
"""Error analysis of Stage-2 generations against their references. Aggregates only.

For each prediction JSONL (one ``{"sample_key", "pred", "ref"}`` per line):

* length (words), distinct outputs, share taken by the single most frequent
  output -- how template-like the model is;
* temporal / comparison language ("unchanged", "compared with prior", ...):
  how often the references use it and how often the model does. A single-image
  model has no prior study, so this is text it cannot ground;
* a lexicon clinical-efficacy proxy: ``safety.claims.LexiconClaimParser``
  labels reference and generation alike (14 CheXpert findings, polarity), then
  per-finding precision / recall / F1 of POSITIVE mentions (uncertain counted
  positive, as CheXbert-based CE usually does), plus mention recall regardless
  of polarity (how much of what the radiologist talked about the model talks
  about at all). It is a rule-based stand-in for CheXbert CE: use it to compare
  systems on the same studies, never as an absolute CE number.

No report text, identifier or path is printed.

    python scripts/analyze_stage2_generations.py --pred name=<file.jsonl> [--pred ...] \\
        [--report <private dir>/analysis.json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from safety.claims import ABNORMALITY_SYNONYMS, NEGATIVE, LexiconClaimParser  # noqa: E402

TEMPORAL = re.compile(
    r"\b(unchanged|stable|again|interval|compared|comparison|prior|previous|previously|"
    r"since|persistent|persists|new|newly|increased|decreased|improved|improving|"
    r"worsened|worsening|resolved|resolving|redemonstrat\w*|no longer)\b",
    re.IGNORECASE,
)
FIVE = ("Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion")
FINDINGS = tuple(ABNORMALITY_SYNONYMS)


def labels(parser: LexiconClaimParser, text: str) -> dict[str, str]:
    """finding -> polarity; a positive or uncertain mention beats a negative one."""
    out: dict[str, str] = {}
    for claim in parser.parse(text or ""):
        if out.get(claim.finding) in (None, NEGATIVE):
            out[claim.finding] = claim.polarity
    return out


def analyse(rows: list[dict], parser: LexiconClaimParser) -> dict:
    preds = [r["pred"] or "" for r in rows]
    refs = [r["ref"] or "" for r in rows]
    n = len(rows)
    words_p = np.array([len(p.split()) for p in preds])
    words_r = np.array([len(r.split()) for r in refs])
    top = Counter(preds).most_common(1)[0][1] if preds else 0
    t_ref = np.array([bool(TEMPORAL.search(r)) for r in refs])
    t_pred = np.array([bool(TEMPORAL.search(p)) for p in preds])
    lp = [labels(parser, p) for p in preds]
    lr = [labels(parser, r) for r in refs]
    per = {}
    for f in FINDINGS:
        gp = np.array([d.get(f) not in (None, NEGATIVE) for d in lp])
        gr = np.array([d.get(f) not in (None, NEGATIVE) for d in lr])
        mp = np.array([f in d for d in lp])
        mr = np.array([f in d for d in lr])
        tp = int((gp & gr).sum())
        prec = tp / gp.sum() if gp.sum() else float("nan")
        rec = tp / gr.sum() if gr.sum() else float("nan")
        f1 = (2 * prec * rec / (prec + rec)) if gp.sum() and gr.sum() and (prec + rec) > 0 else 0.0
        per[f] = {"ref_pos": int(gr.sum()), "pred_pos": int(gp.sum()), "precision": prec,
                  "recall": rec, "f1": f1,
                  "mention_recall": float((mp & mr).sum() / mr.sum()) if mr.sum() else float("nan")}
    scored = [f for f in FINDINGS if per[f]["ref_pos"] >= 10]
    return {
        "n": n,
        "words_median_pred": float(np.median(words_p)), "words_median_ref": float(np.median(words_r)),
        "words_ratio_mean": float((words_p / np.maximum(words_r, 1)).mean()),
        "distinct_outputs": len(set(preds)), "top_output_share": top / n if n else 0.0,
        "temporal_ref": float(t_ref.mean()), "temporal_pred": float(t_pred.mean()),
        "temporal_pred_given_ref": float(t_pred[t_ref].mean()) if t_ref.any() else float("nan"),
        "temporal_pred_given_not_ref": float(t_pred[~t_ref].mean()) if (~t_ref).any() else float("nan"),
        "lexicon_ce_macro_f1": float(np.mean([per[f]["f1"] for f in scored])),
        "lexicon_ce_f1_5": float(np.mean([per[f]["f1"] for f in FIVE])),
        "lexicon_ce_macro_precision": float(np.nanmean([per[f]["precision"] for f in scored])),
        "lexicon_ce_macro_recall": float(np.nanmean([per[f]["recall"] for f in scored])),
        "mention_recall_macro": float(np.nanmean([per[f]["mention_recall"] for f in scored])),
        "per_finding": per,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", action="append", required=True, help="name=path.jsonl")
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args(argv)
    parser = LexiconClaimParser()
    out = {}
    for spec in args.pred:
        name, path = spec.split("=", 1)
        rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
        res = analyse(rows, parser)
        out[name] = res
        print(f"[gen] {name}: n={res['n']} words pred/ref {res['words_median_pred']:.0f}/"
              f"{res['words_median_ref']:.0f} distinct={res['distinct_outputs']} "
              f"top_share={res['top_output_share']:.3f} temporal ref/pred "
              f"{res['temporal_ref']:.3f}/{res['temporal_pred']:.3f} | lexCE macroF1 "
              f"{res['lexicon_ce_macro_f1']:.3f} F1-5 {res['lexicon_ce_f1_5']:.3f} "
              f"P {res['lexicon_ce_macro_precision']:.3f} R {res['lexicon_ce_macro_recall']:.3f} "
              f"mention_recall {res['mention_recall_macro']:.3f}", flush=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(out, indent=2, sort_keys=True, default=float) + "\n",
                               encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
