#!/usr/bin/env python3
"""Clinical Efficacy (CE) of Stage-2 generations with CheXbert (``f1chexbert``).

The CE that report-generation papers report (MLRG, PromptMRG, R2Gen lineage):
CheXbert labels both the generated and the reference report; per label,
Positive and Uncertain count as positive ("rrg" mode of ``f1chexbert``); then
precision / recall / F1 over the 14 labels (micro and macro) and over the five
common ones. Added 2026-10-11 at the user's request -- it replaces the lexicon
proxy wherever a real CE number is quoted.

Runs in its OWN venv so it cannot move the training stack's torch/transformers:

    uv venv --seed ~/.venvs/chexbert --python 3.11
    ~/.venvs/chexbert/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
    ~/.venvs/chexbert/bin/pip install f1chexbert "transformers==4.53.2"   # 5.x lacks encode_plus
    HF_HUB_OFFLINE=1 ~/.venvs/chexbert/bin/python scripts/score_ce_chexbert.py \\
        --pred P0=<dir>/generated_val.jsonl --pred P1=<...> --baseline P0 \\
        --label-cache <private dir>/chexbert_labels.json --output <private dir>/ce.json

Weights: ``StanfordAIMI/RRG_scorers/chexbert.pth`` (1,314,414,442 bytes, sha256
``6550703c...0fde1``) in ``~/.cache/chexbert/``. On the training host the HF CDN
delivered ~9 KB/s (2026-10-11), so the file was fetched elsewhere, checksum-verified
and copied there; ``bert-base-uncased`` comes from the existing HF cache, hence
``HF_HUB_OFFLINE=1``. Rows may use
``pred``/``ref`` or ``generated``/``reference``. ``--label-cache`` holds
per-report label vectors keyed by a hash of the text (no text) and makes
repeated scoring of the same references free. Prints aggregates only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

FIVE = ("Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion")


def load_rows(path: Path) -> dict[str, tuple[str, str]]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            out[d["sample_key"]] = (d.get("pred", d.get("generated", "")) or "",
                                    d.get("ref", d.get("reference", "")) or "")
    return out


def text_key(text: str) -> str:
    return hashlib.blake2b(text.strip().encode(), digest_size=12).hexdigest()


class Labeler:
    def __init__(self, cache_path: Path | None, device: str):
        from f1chexbert import F1CheXbert

        self.model = F1CheXbert(device=device)
        self.names = list(self.model.target_names)
        self.cache_path = cache_path
        self.cache = (json.loads(cache_path.read_text()) if cache_path and cache_path.is_file()
                      else {})

    def __call__(self, texts: list[str]) -> np.ndarray:
        out = []
        for text in texts:
            key = text_key(text)
            if key not in self.cache:
                self.cache[key] = self.model.get_label(text.strip())
            out.append(self.cache[key])
        return np.asarray(out, dtype=bool)

    def save(self) -> None:
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache))


def prf(pred: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    tp = (pred & ref).sum(0).astype(float)
    fp = (pred & ~ref).sum(0).astype(float)
    fn = (~pred & ref).sum(0).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        r = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f = np.where(p + r > 0, 2 * p * r / (p + r), 0.0)
    mp = tp.sum() / max(tp.sum() + fp.sum(), 1)
    mr = tp.sum() / max(tp.sum() + fn.sum(), 1)
    return {"micro_precision": float(mp), "micro_recall": float(mr),
            "micro_f1": float(2 * mp * mr / (mp + mr)) if mp + mr else 0.0,
            "macro_precision": float(p.mean()), "macro_recall": float(r.mean()),
            "macro_f1": float(f.mean())}


def scores(pred: np.ndarray, ref: np.ndarray, five: np.ndarray) -> dict[str, float]:
    out = {f"{k}_14": v for k, v in prf(pred, ref).items()}
    out.update({f"{k}_5": v for k, v in prf(pred[:, five], ref[:, five]).items()})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", action="append", required=True, help="NAME=path.jsonl")
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--label-cache", type=Path, default=None)
    ap.add_argument("--resamples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)

    runs = {spec.split("=", 1)[0]: load_rows(Path(spec.split("=", 1)[1])) for spec in args.pred}
    if args.baseline is not None and args.baseline not in runs:
        raise SystemExit(f"--baseline {args.baseline} is not one of {sorted(runs)}")
    labeler = Labeler(args.label_cache, args.device)
    five = np.array([labeler.names.index(n) for n in FIVE])
    report: dict = {"labeler": "f1chexbert (CheXbert), rrg mode: positive or uncertain = 1",
                    "runs": {}, "paired": {}}
    labelled = {}
    for name, rows in runs.items():
        keys = sorted(rows)
        pred = labeler([rows[k][0] for k in keys])
        ref = labeler([rows[k][1] for k in keys])
        labelled[name] = dict(zip(keys, zip(pred, ref)))
        report["runs"][name] = {"n": len(keys), **scores(pred, ref, five)}
        r = report["runs"][name]
        print(f"[ce] {name}: n={len(keys)} micro F1-14 {r['micro_f1_14']:.4f} macro F1-14 "
              f"{r['macro_f1_14']:.4f} micro F1-5 {r['micro_f1_5']:.4f} macro F1-5 "
              f"{r['macro_f1_5']:.4f}", flush=True)
    labeler.save()
    if args.baseline is not None:
        rng = np.random.default_rng(args.seed)
        base = labelled[args.baseline]
        for name, rows in labelled.items():
            if name == args.baseline:
                continue
            keys = sorted(set(base) & set(rows))
            ref = np.stack([base[k][1] for k in keys])
            pb = np.stack([base[k][0] for k in keys])
            pc = np.stack([rows[k][0] for k in keys])
            n = len(keys)
            idx = rng.integers(0, n, (args.resamples, n))
            entry = {"n": n}
            for metric in ("micro_f1_14", "macro_f1_14", "micro_f1_5", "macro_f1_5"):
                full = scores(pc, ref, five)[metric] - scores(pb, ref, five)[metric]
                boot = [scores(pc[i], ref[i], five)[metric] - scores(pb[i], ref[i], five)[metric]
                        for i in idx]
                lo, hi = np.percentile(boot, [2.5, 97.5])
                entry[metric] = {"delta": float(full), "ci95": [float(lo), float(hi)]}
            report["paired"][f"{name}_minus_{args.baseline}"] = entry
            print(f"[ce-paired] {name} - {args.baseline} (n={n}): " + "  ".join(
                f"{m} {v['delta']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]"
                for m, v in entry.items() if isinstance(v, dict)), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
