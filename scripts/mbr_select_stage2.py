#!/usr/bin/env python3
"""Minimum-Bayes-risk (MBR) selection over Stage-2 candidate reports.

Added 2026-10-14 to raise BERTScore without retraining. For every study it
keeps the hypothesis with the highest EXPECTED utility against the model's own
sampled reports -- never against the real reference, which plays no part in
the selection:

    E[u(h)] = mean over samples y, y not h itself, of  u(h, y)

Hypotheses are the ``generated`` text of every ``--hyp`` file (e.g. the beam
and the greedy decode) plus every sample in ``--samples``
(``generate_stage2_reports.py --num-samples``). The pseudo-references are the
samples ONLY: a deterministic decode is not a draw from the model's
distribution, so letting it vote would bias the expectation towards itself.
A sample is never scored against itself.

The utility is BERTScore F1 with ``distilroberta-base`` and idf off, the model
and setting of ``score_stage2_paper_protocol.py``. Baseline rescaling is a
positive affine map of F1, so it cannot change the argmax and is not applied.
Identical texts get utility 1.0 without a model call.

⚠ The utility IS the evaluation metric's model, so MBR is expected to raise
BERTScore more than the other metrics. Name it "MBR decoding (utility =
BERTScore)" beside every number it produces, and score the other metrics and
CheXbert CE as well.

Every input must cover the same studies with the same reference text (checked);
rows join on ``sample_key``. Writes ``generated_<split>.jsonl`` (sample_key,
generated, reference, mbr_source) for the scorers and ``mbr_summary.json``,
which holds aggregates only. Run on the training host:

    python scripts/mbr_select_stage2.py --split val \\
        --hyp beam=<dir>/gen_val_beam/generated_val.jsonl \\
        --hyp greedy=<dir>/gen_val_greedy/generated_val.jsonl \\
        --samples <dir>/gen_val_samples/candidates_val.jsonl \\
        --output-dir <private dir>/mbr_val
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

UTILITY_MODEL = "distilroberta-base"
SAMPLE_SOURCE = "sample"

#: (texts_a, texts_b) -> utility per pair; symmetric in its two arguments.
UtilityFn = Callable[[list[str], list[str]], np.ndarray]


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def pool_candidates(hyp_files: dict[str, Path], samples_file: Path) -> list[dict]:
    """One entry per study: texts, their sources, and which texts are samples.

    Hypothesis files come first, in the order given, then the samples in file
    order -- the order that breaks ties. Fails when the files describe
    different studies or disagree about a study's reference, since either means
    the pool would mix two cohorts.
    """
    samples = {}
    for row in _rows(samples_file):
        if "candidates" not in row:
            raise ValueError(f"{samples_file.name}: rows need 'candidates' "
                             "(generate_stage2_reports.py --num-samples)")
        samples[row["sample_key"]] = row
    if not samples:
        raise ValueError(f"{samples_file.name} holds no study")
    hyps = {}
    for name, path in hyp_files.items():
        if name == SAMPLE_SOURCE:
            raise ValueError(f"--hyp name {SAMPLE_SOURCE!r} is reserved for the samples")
        rows = {}
        for row in _rows(path):
            if "generated" not in row:
                raise ValueError(f"{path.name}: rows need 'generated'")
            rows[row["sample_key"]] = row
        if set(rows) != set(samples):
            raise ValueError(
                f"--hyp {name} covers {len(rows)} studies and the samples {len(samples)}, "
                f"{len(set(rows) ^ set(samples))} not shared: not the same cohort")
        hyps[name] = rows
    studies = []
    for key in sorted(samples):
        reference = str(samples[key].get("reference", "")).strip()
        texts, sources = [], []
        for name, rows in hyps.items():
            if str(rows[key].get("reference", "")).strip() != reference:
                raise ValueError(f"--hyp {name} and the samples disagree on a reference")
            texts.append(str(rows[key]["generated"]))
            sources.append(name)
        sampled = [str(t) for t in samples[key]["candidates"]]
        if len(sampled) < 2:
            raise ValueError("every study needs at least 2 samples: with one, a sampled "
                             "hypothesis has nothing to be scored against")
        texts += sampled
        sources += [SAMPLE_SOURCE] * len(sampled)
        studies.append({
            "sample_key": key,
            "reference": samples[key].get("reference", ""),
            "texts": texts,
            "sources": sources,
            "is_ref": [s == SAMPLE_SOURCE for s in sources],
        })
    return studies


def needed_pairs(texts: Sequence[str], is_ref: Sequence[bool]) -> list[tuple[int, int]]:
    """Unordered pairs (i < j) the expectation reads: at least one side a sample."""
    return [(i, j) for i in range(len(texts)) for j in range(i + 1, len(texts))
            if is_ref[i] or is_ref[j]]


def expected_utility(utility: np.ndarray, is_ref: Sequence[bool]) -> np.ndarray:
    """E[u(h_i)] over the pseudo-references, never comparing a text to itself.

    ``utility`` is the symmetric [K, K] matrix; its diagonal is ignored.
    """
    refs = np.asarray(is_ref, dtype=bool)
    k = len(refs)
    out = np.empty(k)
    for i in range(k):
        cols = refs.copy()
        cols[i] = False
        if not cols.any():
            raise ValueError("a hypothesis has no pseudo-reference besides itself")
        out[i] = utility[i, cols].mean()
    return out


def select(studies: list[dict], utility_fn: UtilityFn, chunk: int = 250) -> list[dict]:
    """Fill ``chosen`` / ``expected`` per study, scoring every needed pair once.

    Pairs go to ``utility_fn`` ``chunk`` studies at a time: bert_score keeps an
    embedding per distinct sentence in host memory, which for a whole split
    (~10 candidates x 2,800 studies) would be several GB. np.argmax keeps the
    first maximum, so ties go to the earlier hypothesis.
    """
    for start in range(0, len(studies), chunk):
        block = studies[start:start + chunk]
        flat_a, flat_b, where = [], [], []
        matrices = []
        for b_idx, study in enumerate(block):
            texts = study["texts"]
            m = np.full((len(texts), len(texts)), np.nan)
            for i, j in needed_pairs(texts, study["is_ref"]):
                if texts[i] == texts[j]:
                    m[i, j] = m[j, i] = 1.0
                else:
                    flat_a.append(texts[i])
                    flat_b.append(texts[j])
                    where.append((b_idx, i, j))
            matrices.append(m)
        scores = (np.asarray(utility_fn(flat_a, flat_b), dtype=float) if flat_a
                  else np.zeros(0))
        if scores.shape != (len(flat_a),):
            raise ValueError(f"utility returned shape {scores.shape} for {len(flat_a)} pairs")
        for (b_idx, i, j), value in zip(where, scores, strict=True):
            matrices[b_idx][i, j] = matrices[b_idx][j, i] = value
        for study, m in zip(block, matrices, strict=True):
            expected = expected_utility(m, study["is_ref"])
            if np.isnan(expected).any():
                raise ValueError("a needed candidate pair was never scored")
            study["expected"] = expected
            study["chosen"] = int(np.argmax(expected))
        print(f"[mbr] {min(start + chunk, len(studies))}/{len(studies)} studies", flush=True)
    return studies


def bertscore_utility(device: str, batch_size: int) -> UtilityFn:
    """BERTScore F1 of each pair; the scorer is loaded once, on first use."""
    holder: dict = {}

    def score(a: list[str], b: list[str]) -> np.ndarray:
        if "scorer" not in holder:
            from bert_score import BERTScorer

            holder["scorer"] = BERTScorer(model_type=UTILITY_MODEL, lang="en", idf=False,
                                          rescale_with_baseline=False,
                                          batch_size=batch_size, device=device)
        _, _, f1 = holder["scorer"].score(a, b)
        return f1.numpy()
    return score


def summarize(studies: list[dict], hyp_names: Sequence[str]) -> dict:
    """Aggregates only -- no report text, no identifier."""
    chosen = Counter(s["sources"][s["chosen"]] for s in studies)
    per_source = {}
    for name in list(hyp_names) + [SAMPLE_SOURCE]:
        values = [s["expected"][i] for s in studies
                  for i, src in enumerate(s["sources"]) if src == name]
        per_source[name] = round(float(np.mean(values)), 5) if values else None
    return {
        "n_studies": len(studies),
        "hypotheses_per_study": sorted({len(s["texts"]) for s in studies}),
        "samples_per_study": sorted({sum(s["is_ref"]) for s in studies}),
        "chosen_source_counts": dict(sorted(chosen.items())),
        "mean_expected_utility_by_source": per_source,
        "mean_expected_utility_chosen": round(float(np.mean(
            [s["expected"][s["chosen"]] for s in studies])), 5),
        "chosen_words_median": float(np.median(
            [len(s["texts"][s["chosen"]].split()) for s in studies])),
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hyp", action="append", default=[],
                    help="NAME=generated_<split>.jsonl, a deterministic decode to consider")
    ap.add_argument("--samples", type=Path, required=True,
                    help="candidates_<split>.jsonl from generate_stage2_reports.py --num-samples")
    ap.add_argument("--split", required=True, choices=("val", "test"))
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--device", default=None, help="default: cuda when available, else cpu")
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args(argv)

    from scripts.evaluate_explanation import _assert_private_output_location

    output_dir = _assert_private_output_location(args.output_dir)
    hyp_files = {}
    for spec in args.hyp:
        name, _, path = spec.partition("=")
        if not path or name in hyp_files:
            raise SystemExit(f"--hyp {spec!r}: expected a unique NAME=path")
        hyp_files[name] = Path(path)
    device = args.device
    if device is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    started = time.time()
    studies = pool_candidates(hyp_files, args.samples)
    n_pairs = sum(len(needed_pairs(s["texts"], s["is_ref"])) for s in studies)
    print(f"[mbr] {len(studies)} studies, {n_pairs} candidate pairs, device={device}", flush=True)
    select(studies, bertscore_utility(device, args.batch_size))

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"generated_{args.split}.jsonl"
    with out_path.open("w", encoding="utf-8") as handle:
        for s in studies:
            handle.write(json.dumps({
                "sample_key": s["sample_key"],
                "generated": s["texts"][s["chosen"]],
                "reference": s["reference"],
                "mbr_source": s["sources"][s["chosen"]],
            }, ensure_ascii=False) + "\n")
    summary = {
        "method": "MBR decoding, utility = BERTScore F1 against the samples only",
        "utility_model": UTILITY_MODEL,
        "utility_idf": False,
        "utility_rescaled": False,
        "split": args.split,
        "hyp_files": {k: str(v) for k, v in hyp_files.items()},
        "samples_file": str(args.samples),
        "n_pairs": n_pairs,
        "device": device,
        "wall_seconds": round(time.time() - started, 1),
        **summarize(studies, list(hyp_files)),
    }
    (output_dir / "mbr_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[mbr] chosen by source {summary['chosen_source_counts']}; "
          f"mean E[u] by source {summary['mean_expected_utility_by_source']}; "
          f"written {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
