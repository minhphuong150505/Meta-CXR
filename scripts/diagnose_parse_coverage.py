#!/usr/bin/env python3
"""Why does `lexicon_v1` leave 40% of sentences unlabelled?

Measuring coverage says how much is missed. This says WHAT is missed, which is
the number that decides whether a better labeler is worth building.

Four buckets, not three. The split that matters is inside "a finding was
described": a finding that IS one of the repository's 14 labels but the lexicon
did not match is a labeler weakness and is cheap to fix with synonyms; a
finding that is NOT one of the 14 (degenerative change, hiatal hernia, aortic
calcification, scoliosis, surgical clips) cannot be fixed by any labeler
confined to that taxonomy, and needs a bigger one. Reporting them together
would overstate what better synonyms could buy.

  technical   -- comparison with priors, projection, technique, positioning
  normal      -- a normality or negation statement naming no finding
  missed_14   -- describes one of the 14 labels; the lexicon should have caught it
  outside_14  -- describes something real that is not one of the 14

The classifier is keyword-based and therefore approximate. It writes the
sentences it sorted, grouped by bucket, so its work can be checked rather than
trusted -- that file is report text and is the most sensitive artifact here.

PRIVACY: output goes through the same guard as the other XAI commands and is
never printed to stdout. Only counts reach the terminal.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Sequence
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from safety.claims import ABNORMALITY_SYNONYMS  # noqa: E402
from scripts.evaluate_explanation import _assert_private_output_location  # noqa: E402
from training.explainability.sentence_attribution import (  # noqa: E402
    LexiconSentenceLabeler,
    locate_sentences,
)

TECHNICAL = (
    "compar", "prior", "previous", "interval", "since", "earlier", "radiograph",
    "film", "projection", "view", "technique", "technical", "rotat", "penetrat",
    "portable", "upright", "supine", "semi-erect", "exam", "study", "obtained",
    "lordotic", "expiratory", "inspiration", "well expanded", "low volume",
    "limited", "obscur", "overlying", "artifact",
)
NORMAL = (
    "unremarkable", "within normal limits", "normal", "clear", "no acute",
    "grossly", "stable", "unchanged", "no significant", "no evidence of acute",
    "intact", "preserved", "no abnormal",
)
# Things a radiologist genuinely reports that the 14-label taxonomy has no slot
# for. A better labeler *within* those 14 cannot recover these.
OUTSIDE_14 = (
    "degenerative", "spondylo", "scoliosis", "kyphosis", "osteopenia",
    "hiatal hernia", "hernia", "tortuos", "aortic calcification", "calcified",
    "atherosclero", "surgical clip", "clips", "granuloma", "scarring",
    "emphysema", "hyperinflat", "copd", "bronchiectasis", "goiter", "thyroid",
    "breast", "nipple", "gas", "bowel", "abdomen", "diaphragm", "elevation",
    "azygos", "situs", "dextrocardia", "cervical", "shoulder", "rib",
)
# Wording for the 14 labels that `lexicon_v1` does NOT currently match.
MISSED_14 = (
    "heart size", "cardiac silhouette", "cardiomediastinal silhouette",
    "cardiopulmonary", "vascular", "congestion", "infiltrate", "haziness",
    "density", "densities", "blunting", "costophrenic", "fluid",
    "aeration", "collapse", "consolidat", "opacif", "pneumonic",
    "interstitial", "reticular", "septal", "kerley", "perihilar fullness",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", default="validate")
    parser.add_argument("--studies", type=int, default=300)
    parser.add_argument("--sample", type=int, default=30,
                        help="unparsed sentences to write out for a human to read")
    parser.add_argument("--seed", type=int, default=16)
    return parser.parse_args(argv)


def classify(sentence: str) -> str:
    """Approximate, keyword-based, and checked by the file it writes."""
    lowered = sentence.lower()

    def hit(terms):
        return any(term in lowered for term in terms)

    # Order matters. A sentence comparing with a prior is technical even when
    # it names a finding, because the labeler's failure there is not a missing
    # synonym. Normality is checked before the finding buckets for the same
    # reason: "the lungs are clear" is a negation, not a missed opacity.
    if hit(TECHNICAL):
        return "technical"
    if hit(NORMAL):
        return "normal"
    if hit(MISSED_14):
        return "missed_14"
    if hit(OUTSIDE_14):
        return "outside_14"
    return "unclassified"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = _assert_private_output_location(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    frame = pd.read_csv(args.manifest)
    if "split" in frame.columns:
        frame = frame[frame["split"].isin({args.split, "val", "validate"})]
    frame = frame[frame["target_valid"] & frame["ViewPosition"].isin(["PA", "AP"])]
    frame = frame.sample(
        n=min(args.studies, len(frame)), random_state=args.seed
    ).reset_index(drop=True)

    labeler = LexiconSentenceLabeler()
    buckets: dict[str, list[str]] = {}
    total = labelled = 0
    for text in frame["findings_clean"].fillna(""):
        for sentence, _start, _end in locate_sentences(str(text)):
            total += 1
            if labeler.label(sentence):
                labelled += 1
                continue
            buckets.setdefault(classify(sentence), []).append(sentence)

    unparsed = total - labelled
    counts = {name: len(values) for name, values in sorted(buckets.items())}
    summary = {
        "studies": int(len(frame)),
        "sentences": total,
        "labelled": labelled,
        "parse_coverage": labelled / total if total else 0.0,
        "unparsed": unparsed,
        "buckets": counts,
        "bucket_share_of_unparsed": {
            name: round(value / unparsed, 4) for name, value in counts.items()
        } if unparsed else {},
        "labels_in_taxonomy": sorted(ABNORMALITY_SYNONYMS),
    }
    (output_dir / "parse_coverage_diagnosis.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # The sample, grouped so the classifier can be checked. REPORT TEXT.
    rng = random.Random(args.seed)
    flat = [(name, s) for name, values in buckets.items() for s in values]
    rng.shuffle(flat)
    lines = ["# Unparsed sentences, grouped by the classifier's guess.",
             "# PhysioNet credentialed report text -- do not copy out of this file.",
             ""]
    for name in sorted(buckets):
        picked = [s for bucket, s in flat if bucket == name][: args.sample]
        if not picked:
            continue
        lines.append(f"## {name}  ({counts[name]} total, showing {len(picked)})")
        lines.extend(f"  - {s}" for s in picked)
        lines.append("")
    (output_dir / "unparsed_sample.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"studies {summary['studies']}  sentences {total}  "
          f"parse_coverage {summary['parse_coverage']:.3f}")
    print(f"unparsed {unparsed}:")
    for name, value in counts.items():
        print(f"  {name:12s} {value:5d}  {value / unparsed:6.1%} of unparsed")
    print(f"\nwrote {output_dir}/parse_coverage_diagnosis.json and unparsed_sample.md")
    print("unparsed_sample.md contains report text -- read it on this host only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
