#!/usr/bin/env python3
"""META-CXR Table 3: Clinical Efficacy (precision, recall, macro F1).

Takes the output of a CheXpert-style labeler run on the GENERATED reports and
on their REFERENCES -- two CSVs with a ``sample_key`` column and the 14
CheXpert columns (1 positive, 0 negative, -1 uncertain, blank) -- joins them on
``sample_key`` and scores the paper's CE numbers. A finding counts as present
when the labeler says 1.

⚠ The labeler itself (CheXpert labeler / CheXbert) is not part of this
repository; see ``training/evaluation/clinical.py``. These CSVs are derived
from report text: keep them in a private directory, never in the repo.

    python scripts/evaluate_clinical_efficacy.py \\
        --generated-labels <private>/gen_labels.csv \\
        --reference-labels <private>/ref_labels.csv --output <private>/table3_ce.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.evaluation.paper_protocol import CHEXPERT_14, clinical_efficacy  # noqa: E402

#: Paper Table 3, META-CXR row.
PAPER_CE = {"precision": 0.411, "recall": 0.465, "macro_f1": 0.428}


def main(argv: list[str] | None = None) -> int:
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generated-labels", required=True, type=Path)
    parser.add_argument("--reference-labels", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    gen = pd.read_csv(args.generated_labels)
    ref = pd.read_csv(args.reference_labels)
    for name, frame in (("generated", gen), ("reference", ref)):
        missing = [c for c in ("sample_key", *CHEXPERT_14) if c not in frame.columns]
        if missing:
            print(f"{name} labels lack columns {missing}", file=sys.stderr)
            return 2
    merged = gen.merge(ref, on="sample_key", suffixes=("_gen", "_ref"), validate="one_to_one")
    if len(merged) != len(gen) or len(merged) != len(ref):
        print(f"sample_key sets differ: {len(gen)} generated, {len(ref)} reference, "
              f"{len(merged)} matched", file=sys.stderr)
        return 2
    result = clinical_efficacy(
        merged[[f"{c}_gen" for c in CHEXPERT_14]].to_numpy(dtype=float),
        merged[[f"{c}_ref" for c in CHEXPERT_14]].to_numpy(dtype=float),
    )
    print(f"CE precision {result['macro_precision']:.3f} (paper {PAPER_CE['precision']:.3f})  "
          f"recall {result['macro_recall']:.3f} (paper {PAPER_CE['recall']:.3f})  "
          f"macro F1 {result['macro_f1']:.3f} (paper {PAPER_CE['macro_f1']:.3f})  "
          f"n={result['n_reports']}")
    print(f"micro P/R/F1 {result['micro_precision']:.3f} / {result['micro_recall']:.3f} / "
          f"{result['micro_f1']:.3f}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({**result, "paper_table3_ce": PAPER_CE},
                                          indent=2, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
