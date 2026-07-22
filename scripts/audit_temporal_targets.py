#!/usr/bin/env python3
"""Quantify temporal language in FINDINGS targets vs prior availability.

The failure this measures: a target that says "unchanged / compared to prior" while
the model's input has no prior trains confident temporal hallucination. This script
counts how often that happens so a ``temporal_target_policy`` can be chosen with
evidence rather than by guessing.

    python scripts/audit_temporal_targets.py --input outputs/stage2_records.jsonl \
        --output outputs/temporal_target_audit.json

Each record needs a ``ref`` (target) and, ideally, ``prior_available``. With no
``--input`` it runs on synthetic (non-MIMIC) records; those counts are illustrative
only, NOT a measurement of MIMIC-CXR.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts._stage2_fixtures import synthetic_records  # noqa: E402
from stage2.prompts import contains_temporal_language  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="records JSONL with 'ref'/'prior_available'")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs/temporal_target_audit.json")
    parser.add_argument("--ref-key", default="ref")
    parser.add_argument("--prior-key", default="prior_available")
    args = parser.parse_args()

    if args.input:
        rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    else:
        rows = synthetic_records(200)

    total = len(rows)
    temporal = 0
    with_prior = 0
    temporal_without_prior = 0
    examples: list[str] = []
    for record in rows:
        target = str(record.get(args.ref_key, ""))
        prior = bool(record.get(args.prior_key, False))
        is_temporal = contains_temporal_language(target)
        temporal += int(is_temporal)
        with_prior += int(prior)
        if is_temporal and not prior:
            temporal_without_prior += 1
            if len(examples) < 5:
                examples.append(target[:200])

    report = {
        "total_reports": total,
        "temporal_reports": temporal,
        "reports_with_prior_context": with_prior,
        "temporal_but_no_prior": temporal_without_prior,
        "temporal_but_no_prior_rate": round(temporal_without_prior / total, 4) if total else 0.0,
        "heuristic": "lexical regex over interval-change cues; not a clinical judgement",
        "uses_synthetic_records": args.input is None,
        "examples_temporal_without_prior": examples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if args.input is None:
        print("[warn] synthetic records; real MIMIC counts require --input on real targets", file=sys.stderr)


if __name__ == "__main__":
    main()
