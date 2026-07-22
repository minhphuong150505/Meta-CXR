#!/usr/bin/env python3
"""Prompt / target length statistics for a Stage-2 records file.

Accurate token counts need the MedGemma tokenizer (``--tokenizer google/medgemma-1.5-4b-it``,
requires transformers). Without it the script falls back to a whitespace proxy and
marks the output ``"approximate": true`` so no one mistakes a proxy for real token
counts. Writes ``outputs/prompt_length_statistics.json``.

    python scripts/prompt_length_statistics.py --config configs/stage2_prompt_v2.yaml \
        --input outputs/stage2_records.jsonl --max-length 768
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts._stage2_fixtures import synthetic_records  # noqa: E402
from stage2.prompts import PromptBuilder, context_from_record, load_prompt_config  # noqa: E402


def _percentile(values: list[int], pct: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    rank = min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[rank]


def _make_counter(name: str | None):
    if not name:
        return (lambda text: len(text.split())), True, "whitespace"
    from transformers import AutoTokenizer  # imported lazily; heavy dependency

    tokenizer = AutoTokenizer.from_pretrained(name)
    return (lambda text: len(tokenizer(text, add_special_tokens=False).input_ids)), False, name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/stage2_prompt_v2.yaml")
    parser.add_argument("--input", type=Path, help="stage2 records JSONL; omit for synthetic")
    parser.add_argument("--tokenizer", help="HF tokenizer id for exact counts (else whitespace proxy)")
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs/prompt_length_statistics.json")
    args = parser.parse_args()

    config = load_prompt_config(args.config)
    builder = PromptBuilder(config)
    count, approximate, counter_name = _make_counter(args.tokenizer)

    if args.input:
        rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
        rows = rows[: args.num_samples]
    else:
        rows = synthetic_records(args.num_samples)

    prompt_lens: list[int] = []
    target_truncated = 0
    soft_truncated = 0
    soft_tokens = 0
    for record in rows:
        context = context_from_record(record, visual_mode=config.visual_mode)
        rendered = builder.build(context)
        user_text = rendered.user_text()
        prompt_tokens = count(user_text)
        prompt_lens.append(prompt_tokens)
        target_tokens = count(str(record.get("ref", "")))
        # A record is target-truncated when prompt + target exceeds the budget.
        if prompt_tokens + target_tokens > args.max_length:
            target_truncated += 1
        soft = user_text.count("<qformer_soft_token>")
        soft_tokens += soft
        if soft and prompt_tokens > args.max_length:
            soft_truncated += 1

    stats = {
        "approximate": approximate,
        "counter": counter_name,
        "visual_mode": config.visual_mode.value,
        "prompt_version": config.version,
        "max_length": args.max_length,
        "num_samples": len(rows),
        "prompt_tokens": {
            "p50": _percentile(prompt_lens, 50),
            "p90": _percentile(prompt_lens, 90),
            "p95": _percentile(prompt_lens, 95),
            "p99": _percentile(prompt_lens, 99),
            "max": max(prompt_lens) if prompt_lens else 0,
        },
        "target_truncation_rate": round(target_truncated / len(rows), 4) if rows else 0.0,
        "soft_token_truncation_rate": round(soft_truncated / len(rows), 4) if rows else 0.0,
        "uses_synthetic_records": args.input is None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    if approximate:
        print("[warn] whitespace proxy counts; pass --tokenizer for real MedGemma tokens", file=sys.stderr)


if __name__ == "__main__":
    main()
