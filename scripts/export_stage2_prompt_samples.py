#!/usr/bin/env python3
"""Render Stage-2 prompts for a handful of records into a debug JSONL.

Never writes Q-Former embedding tensors. Rendered per-sample prompts DO contain
findings text, so this is a debug tool: point ``--output`` at a private location.

    python scripts/export_stage2_prompt_samples.py \
        --config configs/stage2_prompt_v2.yaml \
        --input outputs/stage2_records.jsonl \
        --num-samples 50 \
        --output outputs/prompt_samples.jsonl

With no ``--input`` it renders synthetic (non-MIMIC) records so the tool is
runnable offline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts._stage2_fixtures import synthetic_records  # noqa: E402
from stage2.prompts import (  # noqa: E402
    PromptBuilder,
    VisualMode,
    context_from_record,
    load_prompt_config,
)
from stage2.prompts.schemas import PartKind  # noqa: E402


def _load_records(path: Path | None, num: int) -> list[dict]:
    if path is None:
        return synthetic_records(num)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:num]


def _word_count(text: str) -> int:
    return len(text.split())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/stage2_prompt_v2.yaml")
    parser.add_argument("--input", type=Path, help="stage2 records JSONL; omit for synthetic")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs/prompt_samples.jsonl")
    parser.add_argument("--soft-token", default="<qformer_soft_token>")
    args = parser.parse_args()

    config = load_prompt_config(args.config)
    builder = PromptBuilder(config)
    records = _load_records(args.input, args.num_samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            context = context_from_record(record, visual_mode=config.visual_mode)
            rendered = builder.build(context)
            user_text = rendered.user_text(args.soft_token)
            structured = [
                p.text for p in rendered.parts if p.kind is PartKind.TEXT and p.text
            ]
            negatives_line = next(
                (t for t in structured if t.startswith("- Clinically relevant absent")), ""
            )
            debug = {
                "study_id": context.study_id,
                "visual_mode": config.visual_mode.value,
                "prompt_version": rendered.prompt_version,
                "prompt_hash": rendered.prompt_hash,
                "template_hash": rendered.template_hash,
                "config_hash": rendered.config_hash,
                "positive_findings": list(context.positive_findings),
                "uncertain_findings": list(context.uncertain_findings),
                "selected_negative_findings": negatives_line.split(":", 1)[-1].strip(),
                "anchor_view": context.anchor_view,
                "auxiliary_views": list(context.auxiliary_views),
                "prior_available": context.prior_available,
                "rendered_user_prompt": user_text,
                "prompt_word_count": _word_count(user_text),
                "target_word_count": _word_count(str(record.get("ref", ""))),
                "soft_token_count": user_text.count(args.soft_token),
            }
            handle.write(json.dumps(debug, ensure_ascii=False) + "\n")
            written += 1

    print(f"[export] wrote {written} debug prompt(s) -> {args.output}")
    print(f"[export] visual_mode={config.visual_mode.value} prompt_hash={builder.build(context).prompt_hash}")


if __name__ == "__main__":
    main()
