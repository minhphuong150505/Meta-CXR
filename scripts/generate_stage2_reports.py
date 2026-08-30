#!/usr/bin/env python3
"""Generate FINDINGS with MedGemma and write the JSONL evaluate_stage2.py reads.

Reuses the project's own Stage-2 engine (`VariantLLM`, native image mode) rather
than a parallel implementation, so the prompt, the chat template and the
generation kwargs are exactly what training would have used.

⚠ WITH NO ADAPTER THIS IS A ZERO-SHOT BASELINE, NOT THIS PROJECT'S STAGE 2.
The Stage-2 QLoRA fine-tune has never been run on this host: there is no
adapter to load. Pass `--adapter` when one exists. Until then every number from
this run describes `google/medgemma-1.5-4b-it` out of the box, and must be
labelled that way -- `mode` in the summary says which.

⚠ It is also NOT the Q-Former route. Stage 1 ships `lambda_itc/itm/lm = 0.0`,
so the Q-Former's image path never trained and its soft tokens carry BLIP-2
initialisation; `--image-mode qformer` is deliberately not offered here.

PRIVACY: generated and reference report text are PhysioNet-derived. The output
goes through the same guard as the other commands, filenames carry no
identifier, and rows are keyed by a blake2 `sample_key`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from hashlib import blake2b
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.evaluate_explanation import _assert_private_output_location  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--limit", type=int, default=300,
                        help="studies to generate; 0 means the whole split")
    parser.add_argument("--adapter", type=Path, default=None,
                        help="Stage-2 LoRA adapter. Without it this is zero-shot.")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--frontal-only", action="store_true", default=True)
    return parser.parse_args(argv)


SPLIT_ALIASES = {"val": ("val", "validate"), "test": ("test",)}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = _assert_private_output_location(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    import torch

    from training.train_eval_figure9_llm_variants_200 import VariantLLM

    frame = pd.read_csv(args.manifest)
    if "split" in frame.columns:
        frame = frame[frame["split"].isin(SPLIT_ALIASES[args.split])]
    frame = frame[frame["target_valid"]]
    if args.frontal_only:
        frame = frame[frame["ViewPosition"].isin(["PA", "AP"])]
    if frame.empty:
        raise SystemExit(f"no study for split {args.split!r}")
    n = len(frame) if args.limit <= 0 else min(args.limit, len(frame))
    frame = frame.sample(n=n, random_state=args.seed).reset_index(drop=True)

    mode = "medgemma_direct_finetuned" if args.adapter else "medgemma_direct_zeroshot"
    print(f"[gen] mode={mode} n={n} split={args.split}", flush=True)

    llm = VariantLLM(
        family="medgemma",
        image_mode="native",
        adapter=args.adapter,
        train_adapter=False,
        quantize_4bit=False,
    )
    llm.assert_vision_tower_frozen()

    path = output_dir / f"generated_{args.split}.jsonl"
    started, failures = time.time(), 0
    with path.open("w", encoding="utf-8") as handle:
        for index in range(n):
            row = frame.iloc[index]
            record = {
                "image_path": str(args.image_root / row.image_path),
                "ref": str(row.findings_clean).strip(),
                "pred_groups": {},
            }
            try:
                generated = llm.generate(record, "fine", args.max_new_tokens)
            except Exception as exc:  # one bad study must not lose the run
                failures += 1
                print(f"[gen] study {index} failed: {type(exc).__name__}", flush=True)
                continue
            handle.write(json.dumps({
                "sample_key": blake2b(str(row.dicom_id).encode(), digest_size=12).hexdigest(),
                "generated": generated,
                "reference": record["ref"],
                "view_position": str(row.ViewPosition),
            }, ensure_ascii=False) + "\n")
            if (index + 1) % 25 == 0:
                rate = (index + 1) / (time.time() - started)
                print(f"[gen] {index + 1}/{n}  {rate:.2f} study/s", flush=True)

    summary = {
        "mode": mode,
        "adapter": str(args.adapter) if args.adapter else None,
        "model_id": llm.model_id,
        "split": args.split,
        "n_requested": n,
        "n_written": n - failures,
        "n_failed": failures,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "wall_seconds": round(time.time() - started, 1),
        "peak_vram_bytes": (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
        ),
        "warning": (
            None if args.adapter else
            "ZERO-SHOT: no Stage-2 adapter was loaded. These are base-model "
            "outputs, not this project's fine-tuned Stage 2."
        ),
    }
    (output_dir / "generation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[gen] n={summary['n_written']} written to {path} in "
          f"{summary['wall_seconds']:.0f}s", flush=True)
    if summary["warning"]:
        print(f"[gen] ⚠ {summary['warning']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
