#!/usr/bin/env python3
"""Render example sentence-level heatmaps over the radiograph they explain.

Companion to `explain_stage2.py`, which stores maps as .npz at the native 16x16
grid on purpose. This turns a handful of them into pictures for a human to look
at, which is a different job with different rules: an overlay on a radiograph
IS a patient image, so it never leaves the host and never goes near the repo.

The join back to the image is reconstructed deterministically from the same
manifest, split, seed and limit the run used -- map ``study_{i:05d}.npz`` is
row ``i`` of that cohort. It is NOT read from a key map, so no new identifier
file is created. The reconstruction is checked against the JSONL line count and
refuses to render on a mismatch, because a silent off-by-one would put every
study's heatmap on a different patient's chest.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.evaluate_explanation import _assert_private_output_location  # noqa: E402
from training.explainability.projection import (  # noqa: E402
    MEDGEMMA_GRID,
    OriginalFrame,
    medgemma_map_in_original,
)

SPLIT_ALIASES = {"val": ("val", "validate"), "test": ("test",)}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, type=Path, help="explain_stage2 output dir")
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--image-root", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--split", default="test", choices=("val", "test"))
    p.add_argument("--seed", type=int, default=16, help="must match the run")
    p.add_argument("--limit", type=int, default=0, help="must match the run")
    p.add_argument("--ablation-studies", type=int, default=100, help="must match the run")
    p.add_argument("--examples", type=int, default=6, help="studies to render")
    p.add_argument("--max-sentences", type=int, default=4, help="panels per study")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out = _assert_private_output_location(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image

    rows = [json.loads(line) for line in
            (args.run_dir / f"explanations_{args.split}.jsonl").read_text().splitlines()]

    frame = pd.read_csv(args.manifest)
    if "split" in frame.columns:
        frame = frame[frame["split"].isin(SPLIT_ALIASES[args.split])]
    frame = frame[frame["target_valid"] & frame["ViewPosition"].isin(["PA", "AP"])]
    wanted = (len(frame) if args.limit <= 0
              else min(len(frame), max(args.limit, args.ablation_studies) + 1))
    frame = frame.sample(n=wanted, random_state=args.seed).reset_index(drop=True)

    planned = len(frame) if args.limit <= 0 else min(args.limit, len(frame) - 1)
    if planned != len(rows):
        raise SystemExit(
            f"cohort reconstruction disagrees with the run: rebuilt {planned} studies "
            f"but the JSONL holds {len(rows)}. Re-check --split/--seed/--limit/"
            "--ablation-studies against the run's summary.json; rendering on a "
            "mismatched join would overlay each map on the wrong patient."
        )

    made = 0
    for index, row in enumerate(rows[: args.examples]):
        study = frame.iloc[index]
        with Image.open(args.image_root / study.image_path) as handle:
            image = handle.convert("L").copy()
        frame_geom = OriginalFrame(width=image.width, height=image.height)
        canvas = frame_geom.canvas(max_side=512)
        base = np.asarray(image.resize((canvas[1], canvas[0]))) / 255.0

        maps = np.load(args.run_dir / row["attribution_map"])["maps"]
        sentences = [s for s in row["sentences"] if s["token_indices"]][: args.max_sentences]
        if not sentences:
            continue

        fig, axes = plt.subplots(1, len(sentences) + 1,
                                 figsize=(3.1 * (len(sentences) + 1), 3.8))
        axes = np.atleast_1d(axes)
        axes[0].imshow(base, cmap="gray"); axes[0].set_title("radiograph", fontsize=9)
        axes[0].axis("off")
        for panel, sentence in enumerate(sentences, start=1):
            cells = torch.tensor(maps[sentence["attribution_index"]]).reshape(-1)
            heat = medgemma_map_in_original(cells, frame_geom, canvas_hw=canvas).numpy()
            axes[panel].imshow(base, cmap="gray")
            axes[panel].imshow(heat, cmap="jet", alpha=0.45)
            labels = ", ".join(f"{lab['finding']}({lab['polarity'][:3]})"
                               for lab in sentence["labels"]) or "unlabelled"
            axes[panel].set_title(
                f"s{sentence['index']}  {labels}\nNLL {sentence['mean_token_nll']:.2f}",
                fontsize=7,
            )
            axes[panel].axis("off")
        # Identifier-free filename, sequential like the maps themselves.
        fig.suptitle(f"study_{index:05d}   grid {MEDGEMMA_GRID.height}x"
                     f"{MEDGEMMA_GRID.width}   labeler {row['labeler']}", fontsize=9)
        fig.tight_layout()
        fig.savefig(out / f"example_{index:05d}.png", dpi=110)
        plt.close(fig)
        made += 1

    print(f"rendered {made} example overlays to {out}")
    print("these are patient images: keep them on this host")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
