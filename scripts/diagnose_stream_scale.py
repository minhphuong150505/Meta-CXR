#!/usr/bin/env python3
"""Measure the per-encoder scale imbalance in the shared visual token sequence.

READ-ONLY DIAGNOSTIC. It loads a trained Stage-1 checkpoint, runs a forward pass
over a handful of studies, and reports two numbers per encoder:

  1. **Token magnitude at the concatenation point** — RMS of each encoder's span
     in the tensor ``SharedVisualTokenProjector`` returns. That tensor is what
     MHCAC and the Q-Former both read, so it is where a scale imbalance would
     actually bite.
  2. **MHCAC cross-attention mass** — the fraction of ``expert_to_image_attention``
     probability that lands on each encoder's patches, per layer.

Why this exists: ``vision_encoders/shared_visual_tokens.py`` projects each stream
to ``visual_dim`` and concatenates, with **no per-stream normalisation**. Only
BioViL-T is LayerNormed, and upstream of the merge
(``blip2_qformer.py``: ``raw_streams["biovil"] = self.ln_vision(cnn_raw)``).
PubMedCLIP and SwinV2 enter the shared sequence at whatever magnitude their own
encoder emits, scaled by their projection. If those magnitudes differ by an order
of magnitude, MHCAC's softmax concentrates on the loudest span regardless of what
the others contain — which would explain why the Table 5 ablation found
all-three ≈ BioViL-only.

This script changes no logic, trains nothing, and writes no checkpoint. It only
reads. Interpretation lives in the caller, not here.

Run it on the training host, not on a dev checkout::

    ssh phuong@minhphuong
    cd ~/Documents/2026/KLTN/Code_github/META-CXR-full-smoke-git
    git pull origin main
    CUDA_VISIBLE_DEVICES=0 python scripts/diagnose_stream_scale.py --num-studies 32
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CFG = "pretraining/configs/ablation/all_three.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Report per-encoder token magnitude and MHCAC attention mass.",
    )
    p.add_argument(
        "--cfg-path",
        default=DEFAULT_CFG,
        help=(
            "Stage-1 config naming the checkpoint to load. Default is the Table 5 "
            f"all-three reference ({DEFAULT_CFG}), which sets load_finetuned and "
            "leaves every encoder active."
        ),
    )
    p.add_argument("--split", default="val", choices=["train", "val", "test"],
                   help="Split to sample studies from. Default val — test stays untouched.")
    p.add_argument("--num-studies", type=int, default=32,
                   help="How many studies to measure. A few dozen is plenty; this is a scale check.")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default=None,
                   help="Defaults to cuda when available, else cpu.")
    p.add_argument("--json-out", default=None,
                   help="Optional path to write the measurements as JSON.")
    p.add_argument(
        "--options", nargs="+", default=None,
        help="Extra LAVIS config overrides, same syntax as pretraining/train.py.",
    )
    return p.parse_args()


def build_model_and_dataset(args):
    """Load model + dataset through the same path pretraining/train.py uses."""
    import torch

    from model.lavis.common.registry import registry

    # train.py sets this before Config(); builders resolve relative paths against it.
    registry.mapping["paths"]["cache_root"] = "."

    from local_config import VIS_ROOT
    from model.lavis import tasks
    from model.lavis.common.config import Config
    from model.lavis.data.ReportDataset import MIMIC_CXR_Dataset

    # Single process, no DDP. Mirrors precompute_features.py, which is also
    # standalone and therefore has to set run.gpu explicitly.
    options = ["run.distributed=false", "run.world_size=1", "run.gpu=0"]
    if args.options:
        options.extend(args.options)

    cfg = Config(SimpleNamespace(cfg_path=args.cfg_path, options=options))

    task = tasks.setup_task(cfg)
    model = task.build_model(cfg)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    dataset = MIMIC_CXR_Dataset(
        vis_processor=None,
        text_processor=None,
        vis_root=VIS_ROOT,
        split=args.split,
        cfg=cfg,
        truncate=args.num_studies,
    )
    return model, dataset, device


def main() -> int:
    args = parse_args()

    import torch
    from torch.utils.data import DataLoader

    model, dataset, device = build_model_and_dataset(args)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=dataset.collater,
    )

    # sum of squares and element count per stream, so the RMS is over every
    # token of every study rather than a mean of per-batch means.
    sq_sum: dict[str, float] = {}
    n_elem: dict[str, int] = {}
    abs_max: dict[str, float] = {}
    span_width: dict[str, int] = {}
    # attention mass per (layer, stream)
    attn_mass: list[dict[str, float]] = []
    n_batches = 0
    n_studies = 0

    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            aux_image = batch.get("aux_image")
            if aux_image is not None:
                aux_image = aux_image.to(device)
            aux_mask = batch.get("aux_mask")
            if aux_mask is not None:
                aux_mask = aux_mask.to(device)
            anchor_view_id = batch.get("anchor_view_id")
            if anchor_view_id is not None:
                anchor_view_id = anchor_view_id.to(device)
            aux_view_ids = batch.get("aux_view_ids")
            if aux_view_ids is not None:
                aux_view_ids = aux_view_ids.to(device)

            # Private call on purpose: this is the exact tensor the two downstream
            # branches consume, and reproducing the encode here would risk
            # measuring something the model never sees. Read-only, eval mode.
            shared = model._encode_image_streams(
                image,
                apply_aug=False,
                aux_image=aux_image,
                aux_mask=aux_mask,
                anchor_view_id=anchor_view_id,
                aux_view_ids=aux_view_ids,
            )

            for name in sorted(shared.spans, key=lambda k: shared.spans[k].start):
                tokens = shared.stream(name).float()
                sq_sum[name] = sq_sum.get(name, 0.0) + float(tokens.pow(2).sum())
                n_elem[name] = n_elem.get(name, 0) + tokens.numel()
                abs_max[name] = max(abs_max.get(name, 0.0), float(tokens.abs().max()))
                span_width[name] = int(tokens.shape[1])

            # MHCAC returns attention weights as its second value. Passing no text
            # and no labels keeps this on the student path, which is what runs at
            # inference; no loss is computed.
            _, attn_list, *_ = model.mhcac(shared, text_embeddings=None, labels=None)

            stream_order = sorted(shared.spans, key=lambda k: shared.spans[k].start)
            if not attn_mass:
                attn_mass = [{} for _ in attn_list]
            for layer_idx, attn in enumerate(attn_list):
                # [B, num_expert_tokens, num_patches]; patches are the per-stream
                # sequences after each was resized to target_patch_count, then
                # concatenated in span order.
                per_stream = attn.float().chunk(len(stream_order), dim=-1)
                for name, chunk in zip(stream_order, per_stream, strict=True):
                    attn_mass[layer_idx][name] = (
                        attn_mass[layer_idx].get(name, 0.0) + float(chunk.sum())
                    )

            n_batches += 1
            n_studies += int(image.shape[0])

    if not n_batches:
        print("No batches produced — check the split and truncate settings.", file=sys.stderr)
        return 1

    rms = {name: (sq_sum[name] / n_elem[name]) ** 0.5 for name in sq_sum}
    ordered = sorted(rms, key=lambda k: -rms[k])
    loudest = ordered[0]

    print(f"\nstudies measured: {n_studies}   split: {args.split}   device: {device}")
    print(f"checkpoint config: {args.cfg_path}")

    print("\n=== token magnitude at the concatenation point ===")
    print(f"{'stream':<14}{'span tokens':>12}{'RMS':>12}{'max|x|':>12}{'RMS ratio':>12}")
    for name in ordered:
        print(f"{name:<14}{span_width[name]:>12}{rms[name]:>12.4f}"
              f"{abs_max[name]:>12.4f}{rms[name] / rms[loudest]:>12.4f}")
    print(f"\nratio is relative to the loudest stream ({loudest}).")

    print("\n=== MHCAC expert->image attention mass (fraction per stream) ===")
    print(f"{'layer':<8}" + "".join(f"{name:>14}" for name in ordered))
    attn_fracs = []
    for layer_idx, layer_mass in enumerate(attn_mass):
        total = sum(layer_mass.values())
        fracs = {name: (layer_mass[name] / total if total else float("nan"))
                 for name in ordered}
        attn_fracs.append(fracs)
        print(f"{layer_idx:<8}" + "".join(f"{fracs[name]:>14.4f}" for name in ordered))
    n_streams = len(ordered)
    print(f"\nuniform attention would be {1.0 / n_streams:.4f} per stream "
          f"({n_streams} streams, equal patch counts).")

    if args.json_out:
        payload = {
            "split": args.split,
            "num_studies": n_studies,
            "cfg_path": args.cfg_path,
            "span_tokens": span_width,
            "token_rms": rms,
            "token_abs_max": abs_max,
            "rms_ratio_to_loudest": {n: rms[n] / rms[loudest] for n in rms},
            "loudest_stream": loudest,
            "attention_fraction_per_layer": attn_fracs,
        }
        Path(args.json_out).write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
