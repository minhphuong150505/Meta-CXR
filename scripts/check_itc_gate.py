"""Decide whether ITC has escaped chance, before spending 36 hours finding out.

The vision-language objectives were switched off on 2026-08-13 because they sat
at EXACTLY chance: train ``loss_itc`` at ``ln(1024)`` with a 1024 queue, val at
``ln(6)`` at eval batch 6, and ``loss_itm`` at 0.6365 — the entropy of this
implementation's 1:2 positive:negative prior, i.e. the optimum for constant
logits. They were re-enabled on 2026-08-16 because the tokens the Q-Former reads
changed materially. That is a reason to retry, not evidence it will work.

A full run now costs ~36 h (batch 8, 27,875 iters/epoch, ~0.46 s/it) against
~9 h with the objectives off. This script is the cheap gate that decides whether
that is worth spending.

WHAT IT MEASURES

One all-to-all bidirectional InfoNCE over a fixed subset, in ``eval()`` mode, with
no queue and no gradient::

    delta = ln(N) - L_itc

``ln(N)`` is the loss of a model whose logits are constant — chance. ``delta`` is
how many nats of separation the contrastive head has actually bought.

HOW TO READ IT

Run it twice: once on an untrained model and once on a checkpoint saved after
~500 optimizer updates. Proceed with the full run only if the later ``delta`` is
**>= 0.10 nats AND larger than the first**. A ``delta`` near zero means chance has
reproduced and the 36 hours buy nothing.

Do not read a single ``delta`` in isolation. An untrained Q-Former starts from a
pretrained BLIP-2 initialisation, so a small positive ``delta`` at step 0 is
expected and is not evidence of anything.

The subset is fixed and ordered, never sampled per call: comparing two runs is
the entire point, and a different subset would make the two numbers
incomparable.
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

from pretraining.itc_gate import (  # noqa: E402
    DEFAULT_PAIRS,
    MIN_DELTA_NATS,
    collect_pairs,
    model_temperature,
    score_itc,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cfg-path", required=True, type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Stage-1 checkpoint. Omit to measure the untrained initialisation.",
    )
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument(
        "--pairs",
        type=int,
        default=DEFAULT_PAIRS,
        help="Fixed subset size. Keep it identical across the runs you compare.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--min-delta",
        type=float,
        default=MIN_DELTA_NATS,
        help="Gate threshold in nats.",
    )
    parser.add_argument(
        "--oversample",
        type=float,
        default=3.0,
        help=(
            "Load pairs*oversample studies and keep the first `pairs` whose "
            "FINDINGS are usable. ~30%% of studies have none, and training masks "
            "those out of ITC -- scoring them here measures nothing."
        ),
    )
    parser.add_argument(
        "--options",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Config overrides, same syntax as pretraining/train.py. Needed to "
            "measure a variant without editing the tracked YAML, e.g. "
            "model.loss.itc_temp_learnable=False. Applied on top of the "
            "single-process overrides this script always sets."
        ),
    )
    return parser.parse_args(argv)


def _build(args):
    """Load the Stage-1 stack without importing anything training-only."""
    import torch
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader

    from local_config import VIS_ROOT
    from model.lavis import tasks
    from model.lavis.common.config import Config
    from model.lavis.common.registry import registry
    from model.lavis.data.ReportDataset import MIMIC_CXR_Dataset

    registry.mapping["paths"]["cache_root"] = "."
    cfg = Config(
        SimpleNamespace(
            cfg_path=str(args.cfg_path),
            options=[
                "run.distributed=false",
                "run.world_size=1",
                "run.gpu=0",
                *args.options,
            ],
        )
    )
    if args.checkpoint is not None:
        OmegaConf.update(cfg.config, "model.load_finetuned", True, merge=False)
        OmegaConf.update(
            cfg.config,
            "model.finetuned",
            str(args.checkpoint.expanduser().resolve()),
            merge=False,
        )
    OmegaConf.update(cfg.config, "run.feature_cache_dir", None, merge=False)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("a CUDA device was requested but CUDA is unavailable")

    model = tasks.setup_task(cfg).build_model(cfg).to(device)
    model.eval()

    dataset = MIMIC_CXR_Dataset(
        vis_processor=None,
        text_processor=None,
        vis_root=VIS_ROOT,
        split=args.split,
        cfg=cfg,
        truncate=int(args.pairs * args.oversample),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,          # the subset must be identical across runs
        num_workers=0,
        collate_fn=dataset.collater,
    )
    return cfg, model, loader, device


def main(argv=None):
    args = parse_args(argv)

    cfg, model, loader, device = _build(args)

    # Score ONLY the pairs training scores: `generation_mask` is False for a
    # study whose report has no usable FINDINGS, and `_image_text_contrastive`
    # drops those rows from the loss and masks them out of the candidate set.
    # The gate once ignored the mask; on val 29.3% of pairs were then an image
    # against an EMPTY string. The shared implementation lives in
    # pretraining/itc_gate.py so the runner's per-epoch gate is identical.
    image_features, text_features, scanned = collect_pairs(
        model, loader, args.pairs, device
    )
    n = image_features.shape[0]
    if n < args.pairs:
        raise ValueError(
            f"only {n} of {scanned} scanned studies have usable FINDINGS, "
            f"needed {args.pairs}; raise --oversample"
        )
    # Mirror the model's effective temperature (pinned or learned).
    temperature, temp_learnable = model_temperature(model)
    report = {
        "split": args.split,
        "checkpoint": str(args.checkpoint) if args.checkpoint else "(untrained)",
        "temp_learnable": temp_learnable,
        "studies_scanned": scanned,
        "valid_fraction": round(n / scanned, 4) if scanned else None,
        **score_itc(image_features, text_features, temperature, args.min_delta),
    }
    chance = report["chance_ln_n"]
    delta = report["delta_nats"]
    passed = report["meets_threshold"]

    print(json.dumps(report, indent=2))
    print()
    print(f"  L_itc  {report['loss_itc']:.4f}   chance ln({n}) = {chance:.4f}")
    print(f"  delta  {delta:+.4f} nats   (threshold {args.min_delta})")
    print(
        f"  true pair ranks {report['mean_rank_of_true_pair_i2t']:.1f} / "
        f"{report['mean_rank_of_true_pair_t2i']:.1f}  vs {report['chance_rank']} at chance"
    )
    print(
        f"  R@1 {report['R@1_i2t']:.4f} / {report['R@1_t2i']:.4f}   "
        f"R@5 {report['R@5_i2t']:.4f} / {report['R@5_t2i']:.4f}   "
        f"(chance {report['R@1_chance']:.4f} / {report['R@5_chance']:.4f}; "
        f"R@5 needs >= {report['R@5_significant_hits']} hits of {n})"
    )
    print()
    if passed:
        print("  ABOVE the threshold on this run alone.")
        print("  Still compare against the untrained measurement: the gate needs")
        print("  delta to have GROWN, not merely to be positive.")
    else:
        print("  BELOW the threshold. Chance has reproduced; a full run would")
        print("  spend ~36 h to reach the same collapsed contrastive head.")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\n  wrote {args.output}")

    # Exit code is advisory only; the decision needs two measurements.
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
