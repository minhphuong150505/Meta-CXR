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
import math
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_PAIRS = 256
MIN_DELTA_NATS = 0.10


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
            options=["run.distributed=false", "run.world_size=1", "run.gpu=0"],
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
        truncate=args.pairs,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,          # the subset must be identical across runs
        num_workers=0,
        collate_fn=dataset.collater,
    )
    return cfg, model, loader, device


def _features(model, samples, device):
    """Return the (image, text) features ITC compares, exactly as forward() does."""
    import torch
    import torch.nn.functional as F

    image = samples["image"].to(device)
    aux_image = samples.get("aux_image")
    if aux_image is not None:
        aux_image = aux_image.to(device)

    # Same entry point and same arguments forward() uses, so the features this
    # gate scores are the features training would have produced.
    shared_visual = model._encode_image_streams(
        image,
        apply_aug=False,
        aux_image=aux_image,
        aux_mask=samples.get("aux_mask"),
        anchor_view_id=samples.get("anchor_view_id"),
        aux_view_ids=samples.get("aux_view_ids"),
    )
    # SharedVisualTokens carries the per-encoder spans MHCAC needs; the Q-Former
    # takes the concatenated sequence, exactly as forward() does at `:1088`.
    image_embeds = shared_visual.tokens
    image_atts = torch.ones(
        image_embeds.shape[:-1], dtype=torch.long, device=device
    )
    query_tokens = model.query_tokens.expand(image_embeds.shape[0], -1, -1)
    query_output = model.Qformer.bert(
        query_embeds=query_tokens,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_atts,
        use_cache=True,
        return_dict=True,
    )
    image_features = F.normalize(
        model.vision_proj(query_output.last_hidden_state), dim=-1
    )

    text_tokens = model.tokenizer(
        samples["text_output"],   # the dataset emits text_output; text_input is commented out
        padding="max_length",
        truncation=True,
        max_length=model.max_txt_len,
        return_tensors="pt",
    ).to(device)
    text_output = model.Qformer.bert(
        text_tokens.input_ids,
        attention_mask=text_tokens.attention_mask,
        return_dict=True,
    )
    text_features = F.normalize(
        model.text_proj(text_output.last_hidden_state[:, 0]), dim=-1
    )
    return image_features, text_features


def main(argv=None):
    args = parse_args(argv)
    import torch
    import torch.nn.functional as F

    cfg, model, loader, device = _build(args)

    image_chunks, text_chunks = [], []
    with torch.no_grad():
        for samples in loader:
            img, txt = _features(model, samples, device)
            image_chunks.append(img.float().cpu())
            text_chunks.append(txt.float().cpu())

    image_features = torch.cat(image_chunks)      # [N, Q, D]
    text_features = torch.cat(text_chunks)        # [N, D]
    n = image_features.shape[0]
    if n < 2:
        raise ValueError(f"need at least 2 pairs to contrast, got {n}")

    # Same reduction as training: max over the 32 query tokens, learned
    # temperature. No queue -- this is a clean all-to-all over the subset, so
    # chance is exactly ln(n) and needs no correction for queue occupancy.
    temperature = float(model.temp.detach().clamp(min=1e-3, max=0.5).cpu())
    sim_i2t = torch.einsum("bqd,nd->bnq", image_features, text_features).amax(-1)
    sim_t2i = torch.einsum("bd,nqd->bnq", text_features, image_features).amax(-1)
    targets = torch.arange(n)
    loss_itc = 0.5 * (
        F.cross_entropy(sim_i2t / temperature, targets)
        + F.cross_entropy(sim_t2i / temperature, targets)
    )

    chance = math.log(n)
    delta = chance - float(loss_itc)
    # Rank of the true pair, averaged over both directions: a scale-free read
    # that does not depend on the learned temperature at all.
    rank_i2t = (sim_i2t > sim_i2t.diagonal()[:, None]).sum(dim=1).float().mean()
    rank_t2i = (sim_t2i > sim_t2i.diagonal()[:, None]).sum(dim=1).float().mean()

    passed = delta >= args.min_delta
    report = {
        "pairs": n,
        "split": args.split,
        "checkpoint": str(args.checkpoint) if args.checkpoint else "(untrained)",
        "temperature": round(temperature, 6),
        "loss_itc": round(float(loss_itc), 4),
        "chance_ln_n": round(chance, 4),
        "delta_nats": round(delta, 4),
        "min_delta": args.min_delta,
        "mean_rank_of_true_pair_i2t": round(float(rank_i2t), 2),
        "mean_rank_of_true_pair_t2i": round(float(rank_t2i), 2),
        "chance_rank": round((n - 1) / 2, 2),
        "meets_threshold": bool(passed),
    }

    print(json.dumps(report, indent=2))
    print()
    print(f"  L_itc  {report['loss_itc']:.4f}   chance ln({n}) = {chance:.4f}")
    print(f"  delta  {delta:+.4f} nats   (threshold {args.min_delta})")
    print(
        f"  true pair ranks {report['mean_rank_of_true_pair_i2t']:.1f} / "
        f"{report['mean_rank_of_true_pair_t2i']:.1f}  vs {report['chance_rank']} at chance"
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
