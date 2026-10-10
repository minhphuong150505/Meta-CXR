#!/usr/bin/env python3
"""Where and why do the 32 META-Former (Q-Former) soft tokens collapse?

``scripts/probe_soft_tokens_cached.py`` measured a mean cosine of +0.9995
between the 32 query outputs of one study (2026-10-11). This script loads one
Stage-1 checkpoint (or none: the BLIP-2 initialisation), runs the image-only
path ``forward_image`` on ``--limit`` studies, and records, per study:

* within-study cosine and participation ratio (effective number of distinct
  tokens) of the query states after the embeddings and after every layer;
* per cross-attention layer: normalised attention entropy (1.0 = uniform over
  the image tokens), how alike the 32 queries' attention maps are, and the
  attention mass each encoder stream receives;
* the image tokens themselves: norm and within-study cosine per stream;
* a counterfactual: the final query states when every image token is replaced
  by the study's MEAN image token. If the outputs barely move, the Q-Former
  reads only the average image -- no spatial information reaches Stage 2.

Aggregates only on stdout and in ``--report``. No identifiers, text or paths.

    python scripts/diagnose_qformer_collapse.py --checkpoint <ckpt.pth | none> \\
        --label 1c --limit 200 --report <private dir>/qformer_<label>.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (_REPO_ROOT, _REPO_ROOT / "training"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def within_cosine(x: torch.Tensor) -> float:
    """Mean off-diagonal cosine between the rows of ``x`` [T, D]."""
    n = torch.nn.functional.normalize(x.float(), dim=-1)
    s = n @ n.T
    t = x.shape[0]
    return float((s.sum() - s.diagonal().sum()) / (t * (t - 1)))


def participation_ratio(x: torch.Tensor) -> float:
    """(sum s^2)^2 / sum s^4 of the singular values: 1 = one direction, T = all distinct."""
    s = torch.linalg.svdvals(x.float())
    p = s.pow(2)
    return float(p.sum().pow(2) / p.pow(2).sum().clamp_min(1e-12))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="Stage-1 .pth, or 'none' for BLIP-2 init")
    ap.add_argument("--label", required=True)
    ap.add_argument("--config", type=Path,
                    default=_REPO_ROOT / "pretraining/configs/mimic_cxr_full.yaml")
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args(argv)

    from training import train_eval_figure9_llm_variants_200 as fig9
    from training.run_context import Stage1Context
    from training.stage1 import lavis_loader as L

    ckpt = None if args.checkpoint == "none" else Path(args.checkpoint)
    context = Stage1Context(run_name="diag", config_path=args.config, checkpoint_path=ckpt)
    cfg = L.build_cfg(context)
    model = L.tasks.setup_task(cfg).build_model(cfg)
    if ckpt is not None:
        state = L.load_torch_checkpoint(ckpt)
        state = state["model"] if isinstance(state, dict) and "model" in state else state
        state = L.filter_state_dict_for_model(model, state)
        missing, unexpected = L.load_state_dict_materializing_meta(model, state)
        q_missing = [k for k in missing if k.startswith(("Qformer", "query_tokens",
                                                          "shared_visual_projector", "ln_vision"))]
        print(f"[diag] {args.label}: loaded; missing={len(missing)} "
              f"(Q-Former/projector missing={len(q_missing)}) unexpected={len(unexpected)}",
              flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    captured: dict = {}
    original_bert_forward = model.Qformer.bert.forward

    def bert_forward(*a, **kw):
        kw["output_attentions"] = True
        kw["output_hidden_states"] = True
        out = original_bert_forward(*a, **kw)
        captured["out"] = out
        captured["image"] = kw.get("encoder_hidden_states")
        return out

    model.Qformer.bert.forward = bert_forward
    model.shared_visual_projector.register_forward_hook(
        lambda _m, _i, o: captured.__setitem__("spans", dict(o.spans)))

    q = model.query_tokens[0].detach().float()
    query_param_cos = within_cosine(q)
    print(f"[diag] {args.label}: query_tokens parameter cosine {query_param_cos:+.4f}, "
          f"norm {float(q.norm(dim=-1).mean()):.3f}", flush=True)

    loader = L.make_stage1_loader(cfg, args.split, args.limit, args.num_workers)
    acc: dict[str, list] = {}

    def add(key, value):
        acc.setdefault(key, []).append(float(value))

    pooled = []
    for batch in loader:
        inputs = {k: v.to(device) for k, v in batch.items()
                  if k in fig9.STAGE1_IMAGE_INPUT_KEYS and torch.is_tensor(v)}
        with torch.no_grad():
            model.forward_image(inputs)
            out, image, spans = captured["out"], captured["image"], captured["spans"]
            final = out.last_hidden_state[0]
            pooled.append(final.mean(0).float().cpu())
            add("final_within_cos", within_cosine(final))
            add("final_pr", participation_ratio(final))
            add("final_pr_centered", participation_ratio(final - final.mean(0, keepdim=True)))
            for layer, h in enumerate(out.hidden_states):
                add(f"layer{layer:02d}_within_cos", within_cosine(h[0, :32]))
            # Every layer appends an entry; only the cross-attention layers hold
            # a [B, heads, 32, N_image] probability tensor.
            maps = [t for t in (out.cross_attentions or ())
                    if torch.is_tensor(t) and t.ndim == 4 and t.shape[-1] == image.shape[1]
                    and t.shape[-2] == 32]
            for i, att in enumerate(maps):
                a = att[0].float()                                   # [heads, 32, N]
                n_img = a.shape[-1]
                ent = -(a.clamp_min(1e-12) * a.clamp_min(1e-12).log()).sum(-1) / math.log(n_img)
                add(f"xattn{i}_entropy", ent.mean())
                add(f"xattn{i}_max_weight", a.max(-1).values.mean())
                rows = torch.nn.functional.normalize(a, dim=-1)      # per head, per query
                sim = rows @ rows.transpose(1, 2)                    # [heads, 32, 32]
                add(f"xattn{i}_query_map_cos",
                    ((sim.sum((1, 2)) - sim.diagonal(dim1=1, dim2=2).sum(1)) / (32 * 31)).mean())
                for name, span in spans.items():
                    add(f"xattn{i}_mass_{name}", a[..., span].sum(-1).mean())
                # Which image token is the sink? Share of attention on each
                # stream's FIRST token (PubMedCLIP CLS, MedCLIP-Swin pooled token).
                for name, span in spans.items():
                    add(f"xattn{i}_first_token_{name}", a[..., span.start].mean())
                top = int(a.mean((0, 1)).argmax())
                acc.setdefault(f"xattn{i}_top_index", []).append(top)
            img = image[0].float()
            for name, span in spans.items():
                add(f"image_{name}_norm", img[span].norm(dim=-1).mean())
                add(f"image_{name}_within_cos", within_cosine(img[span]))
            add("image_all_within_cos", within_cosine(img))
            # Counterfactual: every image token := the study's mean image token.
            mean_img = img.mean(0, keepdim=True).expand_as(img).unsqueeze(0).to(image.dtype)
            cf = original_bert_forward(
                query_embeds=model.query_tokens.expand(1, -1, -1),
                encoder_hidden_states=mean_img,
                encoder_attention_mask=torch.ones(mean_img.shape[:-1], dtype=torch.long,
                                                  device=device),
                return_dict=True,
            ).last_hidden_state[0]
            cos_cf = torch.nn.functional.cosine_similarity(cf.float(), final.float(), dim=-1)
            add("meanimage_vs_real_token_cos", cos_cf.mean())
            add("meanimage_rel_change",
                (cf.float() - final.float()).norm() / final.float().norm().clamp_min(1e-12))

    p = torch.nn.functional.normalize(torch.stack(pooled), dim=-1)
    s = p @ p.T
    n = p.shape[0]
    across = float((s.sum() - s.diagonal().sum()) / (n * (n - 1)))
    summary = {}
    for k, v in sorted(acc.items()):
        if k.endswith("_top_index"):
            idx, cnt = Counter(v).most_common(1)[0]
            summary[f"{k}_mode"] = int(idx)
            summary[f"{k}_mode_share"] = cnt / len(v)
            summary[f"{k}_distinct"] = len(set(v))
        else:
            summary[k] = float(np.mean(v))
    summary.update({"label": args.label, "n_studies": n, "query_param_cos": query_param_cos,
                    "pooled_across_studies_cos": across})
    keys = ["final_within_cos", "final_pr", "final_pr_centered", "pooled_across_studies_cos",
            "meanimage_vs_real_token_cos", "meanimage_rel_change", "image_all_within_cos"]
    print(f"[diag] {args.label}: " + "  ".join(f"{k}={summary[k]:.4f}" for k in keys), flush=True)
    print(f"[diag] {args.label}: within-cos by layer: " + " ".join(
        f"{summary[k]:.3f}" for k in sorted(summary) if k.endswith("_within_cos")
        and k.startswith("layer")), flush=True)
    for i in range(6):
        if f"xattn{i}_entropy" in summary:
            masses = " ".join(f"{name}={summary[f'xattn{i}_mass_{name}']:.3f}"
                              for name in ("biovil", "pubmedclip", "swin")
                              if f"xattn{i}_mass_{name}" in summary)
            firsts = " ".join(f"{name}[0]={summary[f'xattn{i}_first_token_{name}']:.3f}"
                              for name in ("biovil", "pubmedclip", "swin")
                              if f"xattn{i}_first_token_{name}" in summary)
            print(f"[diag] {args.label}: xattn{i} entropy={summary[f'xattn{i}_entropy']:.4f} "
                  f"max_w={summary[f'xattn{i}_max_weight']:.4f} "
                  f"query_map_cos={summary[f'xattn{i}_query_map_cos']:.4f} {masses} | {firsts} "
                  f"top_index mode={summary[f'xattn{i}_top_index_mode']} "
                  f"share={summary[f'xattn{i}_top_index_mode_share']:.2f} "
                  f"distinct={summary[f'xattn{i}_top_index_distinct']}", flush=True)
    print(f"[diag] {args.label}: image tokens " + " ".join(
        f"{name}: norm={summary[f'image_{name}_norm']:.2f} cos={summary[f'image_{name}_within_cos']:.3f}"
        for name in ("biovil", "pubmedclip", "swin") if f"image_{name}_norm" in summary),
        flush=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                               encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
