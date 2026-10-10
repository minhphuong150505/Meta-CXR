#!/usr/bin/env python3
"""Mean-pooled MedGemma image embeddings for the studies of a cached Stage-2 record file.

Diagnostic input for ``scripts/retrieval_report_baselines.py --embedding``: what
MedGemma's OWN vision path (MedSigLIP + projector, the 256 tokens its language
model reads in ``medgemma_direct``) carries about the report, measured the same
way as the Q-Former soft tokens. Selects the same seeded subset as the
retrieval script (``--limit``/``--seed``), so the databases match.

Writes ``<out>.npz`` with ``keys`` (image basename) and ``emb`` [N, hidden]
float16 -- derived features, private, stays on the training host.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=16)
    ap.add_argument("--model", default="google/medgemma-1.5-4b-it")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForImageTextToText, AutoProcessor

    records = torch.load(args.cache, map_location="cpu", weights_only=False, mmap=True)["records"]
    order = np.arange(len(records))
    if args.limit and args.limit < len(order):
        order = np.sort(np.random.default_rng(args.seed).choice(order, args.limit, replace=False))
    paths = [str(records[i]["image_path"]) for i in order]
    keys = [p.rsplit("/", 1)[-1] for p in paths]
    del records

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    model.to("cuda").eval()

    class Images(Dataset):
        def __len__(self):
            return len(paths)

        def __getitem__(self, i):
            image = Image.open(paths[i]).convert("RGB")
            return processor.image_processor(image, return_tensors="pt")["pixel_values"][0]

    loader = DataLoader(Images(), batch_size=args.batch, num_workers=args.workers)
    out, t0 = [], time.time()
    with torch.no_grad():
        for b, pixels in enumerate(loader):
            feats = model.get_image_features(pixel_values=pixels.to("cuda", torch.bfloat16))
            out.append(feats.float().mean(1).cpu().numpy().astype(np.float16))
            if (b + 1) % 200 == 0:
                done = (b + 1) * args.batch
                print(f"[embed] {done}/{len(paths)} {done / (time.time() - t0):.1f} img/s",
                      flush=True)
    emb = np.concatenate(out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, keys=np.array(keys), emb=emb)
    print(f"[embed] wrote {emb.shape} in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
