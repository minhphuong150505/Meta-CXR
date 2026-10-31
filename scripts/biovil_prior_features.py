#!/usr/bin/env python3
"""Swin + BioViL-T features with and without the patient's previous image, for the prior probe.

Plan and reading rule: ``docs/handoff/PLAN-2026-10-31-biovil-prior-probe.md``. BioViL-T is a
two-image encoder: its 512-d output per cell is 256 "static" channels of the current image plus
256 "difference" channels from a temporal transformer over current + previous image. Without a
previous image the difference half is the learned constant ``missing_previous_emb`` -- which is
what the pipeline always runs. This script feeds the previous image where one exists.

Same studies, order, labels and keys as ``scripts/encoder_probe_features.py`` (anchor only,
clean, seeded ``--limit`` subset). Writes ``swin_*``, ``biovil_*`` (no prior, as the encoder
probe), ``biovilprior_*`` (prior where available, else equal to ``biovil_*``) -- ``*_global``
``[N, D]`` and ``*_grid`` ``[N, 49, D]`` float16 -- plus ``has_prior.npy``, ``prior_days.npy``,
``labels.npy``, ``keys.npy``. Derived features of MIMIC-CXR: keep them on the training host.
Prints counts only.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (_REPO_ROOT, _REPO_ROOT / "training", _REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

FRONTAL_RANK = {"PA": 0, "AP": 1}


def prior_frontal_table(split, times):
    """Per study of ``split``: the patient's latest STRICTLY earlier study that has a frontal
    image, that study's frontal image (PA before AP, then the smallest ``dicom_id``), and the gap.

    ``split`` has ``subject_id, study_id, dicom_id, ViewPosition, image_path``; ``times`` maps
    ``study_id`` -> Timestamp. Earlier studies at the same instant: the larger ``study_id``. Returns a frame indexed by ``study_id`` with ``prior_study_id``,
    ``prior_image_path`` (NaN when none) and ``prior_days``.
    """
    import pandas as pd

    studies = split.drop_duplicates("study_id")[["subject_id", "study_id"]].copy()
    studies["t"] = studies["study_id"].map(times)
    if studies["t"].isna().any():
        raise ValueError(f"{int(studies['t'].isna().sum())} studies have no StudyDate")
    frontal = split[split["ViewPosition"].isin(FRONTAL_RANK)].copy()
    frontal["rank"] = frontal["ViewPosition"].map(FRONTAL_RANK)
    frontal = frontal.sort_values(["study_id", "rank", "dicom_id"]).drop_duplicates("study_id")
    cand = frontal[["subject_id", "study_id", "image_path"]].rename(
        columns={"study_id": "prior_study_id", "image_path": "prior_image_path"})
    cand["prior_t"] = cand["prior_study_id"].map(times)
    # Two candidates at the same instant: merge_asof takes the last, i.e. the larger study_id.
    merged = pd.merge_asof(studies.sort_values("t"), cand.sort_values(["prior_t", "prior_study_id"]), left_on="t",
                           right_on="prior_t", by="subject_id", allow_exact_matches=False,
                           direction="backward")
    # allow_exact_matches=False: strictly earlier, so never the study itself or a later one.
    merged["prior_days"] = (merged["t"] - merged["prior_t"]).dt.days
    return merged.set_index("study_id")[["prior_study_id", "prior_image_path", "prior_days"]]


def prior_lookup(split, times, vis_root):
    """dicom stem of an anchor -> (absolute prior image path or None, gap in days or -1).

    A prior is offered only when the anchor itself is frontal (BioViL-T learned frontal pairs).
    """
    table = prior_frontal_table(split, times)
    rows = split.drop_duplicates("dicom_id").set_index("dicom_id")

    def lookup(dicom):
        row = rows.loc[dicom]
        if row["ViewPosition"] not in FRONTAL_RANK:
            return None, -1
        prior = table.loc[row["study_id"]]
        if not isinstance(prior["prior_image_path"], str):
            return None, -1
        return os.path.join(vis_root, prior["prior_image_path"]), int(prior["prior_days"])

    return lookup


class _WithPrior:
    """Adds ``prior_image`` (BioViL tensor of the previous image, zeros if none), ``has_prior``
    and ``prior_days`` to a dataset item, through the dataset's own BioViL transform."""

    def __init__(self, base, indices, lookup):
        self.base, self.indices, self.lookup = base, indices, lookup

    def __len__(self):
        return len(self.indices)

    def biovil_tensor(self, path):
        return self.base.optical_trans(self.base.geometric_trans(self.base.load_image(Path(path))))

    def __getitem__(self, i):
        import torch

        sample = self.base[self.indices[i]]
        path, days = self.lookup(Path(sample["image_path"]).stem)
        sample["prior_image"] = self.biovil_tensor(path) if path else torch.zeros_like(sample["image"])
        sample["has_prior"] = path is not None
        sample["prior_days"] = days
        return sample


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=("train", "val", "test"))
    ap.add_argument("--limit", type=int, default=None, help="seeded random subset of studies")
    ap.add_argument("--seed", type=int, default=16)
    ap.add_argument("--config", type=Path, default=_REPO_ROOT / "pretraining/configs/mimic_cxr_full.yaml")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.split == "test":
        raise SystemExit("the prior probe reads train and val only (plan section 2.2)")

    import pandas as pd
    import torch
    from encoder_probe_features import grid_tokens
    from torch.utils.data import DataLoader

    from local_config import METADATA_CSV, PROCESSED_TRAIN_CSV, PROCESSED_VAL_CSV
    from training.dataio.study_context import study_times
    from training.evaluation.schemas import build_sample_keys
    from training.run_context import Stage1Context
    from training.stage1 import lavis_loader as L

    args.out_dir.mkdir(parents=True, exist_ok=False)
    cfg = L.build_cfg(Stage1Context(run_name="biovilprior", config_path=args.config, checkpoint_path=None))
    cfg.datasets_cfg.mimic_cxr.vis_processor.train.augmentation.enabled = False
    cfg.model_cfg.data.max_aux_views = 0
    cfg.model_cfg.explanation.mask_cache_dir = None
    model = L.tasks.setup_task(cfg).build_model(cfg)
    device = torch.device("cuda")
    model.to(device).eval()
    encoder = model.visual_encoder.encoder

    dataset = L.MIMIC_CXR_Dataset(vis_processor=None, text_processor=None, vis_root=L.VIS_ROOT,
                                  split=args.split, cfg=cfg)
    if getattr(dataset, "augmentation_enabled", False):
        raise RuntimeError("training augmentation is still on")
    split = pd.read_csv(PROCESSED_TRAIN_CSV if args.split == "train" else PROCESSED_VAL_CSV,
                        usecols=["subject_id", "study_id", "dicom_id", "ViewPosition", "image_path"])
    order = np.arange(len(dataset))
    if args.limit and args.limit < len(order):
        order = np.sort(np.random.default_rng(args.seed).choice(order, args.limit, replace=False))
    n = len(order)
    wrapped = _WithPrior(dataset, order.tolist(), prior_lookup(split, study_times(METADATA_CSV), L.VIS_ROOT))
    for i in range(4):  # check (a): the prior transform is the dataset's own BioViL transform
        s = dataset[int(order[i])]
        if not torch.equal(wrapped.biovil_tensor(s["image_path"]), s["image"]):
            raise SystemExit("CHECK (a) FAILED: the prior transform differs from the dataset's image tensor")
    print(f"[biovilprior] {args.split}: {n} of {len(dataset)} studies; check (a) passed", flush=True)
    loader = DataLoader(wrapped, batch_size=args.batch, num_workers=args.workers, collate_fn=dataset.collater)

    store: dict[str, np.ndarray] = {}

    def put(name, pos, value):
        value = value.float().cpu().numpy().astype(np.float16)
        if name not in store:
            store[name] = np.lib.format.open_memmap(args.out_dir / f"{name}.npy", mode="w+",
                                                    dtype=np.float16, shape=(n,) + value.shape[1:])
        store[name][pos:pos + len(value)] = value

    labels = np.full((n, 14), -100, dtype=np.int16)
    has_prior = np.zeros(n, dtype=bool)
    prior_days = np.full(n, -1, dtype=np.int32)
    keys: list[str] = []
    static_cos, diff_change = [], []
    pos, started = 0, time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for b, batch in enumerate(loader):
            bs = batch["image"].shape[0]
            lab = batch["classification_labels"].numpy().astype(np.int16)
            lab[~batch["classification_mask"].numpy().astype(bool)] = -100
            labels[pos:pos + bs] = lab
            ids = batch["image_id"]
            keys.extend(build_sample_keys([str(v) for v in (ids.tolist() if torch.is_tensor(ids) else ids)]))
            m = batch["has_prior"].bool()
            has_prior[pos:pos + bs] = m.numpy()
            prior_days[pos:pos + bs] = batch["prior_days"].numpy()

            cur = batch["image"].to(device)
            patch0, pooled0 = encoder(cur, None, return_patch_embeddings=True)  # [B, 512, 14, 14]
            patch1, pooled1 = patch0.clone(), pooled0.clone()
            if m.any():
                md = m.to(device)
                p, g = encoder(cur[md], batch["prior_image"].to(device)[md], return_patch_embeddings=True)
                static_cos.append(torch.nn.functional.cosine_similarity(
                    p[:, :256].float().flatten(1), patch0[md][:, :256].float().flatten(1), dim=1).cpu())
                diff_change.append((p[:, 256:] - patch0[md][:, 256:]).float().abs().mean((1, 2, 3)).cpu())
                patch1[md], pooled1[md] = p, g
            put("biovil_global", pos, pooled0)
            put("biovil_grid", pos, grid_tokens(patch0.flatten(2).transpose(1, 2), 14, 7))
            put("biovilprior_global", pos, pooled1)
            put("biovilprior_grid", pos, grid_tokens(patch1.flatten(2).transpose(1, 2), 14, 7))
            sw = model.swin(model._swin_input(None, batch["swin_image"].to(device)))
            put("swin_global", pos, sw[:, 0])
            put("swin_grid", pos, sw[:, 1:])
            pos += bs
            if (b + 1) % 100 == 0:
                print(f"[biovilprior] {pos}/{n} {pos / (time.time() - started):.1f} study/s", flush=True)
    assert pos == n, (pos, n)
    np.save(args.out_dir / "labels.npy", labels)
    np.save(args.out_dir / "keys.npy", np.array(keys))
    np.save(args.out_dir / "has_prior.npy", has_prior)
    np.save(args.out_dir / "prior_days.npy", prior_days)
    for arr in store.values():
        arr.flush()
    cos = torch.cat(static_cos) if static_cos else torch.ones(1)
    chg = torch.cat(diff_change) if diff_change else torch.zeros(1)
    days = prior_days[has_prior]
    print(f"[biovilprior] with prior {has_prior.sum()} of {n} ({has_prior.mean():.3f}); gap days median "
          f"{np.median(days) if len(days) else float('nan'):.0f}; static cosine min {float(cos.min()):.5f}; "
          f"difference-channel mean |change| {float(chg.mean()):.4f}", flush=True)
    print(f"[biovilprior] labelled studies {(labels >= 0).any(1).sum()} of {n}; "
          f"{time.time() - started:.0f} s", flush=True)
    if has_prior.any() and (float(cos.min()) < 0.999 or float(chg.mean()) <= 0):
        raise SystemExit("CHECK (b) FAILED: the prior changed the static half, or left the difference half")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
