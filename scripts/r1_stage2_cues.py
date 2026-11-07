#!/usr/bin/env python3
"""Stage-2 cues from OFF + R1 for the PL studies of one split (val), for ``generate_stage2_reports.py --cue-override``.

Plan and reading rule: ``docs/handoff/PLAN-2026-11-04-residual-lateral.md``, section "R1 vào giai đoạn 2 -- bước A"
(user-approved 2026-11-07). Inference-only probe: the Stage-2 adapters trained on OFF keep their soft tokens; only the
P/N/U cues of PL studies change.

* R1 = B2a setup: L only, 3 seeds, fitted on train PL with out-of-fold experts (``residual_stack.r1_trainfit``), scored
  on the split's PL studies; the cue probabilities are the MEAN over the 3 seeds.
* PL cues: headline cutpoints (weighted F1) fitted on the split's PL rows under OFF + R1, applied to those rows.
* ``off_codes``: OFF's own cutpoint file applied to OFF's probabilities, every study -- what the existing OFF records
  should already carry; ``generate_stage2_reports.py`` checks its records against it.
* Keys: ``blake2b(anchor dicom_id, 12)``, exactly ``generate_stage2_reports.sample_key_for``. The dicom ids are read
  from the dataset's own annotation, in memory only; the npz holds digests, codes and probabilities. It is still a
  MIMIC-CXR derivative and stays on the training host.

Codes: 0 Negative, 1 Positive, 2 Uncertain, per finding in ``names`` order.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from hashlib import blake2b
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rs = _load("residual_stack")
pc = _load("residual_pooled_calibration")


def gen_key(dicom_id: str) -> str:
    """``generate_stage2_reports.sample_key_for`` of an image whose filename stem is ``dicom_id``."""
    return blake2b(str(dicom_id).encode(), digest_size=12).hexdigest()


def pl_cues(p_off: np.ndarray, y: np.ndarray, pl: np.ndarray, r1_pl: np.ndarray, names, off_cut: dict,
            grid: int = 121) -> tuple[np.ndarray, np.ndarray, dict]:
    """``(off_codes [n, F], new_codes [n, F], r1_cutpoints)``; new = off outside PL, R1 cutpoints on R1 inside."""
    from training.evaluation.threshold_calibration import apply_cutpoints, fit_severity_cutpoints

    names = tuple(names)
    off_codes = apply_cutpoints(p_off, off_cut, names)
    cut = fit_severity_cutpoints(SimpleNamespace(probabilities=r1_pl, labels=y[pl], pathology_names=names),
                                 grid_size=grid)
    new_codes = off_codes.copy()
    new_codes[pl] = apply_cutpoints(r1_pl, cut, names)
    return off_codes, new_codes, cut


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--off", type=Path, required=True, help="OFF predictions npz of the split (val)")
    ap.add_argument("--view-map", type=Path, required=True)
    ap.add_argument("--expert", required=True, help="L=seed1(a+b),... split experts (fold models averaged)")
    ap.add_argument("--expect-counts", default="PL=797,F1=930,other=81")
    ap.add_argument("--train-off", type=Path, required=True)
    ap.add_argument("--train-expert", required=True, help="L=seed1(f0_oof+f1_oof),... out-of-fold train experts")
    ap.add_argument("--off-cutpoints", type=Path, required=True, help="OFF's cutpoint file (its Stage-2 cues)")
    ap.add_argument("--split", default="val", choices=("val",))
    ap.add_argument("--config", type=Path, default=_REPO_ROOT / "pretraining/configs/mimic_cxr_full.yaml")
    ap.add_argument("--out", type=Path, required=True, help="cue npz")
    args = ap.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")

    from training.evaluation.schemas import ClassificationPredictions
    from training.evaluation.threshold_calibration import load_cutpoint_file
    from training.evaluation.view_keys import DEFAULT_PRIORITY, ViewMap, align, build_view_map

    p_va, y_va, pl, _groups, ex_va = pc.load_split(args.off, args.view_map, args.expert, args.expect_counts)
    names = tuple(ClassificationPredictions.load(args.off).pathology_names)

    toff = ClassificationPredictions.load(args.train_off)
    p_tr = np.asarray(toff.probabilities, dtype=np.float64)
    y_tr = np.asarray(toff.labels, dtype=np.int64)
    y_tr[y_tr < 0] = -1
    _n, tspecs = args.train_expert.split("=", 1)
    ex_tr, tr_keys = [], None
    for spec in tspecs.split(","):
        keys, probs, labels = rs.load_expert_parts(spec, "concat")
        if tr_keys is None:
            tr_keys = keys
            o_tr = align(toff.sample_keys, tr_keys)
        idx = align(keys, tr_keys)
        if not np.array_equal(labels[idx], y_tr[o_tr]):
            raise SystemExit(f"labels of {spec} differ from OFF's train labels")
        ex_tr.append(probs[idx])
    r1_seeds = [rs.r1_trainfit(p_tr[o_tr], [e_tr], y_tr[o_tr], p_va[pl], [e_va])[0]
                for e_va, e_tr in zip(ex_va, ex_tr, strict=True)]
    r1_pl = np.mean(r1_seeds, 0)

    off_cut, _meta = load_cutpoint_file(args.off_cutpoints)
    off_codes, new_codes, r1_cut = pl_cues(p_va, y_va, pl, r1_pl, names, off_cut)

    # anchor dicom ids from the dataset's own annotation (same construction as lateral_lora_cache.py)
    from training.run_context import Stage1Context
    from training.stage1 import lavis_loader as L

    cfg = L.build_cfg(Stage1Context(run_name="r1_cues", config_path=args.config, checkpoint_path=None))
    cfg.datasets_cfg.mimic_cxr.vis_processor.train.augmentation.enabled = False
    cfg.model_cfg.data.max_aux_views = 1
    cfg.model_cfg.data.anchor_priority = list(DEFAULT_PRIORITY)
    cfg.model_cfg.explanation.mask_cache_dir = None
    dataset = L.MIMIC_CXR_Dataset(vis_processor=None, text_processor=None, vis_root=L.VIS_ROOT,
                                  split=args.split, cfg=cfg)
    rows = dataset.annotation[["subject_id", "study_id", "ViewPosition"]].to_dict("records")
    vmap = build_view_map(rows)
    if not np.array_equal(vmap.frontal_keys, ViewMap.load(args.view_map).frontal_keys):
        raise SystemExit("rebuilt view map differs from --view-map")
    dicoms = dataset.annotation["dicom_id"].to_numpy()[vmap.frontal_rows]
    keys = np.array([gen_key(d) for d in dicoms])
    if len(set(keys.tolist())) != len(keys):
        raise SystemExit("generation keys are not unique")

    changed = new_codes != off_codes
    np.savez(args.out, keys=keys, names=np.array(names), is_pl=np.isin(np.arange(len(keys)), pl),
             off_codes=off_codes, new_codes=new_codes, off_probs=p_va)
    summary = {"n": int(len(keys)), "n_pl": int(len(pl)), "changed_cells": int(changed.sum()),
               "changed_studies": int(changed.any(1).sum()),
               "changed_cells_outside_pl": int(changed[~np.isin(np.arange(len(keys)), pl)].sum()),
               "r1_cutpoints": {k: [float(a), float(b)] for k, (a, b) in r1_cut.items()}}
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"[r1-cues] {summary['n']} studies, PL {summary['n_pl']}; cue cells changed {summary['changed_cells']} "
          f"in {summary['changed_studies']} studies (outside PL {summary['changed_cells_outside_pl']})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
