#!/usr/bin/env python3
"""R1 cue-override npz files (val and train) for the Stage-2 CLIs' ``--cue-override`` flags.

Plan and reading rules: ``docs/handoff/PLAN-2026-11-04-residual-lateral.md``, sections "R1 vào giai đoạn 2" (bước A,
bước B). The npz carries the R1 RECIPE (``training/dataio/cue_override.py`` documents the format and how it is
applied to each record's own logits):

* R1 = B2a setup: L expert only, 3 seeds; ``A``, ``B`` per seed fitted on train PL with out-of-fold experts
  (``residual_stack.r1_trainfit``). The same parameters serve both splits.
* ``l_probs``: val = mean of the two fold models per seed; train = out-of-fold predictions per seed.
* ``r1_cut``: headline cutpoints (weighted F1) fitted on VAL PL under OFF + R1 (mean of seeds); ``off_cut``: OFF's
  cutpoint file. Both splits get the same cutpoints.
* Keys: ``blake2b(anchor dicom_id, 12)``. dicom ids are read from each split's dataset annotation in memory only; the
  npz holds digests, probabilities and parameters -- still a MIMIC-CXR derivative, kept on the training host.

Also writes ``<out>_val.json`` with a check: the val cues the recipe gives on OFF's own val probabilities (counts only).
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


def cut_array(cut: dict, names) -> np.ndarray:
    return np.array([[float(cut[n][0]), float(cut[n][1])] for n in names])


def fit_r1_cutpoints(p_pl: np.ndarray, y_pl: np.ndarray, l_pl: list[np.ndarray], params: list, names,
                     grid: int = 121) -> tuple[dict, np.ndarray]:
    """Cutpoints on the PL rows under OFF + R1 (mean over seeds); returns them and the R1 probabilities."""
    from training.evaluation.threshold_calibration import fit_severity_cutpoints

    probs = np.mean([rs.r1_apply_params(p_pl, [lp], A, B) for lp, (A, B) in zip(l_pl, params, strict=True)], 0)
    cut = fit_severity_cutpoints(SimpleNamespace(probabilities=probs, labels=y_pl, pathology_names=tuple(names)),
                                 grid_size=grid)
    return cut, probs


def split_keys(split: str, config: Path, expected_frontal: np.ndarray):
    """Generation keys of every study of ``split`` in dataset order (the order of ``expected_frontal``)."""
    from training.evaluation.view_keys import DEFAULT_PRIORITY, build_view_map
    from training.run_context import Stage1Context
    from training.stage1 import lavis_loader as L

    cfg = L.build_cfg(Stage1Context(run_name="r1_cues", config_path=config, checkpoint_path=None))
    cfg.datasets_cfg.mimic_cxr.vis_processor.train.augmentation.enabled = False
    cfg.model_cfg.data.max_aux_views = 1
    cfg.model_cfg.data.anchor_priority = list(DEFAULT_PRIORITY)
    cfg.model_cfg.explanation.mask_cache_dir = None
    dataset = L.MIMIC_CXR_Dataset(vis_processor=None, text_processor=None, vis_root=L.VIS_ROOT, split=split, cfg=cfg)
    rows = dataset.annotation[["subject_id", "study_id", "ViewPosition"]].to_dict("records")
    vmap = build_view_map(rows)
    if not np.array_equal(vmap.frontal_keys, expected_frontal):
        raise SystemExit(f"{split}: rebuilt view map differs from the predictions' study order")
    keys = np.array([gen_key(d) for d in dataset.annotation["dicom_id"].to_numpy()[vmap.frontal_rows]])
    if len(set(keys.tolist())) != len(keys):
        raise SystemExit(f"{split}: generation keys are not unique")
    return keys, vmap


def write_npz(path: Path, keys, pl_rows: np.ndarray, names, l_pl: list[np.ndarray], params, off_cut, r1_cut):
    pl_row = np.full(len(keys), -1, dtype=np.int64)
    pl_row[pl_rows] = np.arange(len(pl_rows))
    np.savez(path, keys=keys, pl_row=pl_row, names=np.array(names),
             l_probs=np.stack(l_pl).astype(np.float32),
             A=np.stack([A[:, 0] for A, _B in params]), B=np.stack([B for _A, B in params]),
             off_cut=cut_array(off_cut, names), r1_cut=cut_array(r1_cut, names))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--off", type=Path, required=True, help="OFF val predictions npz")
    ap.add_argument("--view-map", type=Path, required=True, help="val view map")
    ap.add_argument("--expert", required=True, help="L=seed1(a+b),... val experts (fold models averaged)")
    ap.add_argument("--expect-counts", default="PL=797,F1=930,other=81")
    ap.add_argument("--train-off", type=Path, required=True, help="OFF train predictions npz (B0)")
    ap.add_argument("--train-expert", required=True, help="L=seed1(f0_oof+f1_oof),... out-of-fold train experts")
    ap.add_argument("--off-cutpoints", type=Path, required=True, help="OFF's cutpoint file (its Stage-2 cues)")
    ap.add_argument("--config", type=Path, default=_REPO_ROOT / "pretraining/configs/mimic_cxr_full.yaml")
    ap.add_argument("--out-prefix", type=Path, required=True, help="writes <prefix>_val.npz and <prefix>_train.npz")
    args = ap.parse_args(argv)
    outs = [Path(f"{args.out_prefix}_{s}.npz") for s in ("val", "train")]
    if any(o.exists() for o in outs):
        raise SystemExit(f"refusing to overwrite {outs}")

    from training.evaluation.schemas import ClassificationPredictions
    from training.evaluation.threshold_calibration import apply_cutpoints, load_cutpoint_file
    from training.evaluation.view_keys import ViewMap, align

    # ---- val
    p_va, y_va, pl_va, _g, ex_va = pc.load_split(args.off, args.view_map, args.expert, args.expect_counts)
    names = tuple(ClassificationPredictions.load(args.off).pathology_names)

    # ---- train: OFF (B0) in dataset order, PL rows = the OOF experts' studies
    toff = ClassificationPredictions.load(args.train_off)
    p_tr = np.asarray(toff.probabilities, dtype=np.float64)
    y_tr = np.asarray(toff.labels, dtype=np.int64)
    y_tr[y_tr < 0] = -1
    if tuple(toff.pathology_names) != names:
        raise SystemExit("train OFF npz has another finding order")
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

    params = [rs.r1_trainfit(p_tr[o_tr], [e], y_tr[o_tr], p_va[pl_va], [v])[1][0] for e, v in zip(ex_tr, ex_va,
                                                                                                    strict=True)]
    off_cut, _meta = load_cutpoint_file(args.off_cutpoints)
    r1_cut, r1_va = fit_r1_cutpoints(p_va[pl_va], y_va[pl_va], ex_va, params, names)

    keys_va, _vm = split_keys("val", args.config, ViewMap.load(args.view_map).frontal_keys)
    write_npz(outs[0], keys_va, pl_va, names, ex_va, params, off_cut, r1_cut)

    keys_tr, vm_tr = split_keys("train", args.config, np.asarray(toff.sample_keys))
    if not set(tr_keys.tolist()) <= set(vm_tr.frontal_keys[vm_tr.groups == "PL"].tolist()):
        raise SystemExit("train OOF expert studies are not all PL studies of the rebuilt train view map")
    write_npz(outs[1], keys_tr, o_tr, names, ex_tr, params, off_cut, r1_cut)

    off_codes = apply_cutpoints(p_va[pl_va], off_cut, names)
    new_codes = apply_cutpoints(r1_va, r1_cut, names)
    summary = {"val_n": int(len(keys_va)), "val_pl": int(len(pl_va)), "train_n": int(len(keys_tr)),
               "train_pl": int(len(o_tr)), "val_changed_cells_on_off_probs": int((off_codes != new_codes).sum()),
               "val_changed_studies_on_off_probs": int((off_codes != new_codes).any(1).sum()),
               "r1_cutpoints": {k: [float(a), float(b)] for k, (a, b) in r1_cut.items()},
               "A_mean": {n: float(np.mean([A[j, 0] for A, _B in params])) for j, n in enumerate(names)}}
    Path(f"{args.out_prefix}.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"[r1-cues] val {summary['val_n']} (PL {summary['val_pl']}), train {summary['train_n']} "
          f"(PL {summary['train_pl']}); val cells changed on OFF's probs {summary['val_changed_cells_on_off_probs']} "
          f"in {summary['val_changed_studies_on_off_probs']} studies", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
