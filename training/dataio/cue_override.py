"""Stage-2 cues of PL studies from OFF + the lateral R1 residual, applied to each record's OWN Stage-1 logits.

Plan: ``docs/handoff/PLAN-2026-11-04-residual-lateral.md``, sections "R1 vào giai đoạn 2" (steps A and B).
numpy + stdlib only (no torch, no LAVIS), so both Stage-2 CLIs can import it at module scope.

The override npz (``scripts/r1_stage2_cues.py``) carries the R1 RECIPE, not finished cues, because Stage-2 train
records are encoded from augmented images and their logits are not the clean OFF predictions:

* ``keys``       -- ``blake2b(anchor dicom_id, 12)`` per study of the split (``sample_key_for`` of the image path);
* ``pl_row``     -- row into ``l_probs`` for PL studies, -1 otherwise;
* ``names``      -- finding order of ``l_probs`` / ``A`` / ``B`` / cutpoints;
* ``l_probs``    -- ``[k, n_pl, F, 3]`` lateral-expert probabilities, one slice per expert seed;
* ``A``, ``B``   -- ``[k, F]`` and ``[k, F, 3]`` R1 parameters per seed (``residual_stack.r1_trainfit``);
* ``off_cut``, ``r1_cut`` -- ``[F, 2]`` cutpoints (t1, t2): OFF's, and the PL cutpoints under OFF + R1.

For a PL record: ``p`` = softmax of its own ``class_logits`` (MHCAC order ``findings``); per seed
``softmax(log p + A log l + B)``; mean over seeds; cutpoints ``r1_cut`` on ``p_pos / (p_pos + p_neg)``. Before that, the
record's current cues are recomputed from ``p`` with ``off_cut`` and must agree with what it carries (>= MIN_AGREEMENT
of cells) -- proof that the npz, the finding order and the record describe the same model. Non-PL records are never
touched. Must run BEFORE ``--train-cues oracle_mix``, so the oracle mix keeps R1's class where it keeps the model's.
"""

from __future__ import annotations

from collections.abc import Sequence
from hashlib import blake2b
from pathlib import Path

import numpy as np

EPS = 1e-7
MIN_AGREEMENT = 0.99
GROUPS = ("negative", "positive", "uncertain")  # code 0, 1, 2


def record_key(record: dict) -> str:
    """``generate_stage2_reports.sample_key_for`` of the record's anchor image."""
    return blake2b(Path(str(record["image_path"])).stem.encode(), digest_size=12).hexdigest()


def load_override(path: str | Path) -> dict:
    with np.load(Path(path), allow_pickle=False) as h:
        ov = {k: h[k] for k in h.files}
    ov["names"] = [str(n) for n in ov["names"]]
    ov["index"] = {str(k): i for i, k in enumerate(ov["keys"])}
    return ov


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def _codes(p: np.ndarray, cut: np.ndarray) -> np.ndarray:
    s = p[..., 1] / np.maximum(p[..., 1] + p[..., 0], 1e-12)
    return np.where(s >= cut[:, 1], 1, np.where(s >= cut[:, 0], 2, 0))


def r1_probs(p: np.ndarray, l_probs: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Mean over seeds of ``softmax(log p + A log l + B)``; ``p`` ``[F, 3]``, ``l_probs`` ``[k, F, 3]``."""
    logp = np.log(np.clip(p, EPS, 1.0))
    out = [_softmax(logp + A[s][:, None] * np.log(np.clip(l_probs[s], EPS, 1.0)) + B[s]) for s in range(len(A))]
    return np.mean(out, 0)


def _groups(codes: np.ndarray, names: Sequence[str], findings: Sequence[str]) -> dict[str, list[str]]:
    by_name = dict(zip(names, (int(c) for c in codes), strict=True))
    out = {"positive": [], "negative": [], "uncertain": []}
    for name in findings:
        if name != "No Finding":
            out[GROUPS[by_name[name]]].append(name)
    return out


def apply_r1_cues(records: list[dict], override: dict, findings: Sequence[str],
                  min_agreement: float = MIN_AGREEMENT) -> dict[str, float | int]:
    """Replace the cues of PL records in place; return counts only. Raises (SystemExit) on any mismatch."""
    names = override["names"]
    if sorted(names) != sorted(findings):
        raise SystemExit("cue override: finding names differ from the Stage-1 head's")
    perm = np.array([list(findings).index(n) for n in names])  # npz order -> record logits order
    agree = total = n_pl = changed_records = changed_cells = 0
    missing = 0
    for record in records:
        i = override["index"].get(record_key(record))
        if i is None:
            missing += 1
            continue
        row = int(override["pl_row"][i])
        if row < 0:
            continue
        logits = np.asarray(record["class_logits"], dtype=np.float64)
        p = _softmax(logits)[perm]  # [F, 3] in npz order
        own = record.get("pred_groups", {})
        expected = _groups(_codes(p, override["off_cut"]), names, findings)
        for g in GROUPS:
            agree += len(set(own.get(g, [])) & set(expected[g]))
            total += len(expected[g])
        new = _groups(_codes(r1_probs(p, override["l_probs"][:, row], override["A"], override["B"]),
                             override["r1_cut"]), names, findings)
        diff = sum(len(set(new[g]) - set(own.get(g, []))) for g in GROUPS)
        n_pl += 1
        changed_cells += diff
        changed_records += int(diff > 0)
        record["pred_groups"] = new
    if missing:
        raise SystemExit(f"cue override: {missing} records are not in the override")
    agreement = agree / total if total else 1.0
    if agreement < min_agreement:
        raise SystemExit(f"cue override: PL records agree with OFF's cutpoint cues on {agreement:.4f} of cells "
                         f"(< {min_agreement}); records and override describe different things")
    return {"pl_records": n_pl, "changed_records": changed_records, "changed_cells": changed_cells,
            "off_agreement": round(agreement, 6)}
