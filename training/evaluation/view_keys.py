"""Frontal / lateral key map per study, for joining a lateral-image pass to frontal predictions.

Plan: ``docs/handoff/PLAN-2026-11-04-residual-lateral.md``.

The Stage-1 eval hook writes ``sample_keys = build_sample_keys([str(image_id)])``,
where ``image_id`` is the ANNOTATION ROW of the study's ANCHOR image
(``ReportDataset.img_ids``). A pass whose anchor is the lateral image therefore keys
each study by its lateral row, and cannot be paired with a frontal-anchor npz by key.
This module rebuilds, from the dataset's own annotation rows, for every study in
dataset order:

* ``frontal_key`` -- the key of the default anchor (``anchor_priority=[PA, AP, lateral]``),
  i.e. what every frontal-anchor npz (OFF, ON, the encoder probe) carries;
* ``lateral_key`` -- the key of the lateral image that ``max_aux_views=1`` pairs with
  it ('' when the study is not PL);
* ``group``       -- ``PL`` (anchor PA/AP and aux view id 2), ``F1`` (anchor PA/AP, no
  aux) or ``other`` (everything else: PA+AP pairs, lateral-only, unknown views);
* ``num_views``   -- ``1 + len(aux)`` at ``max_aux_views=1``.

Same groups as the 2026-10-27 evidence package (val: PL 797, F1 930, other 81).
numpy + stdlib only. The map holds salted digests and group labels, never identifiers;
it is still a derivative of MIMIC-CXR and stays on the training host.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from model.lavis.data.mimic_cxr_utils import build_study_index
except ImportError:  # pragma: no cover - script use outside the repo root
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from model.lavis.data.mimic_cxr_utils import build_study_index

from training.evaluation.schemas import build_sample_keys

PL, F1, OTHER = "PL", "F1", "other"
GROUPS = (PL, F1, OTHER)
FRONTAL_VIEW_IDS = (0, 1)  # PA, AP
LATERAL_VIEW_ID = 2  # LATERAL, LL
DEFAULT_PRIORITY = ("PA", "AP", "lateral")
LATERAL_PRIORITY = ("lateral", "PA", "AP")


def study_group(study: Mapping[str, Any]) -> str:
    """``PL`` / ``F1`` / ``other`` of one ``build_study_index`` entry (default priority)."""
    if study["anchor_view_id"] in FRONTAL_VIEW_IDS:
        aux = list(study["aux_view_ids"])
        if not aux:
            return F1
        if aux == [LATERAL_VIEW_ID]:
            return PL
    return OTHER


@dataclass
class ViewMap:
    """Per study, in dataset order: frontal / lateral key, group, number of views."""

    frontal_keys: np.ndarray
    lateral_keys: np.ndarray
    groups: np.ndarray
    num_views: np.ndarray
    frontal_rows: np.ndarray  # annotation row of the default anchor (in memory only)
    lateral_rows: np.ndarray  # annotation row of the PL lateral, -1 otherwise

    def __len__(self) -> int:
        return len(self.frontal_keys)

    def counts(self) -> dict[str, int]:
        return {g: int(np.sum(self.groups == g)) for g in GROUPS}

    def save(self, path: str | Path) -> Path:
        """Keys, groups and view counts only -- row positions are not written."""
        path = Path(path)
        np.savez(path, frontal_keys=self.frontal_keys, lateral_keys=self.lateral_keys,
                 groups=self.groups, num_views=self.num_views)
        return path

    @classmethod
    def load(cls, path: str | Path) -> ViewMap:
        with np.load(Path(path), allow_pickle=False) as h:
            n = len(h["frontal_keys"])
            return cls(frontal_keys=h["frontal_keys"], lateral_keys=h["lateral_keys"],
                       groups=h["groups"], num_views=h["num_views"],
                       frontal_rows=np.full(n, -1), lateral_rows=np.full(n, -1))


def build_view_map(rows: Iterable[Mapping[str, Any]], *, salt: str = "meta-cxr-eval") -> ViewMap:
    """The map for annotation ``rows`` (``subject_id``, ``study_id``, ``ViewPosition``).

    ``rows`` must be the dataset's own ``annotation`` in its own order: keys digest the
    row POSITION, exactly as the eval hook digests ``image_id``.
    """
    rows = list(rows)
    studies = build_study_index(rows, anchor_priority=DEFAULT_PRIORITY, max_aux_views=1)
    groups = np.array([study_group(s) for s in studies], dtype="U5")
    frontal_rows = np.array([s["anchor"] for s in studies], dtype=np.int64)
    lateral_rows = np.array([s["aux"][0] if g == PL else -1 for s, g in zip(studies, groups, strict=True)],
                            dtype=np.int64)
    frontal_keys = build_sample_keys([str(r) for r in frontal_rows], salt=salt)
    lateral_keys = np.where(
        lateral_rows >= 0,
        build_sample_keys([str(r) for r in lateral_rows], salt=salt),
        "",
    ).astype(frontal_keys.dtype)
    num_views = np.array([1 + len(s["aux"]) for s in studies], dtype=np.int64)
    return ViewMap(frontal_keys, lateral_keys, groups, num_views, frontal_rows, lateral_rows)


def lateral_pass_positions(
    view_map: ViewMap,
    lateral_studies: Sequence[Mapping[str, Any]],
    order: Iterable[int] | None = None,
) -> np.ndarray:
    """Positions (dataset indices) of PL studies in a lateral-anchor dataset, in ``order``.

    ``lateral_studies`` is ``dataset.studies`` of a dataset built with
    ``anchor_priority=[lateral, PA, AP]``. Raises unless both indexes list the same
    studies in the same order and every PL study's lateral anchor is exactly the
    lateral that ``max_aux_views=1`` pairs with its frontal anchor.
    """
    if len(lateral_studies) != len(view_map):
        raise ValueError(
            f"lateral-anchor index has {len(lateral_studies)} studies, map has {len(view_map)}")
    positions = np.arange(len(view_map)) if order is None else np.asarray(list(order), dtype=np.int64)
    keep = [int(i) for i in positions if view_map.groups[i] == PL]
    for i in keep:
        study = lateral_studies[i]
        if study["anchor_view_id"] != LATERAL_VIEW_ID or int(study["anchor"]) != int(view_map.lateral_rows[i]):
            raise ValueError(f"study at dataset index {i}: lateral anchor does not match the PL lateral")
    return np.asarray(keep, dtype=np.int64)


def check_num_views(view_map: ViewMap, num_views: np.ndarray, keys: np.ndarray) -> None:
    """Raise unless a multi-view npz's ``num_views`` matches the rebuilt map row by row."""
    if not np.array_equal(np.asarray(keys), view_map.frontal_keys):
        raise ValueError("npz keys are not the map's frontal keys in dataset order")
    if not np.array_equal(np.asarray(num_views, dtype=np.int64), view_map.num_views):
        raise ValueError("npz num_views disagrees with the rebuilt study index")


def align(keys: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Index ``idx`` with ``keys[idx] == target``; raises if a target key is missing or repeated."""
    keys, target = np.asarray(keys), np.asarray(target)
    if len(np.unique(keys)) != len(keys):
        raise ValueError("duplicate keys")
    lookup = {k: i for i, k in enumerate(keys.tolist())}
    missing = [k for k in target.tolist() if k not in lookup]
    if missing:
        raise ValueError(f"{len(missing)} of {len(target)} keys not found")
    return np.array([lookup[k] for k in target.tolist()], dtype=np.int64)
