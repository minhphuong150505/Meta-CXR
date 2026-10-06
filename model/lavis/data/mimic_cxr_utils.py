"""Dependency-light helpers for MIMIC-CXR study sampling."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


VIEW_ID_MAP = {"PA": 0, "AP": 1, "LATERAL": 2, "LL": 2}
UNKNOWN_VIEW_ID = 3

# Upper bound on auxiliary views per study. One was the only value ever trained
# (and stays the shipped default); 2-3 exist for the multi-view ablation
# (docs/handoff/PLAN-2026-10-06-multi-aux-views.md). Measured on
# full_allviews_v2: a second complementary view exists for 4.2% of train /
# 4.9% val / 5.3% test studies, a third for 0.1% or fewer, so 3 already covers
# every study that has one.
MAX_SUPPORTED_AUX_VIEWS = 3


def view_id(view_position: Any) -> int:
    """Map a MIMIC view label to the compact IDs used by view fusion."""
    if view_position is None:
        return UNKNOWN_VIEW_ID
    value = str(view_position).strip().upper()
    if not value or value == "NAN":
        return UNKNOWN_VIEW_ID
    return VIEW_ID_MAP.get(value, UNKNOWN_VIEW_ID)


def build_study_index(
    rows: Iterable[Mapping[str, Any]],
    anchor_priority: Iterable[str] = ("PA", "AP", "LATERAL"),
    max_aux_views: int = 1,
) -> list[dict[str, Any]]:
    """Choose one anchor and up to ``max_aux_views`` complementary views per study.

    Positions in the returned dictionaries index the input row order.  Study
    identity includes both ``subject_id`` and ``study_id`` so malformed inputs
    cannot mix patients that happen to share an identifier.
    """
    if not 0 <= max_aux_views <= MAX_SUPPORTED_AUX_VIEWS:
        raise ValueError(
            f"max_aux_views must be in 0..{MAX_SUPPORTED_AUX_VIEWS}; got {max_aux_views}"
        )

    records = list(rows)
    ranks: dict[int, int] = {}
    for rank, name in enumerate(anchor_priority):
        ranks.setdefault(view_id(name), rank)
    default_rank = len(ranks)

    groups: dict[tuple[Any, Any], list[int]] = {}
    row_view_ids: list[int] = []
    for position, row in enumerate(records):
        try:
            study_key = (row["subject_id"], row["study_id"])
        except KeyError as exc:
            raise ValueError(f"Missing study identity column: {exc.args[0]}") from exc
        groups.setdefault(study_key, []).append(position)
        row_view_ids.append(view_id(row.get("ViewPosition")))

    studies: list[dict[str, Any]] = []
    for study_key, positions in groups.items():
        ordered = sorted(
            positions,
            key=lambda position: ranks.get(row_view_ids[position], default_rank),
        )
        anchor = ordered[0]
        anchor_view = row_view_ids[anchor]

        # Repeated acquisitions of the anchor projection do not add the view
        # diversity intended by this path. Prefer PA -> AP -> lateral and leave
        # aux empty when a genuinely complementary view is unavailable.
        complementary = [
            position for position in ordered[1:]
            if row_view_ids[position] != anchor_view
        ]
        aux = complementary[:max_aux_views]
        studies.append(
            {
                "study_key": study_key,
                "anchor": anchor,
                "aux": aux,
                "anchor_view_id": anchor_view,
                "aux_view_ids": [row_view_ids[position] for position in aux],
            }
        )

    return studies
