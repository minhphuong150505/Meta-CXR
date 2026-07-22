"""Map a plain Stage-2 record dict to a ``PromptContext``.

Bridges the record shapes produced by ``build_stage1_records`` (Q-Former) and
``build_records`` (native manifest) — plus any extra view/prior metadata threaded
in — onto the typed builder input, without importing torch.
"""

from __future__ import annotations

from typing import Any, Mapping

from .schemas import PromptContext, VisualMode


def _as_tuple(value: Any) -> tuple[str, ...]:
    if not value:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(str(v) for v in value)
    return (str(value),)


def _as_prob_map(value: Any) -> dict[str, float]:
    if isinstance(value, Mapping):
        return {str(k): float(v) for k, v in value.items()}
    return {}


def context_from_record(
    record: Mapping[str, Any],
    *,
    visual_mode: VisualMode,
    qformer_token_count: int | None = None,
    prompt_version: str | None = None,
) -> PromptContext:
    """Build a ``PromptContext``. Absent fields become None/empty, never invented.

    Accepts either a ``pred_groups`` dict ({"positive": [...], ...}) or explicit
    ``positive_findings`` / ``uncertain_findings`` / ``negative_findings`` keys.
    """
    groups = record.get("pred_groups") or {}
    positive = _as_tuple(record.get("positive_findings") or groups.get("positive"))
    uncertain = _as_tuple(record.get("uncertain_findings") or groups.get("uncertain"))
    negative = _as_tuple(record.get("negative_findings") or groups.get("negative"))

    token_count = qformer_token_count
    if token_count is None and visual_mode.uses_soft_tokens:
        token_count = record.get("qformer_token_count")

    return PromptContext(
        study_id=str(record.get("study_id") or record.get("sample_key") or "unknown"),
        visual_mode=visual_mode,
        positive_findings=positive,
        uncertain_findings=uncertain,
        negative_findings=negative,
        positive_probabilities=_as_prob_map(record.get("positive_probabilities")),
        uncertain_probabilities=_as_prob_map(record.get("uncertain_probabilities")),
        negative_probabilities=_as_prob_map(record.get("negative_probabilities")),
        anchor_view=record.get("anchor_view"),
        auxiliary_views=_as_tuple(record.get("auxiliary_views")),
        prior_available=bool(record.get("prior_available", False)),
        comparison_available=bool(record.get("comparison_available", False)),
        indication=record.get("indication"),
        technique=record.get("technique"),
        has_support_devices=bool(record.get("has_support_devices", False)),
        qformer_token_count=token_count,
        prompt_version=prompt_version,
    )
