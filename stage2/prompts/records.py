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
    # A guided mode whose record carries no prediction keys AT ALL is the
    # wiring bug this guard exists for: `_as_tuple(None)` returns (), the
    # builder emits a prompt with no cues, and the run trains for days on
    # prompts indistinguishable from the unguided arm. Distinguish "no keys"
    # from "keys present but empty" -- the latter is a legitimate prediction
    # (a study Stage 1 called normal) and must still be allowed through.
    prediction_keys = (
        "pred_groups",
        "positive_findings",
        "uncertain_findings",
        "negative_findings",
    )
    if visual_mode.includes_structured and not any(key in record for key in prediction_keys):
        raise ValueError(
            f"visual_mode={visual_mode.value} places Stage-1 findings in the "
            "prompt, but this record carries none of "
            f"{prediction_keys}. Native manifest records have no predictions "
            "attached; a guided mode needs records built with a Stage-1 pass."
        )
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
