"""``PromptBuilder`` — the single Stage-2 prompt entry point for train and inference.

Given a ``PromptContext`` and a ``PromptConfig`` it emits an ordered list of
model-agnostic ``PromptPart``s plus reproducibility metadata. It touches no model,
no tokenizer and no torch, so train and inference can call the *same* function and
a test can assert byte-for-byte parity.
"""

from __future__ import annotations

import hashlib
from typing import Callable

from . import templates
from .policies import (
    NegativePolicy,
    NormalPolicy,
    UncertaintyPolicy,
    render_uncertain,
    select_negative_findings,
)
from .schemas import PartKind, PromptContext, PromptPart, RenderedPrompt, VisualMode
from .templates import join_or_none
from .validation import PromptConfig, PromptConfigError

_ABSENT_PRIORITY = 2
_CONTEXT_PRIORITY = 1


class PromptBuilder:
    def __init__(self, config: PromptConfig):
        self.config = config

    # -- public API ---------------------------------------------------------
    def build(self, context: PromptContext) -> RenderedPrompt:
        self._validate(context)
        parts: list[PromptPart] = []
        parts.extend(self._visual_parts(context))
        if self.config.visual_mode.includes_structured:
            parts.extend(self._structured_parts(context))
        parts.extend(self._context_parts(context))
        parts.append(self._instruction_part(context))

        template_hash = templates.template_hash(self.config.visual_mode)
        config_hash = self.config.config_hash()
        prompt_hash = hashlib.sha256(
            f"{self.config.version}\x1f{template_hash}\x1f{config_hash}".encode("utf-8")
        ).hexdigest()[:16]
        return RenderedPrompt(
            parts=tuple(parts),
            prompt_version=self.config.version,
            template_hash=template_hash,
            config_hash=config_hash,
            prompt_hash=prompt_hash,
        )

    def build_user_messages(self, context: PromptContext) -> list[PromptPart]:
        """Ordered user-turn parts. Called identically by train and inference."""
        return list(self.build(context).parts)

    def render_user_text(
        self, context: PromptContext, soft_token: str = "<qformer_soft_token>"
    ) -> str:
        return self.build(context).user_text(soft_token)

    # -- validation ---------------------------------------------------------
    def _validate(self, context: PromptContext) -> None:
        if context.visual_mode != self.config.visual_mode:
            raise PromptConfigError(
                f"context.visual_mode ({context.visual_mode.value}) does not match "
                f"config.visual_mode ({self.config.visual_mode.value})"
            )
        overlap = (
            set(context.positive_findings) & set(context.uncertain_findings)
            | set(context.positive_findings) & set(context.negative_findings)
            | set(context.uncertain_findings) & set(context.negative_findings)
        )
        if overlap:
            raise PromptConfigError(
                "a finding may not appear in more than one class: "
                + ", ".join(sorted(overlap))
            )
        if self.config.visual_mode.uses_soft_tokens:
            if not context.qformer_token_count or context.qformer_token_count <= 0:
                raise PromptConfigError(
                    f"{self.config.visual_mode.value} requires a positive "
                    "qformer_token_count matching the Q-Former embedding count"
                )

    # -- visual channel -----------------------------------------------------
    def _visual_parts(self, context: PromptContext) -> list[PromptPart]:
        parts: list[PromptPart] = []
        mode = self.config.visual_mode
        if mode.uses_native_image:
            parts.append(PromptPart(kind=PartKind.IMAGE, image_role="anchor"))
            if mode.includes_aux_image and context.auxiliary_views:
                parts.append(PromptPart(kind=PartKind.IMAGE, image_role="auxiliary"))
        if mode.uses_soft_tokens:
            parts.append(PromptPart(kind=PartKind.TEXT, text=templates.VISUAL_HEADER))
            parts.append(
                PromptPart(kind=PartKind.SOFT_TOKENS, count=context.qformer_token_count)
            )
        return parts

    # -- structured cues ----------------------------------------------------
    def _structured_parts(self, context: PromptContext) -> list[PromptPart]:
        cfg = self.config
        if context.is_structurally_normal:
            return self._normal_parts(context)

        possible = render_uncertain(
            context,
            cfg.uncertainty_policy,
            low_confidence_threshold=cfg.low_confidence_threshold,
        )
        absent = select_negative_findings(
            context,
            cfg.negative_policy,
            max_negative_findings=cfg.max_negative_findings,
            negative_probability_threshold=cfg.negative_probability_threshold,
            critical_findings=cfg.critical_findings,
        )
        head = (
            f"{templates.STRUCTURED_HEADER}\n"
            f"- {templates.PRESENT_LABEL}: {join_or_none(tuple(context.positive_findings))}\n"
            f"- {templates.POSSIBLE_LABEL}: {join_or_none(possible)}"
        )
        parts = [PromptPart(kind=PartKind.TEXT, text=head)]
        if absent:
            parts.append(
                PromptPart(
                    kind=PartKind.TEXT,
                    text=f"- {templates.ABSENT_LABEL}: {join_or_none(absent)}",
                    budget_priority=_ABSENT_PRIORITY,
                )
            )
        return parts

    def _normal_parts(self, context: PromptContext) -> list[PromptPart]:
        policy = self.config.normal_policy
        if policy is NormalPolicy.NO_STRUCTURED_NORMAL_STATEMENT:
            return []
        if policy is NormalPolicy.COMPACT_SUMMARY:
            return [
                PromptPart(
                    kind=PartKind.TEXT,
                    text=f"{templates.STRUCTURED_HEADER}\n{templates.COMPACT_NORMAL_STATEMENT}",
                )
            ]
        # critical_negatives / all_negatives share the P/N/U layout with an empty
        # present + possible; the absent line differs by which negatives are shown.
        negative_policy = (
            NegativePolicy.ALL
            if policy is NormalPolicy.ALL_NEGATIVES
            else NegativePolicy.CRITICAL_ONLY
        )
        absent = select_negative_findings(
            context,
            negative_policy,
            max_negative_findings=self.config.max_negative_findings,
            negative_probability_threshold=self.config.negative_probability_threshold,
            critical_findings=self.config.critical_findings,
        )
        head = (
            f"{templates.STRUCTURED_HEADER}\n"
            f"- {templates.PRESENT_LABEL}: {templates.NONE_TOKEN}\n"
            f"- {templates.POSSIBLE_LABEL}: {templates.NONE_TOKEN}"
        )
        parts = [PromptPart(kind=PartKind.TEXT, text=head)]
        if absent:
            parts.append(
                PromptPart(
                    kind=PartKind.TEXT,
                    text=f"- {templates.ABSENT_LABEL}: {join_or_none(absent)}",
                    budget_priority=_ABSENT_PRIORITY,
                )
            )
        return parts

    # -- study context ------------------------------------------------------
    def _context_parts(self, context: PromptContext) -> list[PromptPart]:
        cfg = self.config
        lines: list[str] = []
        if cfg.include_views and (context.anchor_view or context.auxiliary_views):
            lines.append(f"- Views: {self._view_description(context)}")
        if cfg.include_prior_flag:
            available = "yes" if (context.comparison_available or context.prior_available) else "no"
            lines.append(f"- Prior comparison available: {available}")
        if cfg.include_indication and context.indication:
            lines.append(f"- Indication: {context.indication.strip()}")
        if cfg.include_technique and context.technique:
            lines.append(f"- Technique: {context.technique.strip()}")
        if not lines:
            return []
        body = templates.CONTEXT_HEADER + "\n" + "\n".join(lines)
        return [PromptPart(kind=PartKind.TEXT, text=body, budget_priority=_CONTEXT_PRIORITY)]

    @staticmethod
    def _view_description(context: PromptContext) -> str:
        anchor = context.anchor_view
        aux = tuple(context.auxiliary_views)
        if anchor and aux:
            return f"{anchor} (anchor), {', '.join(aux)} (auxiliary)"
        if anchor:
            return anchor
        return ", ".join(aux)

    # -- instruction --------------------------------------------------------
    def _instruction_part(self, context: PromptContext) -> PromptPart:
        cfg = self.config
        mode = cfg.visual_mode
        lines = [templates.TASK_LINE]
        if mode.includes_structured:
            lines.append(
                templates.EVIDENCE_PRIMARY_VISUAL
                if cfg.visual_primary
                else templates.EVIDENCE_NEUTRAL
            )
            lines.append(templates.EVIDENCE_UNCERTAIN)
        else:
            lines.append(templates.NATIVE_ONLY_BODY)
        prior_known = context.comparison_available or context.prior_available
        if cfg.forbid_comparison_without_prior and not prior_known:
            lines.append(templates.NO_PRIOR_GUARD)
        lines.append(templates.FORBID_META if mode.includes_structured else templates.FORBID_META_NATIVE)
        lines.append(templates.sentence_constraint(cfg.min_sentences, cfg.max_sentences))
        return PromptPart(kind=PartKind.TEXT, text=" ".join(lines))


def fit_to_budget(
    rendered: RenderedPrompt,
    count_fn: Callable[[str], int],
    max_tokens: int,
) -> RenderedPrompt:
    """Drop droppable parts (negatives first, then metadata) to meet a budget.

    ``count_fn`` measures a rendered text chunk. Visual tokens and the instruction
    (priority 0) are never dropped; the assistant target is not part of this
    object and is never touched here.
    """
    parts = list(rendered.parts)

    def total() -> int:
        return sum(
            count_fn(part.text) if part.kind is PartKind.TEXT and part.text else (part.count or 0)
            for part in parts
        )

    for priority in sorted({p.budget_priority for p in parts if p.budget_priority > 0}, reverse=True):
        if total() <= max_tokens:
            break
        parts = [p for p in parts if p.budget_priority != priority]
    return RenderedPrompt(
        parts=tuple(parts),
        prompt_version=rendered.prompt_version,
        template_hash=rendered.template_hash,
        config_hash=rendered.config_hash,
        prompt_hash=rendered.prompt_hash,
    )
