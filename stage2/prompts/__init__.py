"""Stage-2 prompt builder: one shared, versioned, torch-free prompt entry point.

Train and inference both call ``PromptBuilder.build_user_messages`` so the user
turn is provably identical. See ``docs/stage2_prompt_design.md``.
"""

from __future__ import annotations

from .builder import PromptBuilder, fit_to_budget
from .ontology import (
    DEFAULT_CRITICAL_FINDINGS,
    MODELED_FINDINGS,
    NO_FINDING,
    validate_critical_findings,
)
from .records import context_from_record
from .policies import (
    NegativePolicy,
    NormalPolicy,
    TemporalTargetPolicy,
    UncertaintyPolicy,
    apply_temporal_target_policy,
    contains_temporal_language,
)
from .schemas import (
    PartKind,
    PromptContext,
    PromptPart,
    RenderedPrompt,
    VisualMode,
)
from .validation import (
    PromptConfig,
    PromptConfigError,
    config_from_mapping,
    load_prompt_config,
)

__all__ = [
    "PromptBuilder",
    "fit_to_budget",
    "PromptConfig",
    "PromptConfigError",
    "load_prompt_config",
    "config_from_mapping",
    "context_from_record",
    "PromptContext",
    "PromptPart",
    "PartKind",
    "RenderedPrompt",
    "VisualMode",
    "NormalPolicy",
    "NegativePolicy",
    "UncertaintyPolicy",
    "TemporalTargetPolicy",
    "apply_temporal_target_policy",
    "contains_temporal_language",
    "MODELED_FINDINGS",
    "NO_FINDING",
    "DEFAULT_CRITICAL_FINDINGS",
    "validate_critical_findings",
]
