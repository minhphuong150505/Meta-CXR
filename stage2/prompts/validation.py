"""``PromptConfig`` — the validated, hashable configuration for a prompt run.

Loaded from the ``prompt:`` block of a YAML file (``configs/stage2_prompt_v2.yaml``)
or built directly. Invalid policy values raise ``PromptConfigError`` with a clear
message rather than failing deep inside the builder.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .ontology import DEFAULT_CRITICAL_FINDINGS
from .policies import (
    NegativePolicy,
    NormalPolicy,
    TemporalTargetPolicy,
    UncertaintyPolicy,
)
from .schemas import VisualMode


class PromptConfigError(ValueError):
    """Raised for any invalid prompt configuration."""


def _coerce(enum_cls, value, field_name):
    try:
        return enum_cls(value)
    except ValueError:
        allowed = ", ".join(member.value for member in enum_cls)
        raise PromptConfigError(
            f"invalid {field_name} {value!r}; expected one of: {allowed}"
        ) from None


@dataclass(frozen=True)
class PromptConfig:
    version: str = "stage2_prompt_v2"
    visual_mode: VisualMode = VisualMode.QFORMER_GUIDED
    normal_policy: NormalPolicy = NormalPolicy.COMPACT_SUMMARY
    negative_policy: NegativePolicy = NegativePolicy.CRITICAL_ONLY
    max_negative_findings: int = 4
    negative_probability_threshold: float = 0.85
    uncertainty_policy: UncertaintyPolicy = UncertaintyPolicy.EXPLICIT_POSSIBLE
    low_confidence_threshold: float = 0.60
    critical_findings: tuple[str, ...] = DEFAULT_CRITICAL_FINDINGS
    visual_primary: bool = True
    structured_auxiliary: bool = True
    include_views: bool = True
    include_indication: bool = True
    include_technique: bool = True
    include_prior_flag: bool = True
    forbid_comparison_without_prior: bool = True
    temporal_target_policy: TemporalTargetPolicy = TemporalTargetPolicy.KEEP
    section: str = "findings"
    min_sentences: int = 1
    max_sentences: int = 4
    allow_bullets: bool = False
    allow_impression: bool = False
    allow_recommendations: bool = False
    preserve_visual_tokens: bool = True
    truncate_negatives_first: bool = True

    def __post_init__(self) -> None:
        if self.section != "findings":
            raise PromptConfigError(
                f"only section=findings is supported, got {self.section!r}"
            )
        if not 1 <= self.min_sentences <= self.max_sentences:
            raise PromptConfigError(
                "require 1 <= min_sentences <= max_sentences, got "
                f"{self.min_sentences}..{self.max_sentences}"
            )
        if self.max_negative_findings < 0:
            raise PromptConfigError("max_negative_findings must be >= 0")
        for name, value in (
            ("negative_probability_threshold", self.negative_probability_threshold),
            ("low_confidence_threshold", self.low_confidence_threshold),
        ):
            if not 0.0 <= value <= 1.0:
                raise PromptConfigError(f"{name} must be in [0, 1], got {value}")

    def canonical(self) -> dict[str, Any]:
        """Deterministic, JSON-safe view used for the config hash."""
        raw = asdict(self)
        return {
            key: (value.value if hasattr(value, "value") else value)
            for key, value in raw.items()
        }

    def config_hash(self, length: int = 16) -> str:
        encoded = json.dumps(self.canonical(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:length]


_SCALAR_FIELDS = {
    "version": str,
    "max_negative_findings": int,
    "negative_probability_threshold": float,
    "low_confidence_threshold": float,
    "min_sentences": int,
    "max_sentences": int,
    "section": str,
}
_BOOL_FIELDS = (
    "visual_primary",
    "structured_auxiliary",
    "include_views",
    "include_indication",
    "include_technique",
    "include_prior_flag",
    "forbid_comparison_without_prior",
    "allow_bullets",
    "allow_impression",
    "allow_recommendations",
    "preserve_visual_tokens",
    "truncate_negatives_first",
)


def config_from_mapping(block: Mapping[str, Any]) -> PromptConfig:
    """Build a ``PromptConfig`` from a nested ``prompt:`` mapping.

    Unknown keys raise, so a typo in a config file is caught rather than silently
    ignored (which would leave the default policy in force).
    """
    kwargs: dict[str, Any] = {}
    block = dict(block)

    for name, caster in _SCALAR_FIELDS.items():
        if name in block:
            kwargs[name] = caster(block.pop(name))
    for name in _BOOL_FIELDS:
        if name in block:
            kwargs[name] = bool(block.pop(name))

    if "visual_mode" in block:
        kwargs["visual_mode"] = _coerce(VisualMode, block.pop("visual_mode"), "visual_mode")
    if "normal_policy" in block:
        kwargs["normal_policy"] = _coerce(NormalPolicy, block.pop("normal_policy"), "normal_policy")
    if "negative_policy" in block:
        kwargs["negative_policy"] = _coerce(
            NegativePolicy, block.pop("negative_policy"), "negative_policy"
        )
    if "uncertainty_policy" in block:
        kwargs["uncertainty_policy"] = _coerce(
            UncertaintyPolicy, block.pop("uncertainty_policy"), "uncertainty_policy"
        )
    if "critical_findings" in block:
        kwargs["critical_findings"] = tuple(block.pop("critical_findings"))

    # Nested convenience blocks (as written in the YAML example).
    evidence = block.pop("evidence_priority", None)
    if isinstance(evidence, Mapping):
        if "visual_primary" in evidence:
            kwargs["visual_primary"] = bool(evidence["visual_primary"])
        if "structured_auxiliary" in evidence:
            kwargs["structured_auxiliary"] = bool(evidence["structured_auxiliary"])
    context = block.pop("context", None)
    if isinstance(context, Mapping):
        for name in ("include_views", "include_indication", "include_technique", "include_prior_flag"):
            if name in context:
                kwargs[name] = bool(context[name])
    temporal = block.pop("temporal", None)
    if isinstance(temporal, Mapping):
        if "forbid_comparison_without_prior" in temporal:
            kwargs["forbid_comparison_without_prior"] = bool(
                temporal["forbid_comparison_without_prior"]
            )
        if "target_policy" in temporal:
            kwargs["temporal_target_policy"] = _coerce(
                TemporalTargetPolicy, temporal["target_policy"], "temporal.target_policy"
            )
    output = block.pop("output", None)
    if isinstance(output, Mapping):
        for name in ("section", "min_sentences", "max_sentences"):
            if name in output:
                kwargs[name] = _SCALAR_FIELDS[name](output[name])
        for name in ("allow_bullets", "allow_impression", "allow_recommendations"):
            if name in output:
                kwargs[name] = bool(output[name])
    budget = block.pop("token_budget", None)
    if isinstance(budget, Mapping):
        for name in ("preserve_visual_tokens", "truncate_negatives_first"):
            if name in budget:
                kwargs[name] = bool(budget[name])

    if block:
        raise PromptConfigError(f"unknown prompt config key(s): {', '.join(sorted(block))}")
    return PromptConfig(**kwargs)


def load_prompt_config(path: str | Path) -> PromptConfig:
    """Load a ``PromptConfig`` from the ``prompt:`` block of a YAML file."""
    import yaml

    text = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(text) or {}
    if not isinstance(data, Mapping):
        raise PromptConfigError(f"{path}: top-level YAML must be a mapping")
    block = data.get("prompt", data)
    if not isinstance(block, Mapping):
        raise PromptConfigError(f"{path}: 'prompt' must be a mapping")
    return config_from_mapping(block)
