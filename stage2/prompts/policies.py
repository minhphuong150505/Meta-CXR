"""Policy enums and the selection logic that turns Stage-1 classes into prompt text.

Every function here is pure and torch-free. The guiding rule: Stage-1 output is a
*prediction*, so no policy may promote uncertain→positive, none may assert a
negative as ground-truth fact, and a normal summary must still leave room for the
image to override it.
"""

from __future__ import annotations

import re
from enum import Enum

from .ontology import validate_critical_findings
from .schemas import PromptContext


class NormalPolicy(str, Enum):
    COMPACT_SUMMARY = "compact_summary"
    CRITICAL_NEGATIVES = "critical_negatives"
    ALL_NEGATIVES = "all_negatives"
    NO_STRUCTURED_NORMAL_STATEMENT = "no_structured_normal_statement"


class NegativePolicy(str, Enum):
    NONE = "none"
    ALL = "all"
    CRITICAL_ONLY = "critical_only"
    TOP_K_CONFIDENT = "top_k_confident"
    THRESHOLDED = "thresholded"
    NORMAL_SUMMARY = "normal_summary"


class UncertaintyPolicy(str, Enum):
    EXPLICIT_POSSIBLE = "explicit_possible"
    PROBABILITY_BINS = "probability_bins"
    OMIT_LOW_CONFIDENCE = "omit_low_confidence"
    PRESERVE_THREE_CLASS = "preserve_three_class"


class TemporalTargetPolicy(str, Enum):
    KEEP = "keep"
    REMOVE_TEMPORAL_CLAUSES = "remove_temporal_clauses"
    EXCLUDE_SAMPLE = "exclude_sample"
    REQUIRE_PRIOR_CONTEXT = "require_prior_context"


# Confidence-bin cut points. These are placeholders that MUST be re-derived on the
# validation set before ``probability_bins`` is used in a real run; using raw or
# uncalibrated probabilities here would put unjustified confidence words in the
# prompt. Documented in stage2_prompt_design.md.
HIGH_CONFIDENCE = 0.85
MODERATE_CONFIDENCE = 0.60


def confidence_bin(probability: float) -> str:
    if probability >= HIGH_CONFIDENCE:
        return "high confidence"
    if probability >= MODERATE_CONFIDENCE:
        return "moderate confidence"
    return "low confidence"


# Temporal / interval-change cues. Word-boundaried and case-insensitive. Used both
# by the prompt guard wording and by the target audit / rewrite policy. This is a
# lexical heuristic, not a clinical judgement -- see stage2_temporal_target_audit.md.
_TEMPORAL_PATTERNS = [
    r"compared (?:with|to) (?:the )?prior",
    r"comparison (?:with|to) (?:the )?prior",
    r"since (?:the )?prior",
    r"new since",
    r"interval (?:change|increase|decrease|development|worsening|improvement|resolution)",
    r"previously (?:seen|noted|demonstrated|identified)",
    r"prior (?:study|exam|examination|radiograph|imaging|chest)",
    r"\bunchanged\b",
    r"\bre-?demonstrated\b",
    r"\bre-?evaluation\b",
    r"no significant interval",
]
_TEMPORAL_RE = re.compile("|".join(_TEMPORAL_PATTERNS), re.IGNORECASE)

# These words are temporal only in context (e.g. "stable" as in "stable cardiac
# silhouette" can be non-temporal). They are matched at the clause level in
# remove_temporal_clauses but reported separately so the heuristic stays honest.
_TEMPORAL_SOFT_RE = re.compile(
    r"\b(stable|improved|improving|worsened|worsening|resolved|resolving|"
    r"decreased|increased|persistent|redemonstration)\b",
    re.IGNORECASE,
)


def contains_temporal_language(text: str) -> bool:
    return bool(_TEMPORAL_RE.search(text or "") or _TEMPORAL_SOFT_RE.search(text or ""))


def _cap(names: tuple[str, ...], limit: int) -> tuple[str, ...]:
    return names if limit <= 0 else names[:limit]


def select_negative_findings(
    context: PromptContext,
    policy: NegativePolicy,
    *,
    max_negative_findings: int,
    negative_probability_threshold: float,
    critical_findings: tuple[str, ...],
) -> tuple[str, ...]:
    """Choose which absent findings to state, per policy. Never raises on a
    missing pathology or missing probabilities; falls back deterministically."""
    negatives = tuple(context.negative_findings)
    if policy is NegativePolicy.NONE:
        return ()
    if policy is NegativePolicy.ALL:
        return _cap(negatives, max_negative_findings)
    if policy in (NegativePolicy.CRITICAL_ONLY, NegativePolicy.NORMAL_SUMMARY):
        critical = validate_critical_findings(critical_findings)
        selected = tuple(name for name in negatives if name in critical)
        return _cap(selected, max_negative_findings)
    if policy is NegativePolicy.TOP_K_CONFIDENT:
        probs = context.negative_probabilities
        if probs:
            ranked = sorted(negatives, key=lambda n: (-probs.get(n, 0.0), n))
        else:
            ranked = sorted(negatives)  # deterministic when probabilities absent
        limit = max_negative_findings if max_negative_findings > 0 else len(ranked)
        return tuple(ranked[:limit])
    if policy is NegativePolicy.THRESHOLDED:
        probs = context.negative_probabilities
        selected = tuple(
            name
            for name in negatives
            if probs.get(name, 0.0) >= negative_probability_threshold
        )
        return _cap(selected, max_negative_findings)
    raise ValueError(f"unhandled negative policy {policy!r}")


def render_uncertain(
    context: PromptContext,
    policy: UncertaintyPolicy,
    *,
    low_confidence_threshold: float,
) -> tuple[str, ...]:
    """Render uncertain findings under the chosen policy.

    Returns display strings (a finding may be annotated with a confidence bin).
    Uncertain is never promoted to positive here.
    """
    uncertains = tuple(context.uncertain_findings)
    if not uncertains:
        return ()  # nothing to render; never demand probabilities for an empty set
    if policy is UncertaintyPolicy.PROBABILITY_BINS:
        probs = context.uncertain_probabilities
        if not probs:
            raise ValueError(
                "uncertainty_policy=probability_bins requires "
                "uncertain_probabilities calibrated on the validation set"
            )
        return tuple(
            f"{name} ({confidence_bin(probs.get(name, 0.0))})" for name in uncertains
        )
    if policy is UncertaintyPolicy.OMIT_LOW_CONFIDENCE:
        probs = context.uncertain_probabilities
        if not probs:
            return uncertains  # nothing to rank on; keep all, documented
        return tuple(
            name for name in uncertains if probs.get(name, 0.0) >= low_confidence_threshold
        )
    # EXPLICIT_POSSIBLE and PRESERVE_THREE_CLASS render the names as-is; the
    # cautious framing lives in the template's line label and instruction.
    return uncertains


def apply_temporal_target_policy(
    target: str, prior_available: bool, policy: TemporalTargetPolicy
) -> tuple[str | None, str]:
    """Apply a temporal policy to a *training target*. Returns (new_target, action).

    ``new_target is None`` means "exclude this sample". ``keep`` is the identity
    (backward compatible). This is deliberately not auto-applied by training; a
    run opts in via config and the choice is recorded. See
    stage2_temporal_target_audit.md.
    """
    if policy is TemporalTargetPolicy.KEEP or prior_available:
        return target, "kept"
    if not contains_temporal_language(target):
        return target, "kept"
    if policy is TemporalTargetPolicy.EXCLUDE_SAMPLE:
        return None, "excluded"
    if policy is TemporalTargetPolicy.REQUIRE_PRIOR_CONTEXT:
        # The run must supply prior context for this sample or drop it; the
        # policy does not silently rewrite. Signalled to the caller.
        return None, "requires_prior_context"
    if policy is TemporalTargetPolicy.REMOVE_TEMPORAL_CLAUSES:
        return _strip_temporal_sentences(target), "temporal_clauses_removed"
    raise ValueError(f"unhandled temporal policy {policy!r}")


def _strip_temporal_sentences(text: str) -> str:
    """Drop whole sentences that contain a temporal cue. Sentence-level, not
    word-level, so we never leave a dangling half clause. Heuristic and lossy --
    documented as such."""
    sentences = re.split(r"(?<=[.!?])\s+", text or "")
    kept = [s for s in sentences if s and not contains_temporal_language(s)]
    return " ".join(kept).strip()
