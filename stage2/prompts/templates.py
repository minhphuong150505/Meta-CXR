"""Versioned prompt text fragments and template hashing.

All user-facing wording lives here as module constants so a single ``template_hash``
covers every string that can reach the model. Changing any fragment changes the
hash, which is what makes a run's prompt reproducible from its recorded metadata.
"""

from __future__ import annotations

import hashlib

from .schemas import VisualMode

TEMPLATE_VERSION = "stage2_prompt_v2"

NONE_TOKEN = "none"

# --- section headers -------------------------------------------------------
VISUAL_HEADER = "Visual study features:"
STRUCTURED_HEADER = "Auxiliary Stage-1 predictions, which may be imperfect:"
CONTEXT_HEADER = "Study context:"
PRESENT_LABEL = "Present"
POSSIBLE_LABEL = "Possible or uncertain"
ABSENT_LABEL = "Clinically relevant absent"

COMPACT_NORMAL_STATEMENT = (
    "No high-confidence positive or uncertain abnormality was predicted among the "
    "modeled findings."
)

# --- instruction bodies ----------------------------------------------------
TASK_LINE = (
    "Generate only the FINDINGS section for the current chest radiograph study."
)

EVIDENCE_PRIMARY_VISUAL = (
    "Use the visual study features as the primary evidence. Treat the structured "
    "predictions only as auxiliary cues that may be wrong. Report supported "
    "abnormalities and clinically relevant negative findings. Do not repeat a "
    "prediction that the image does not support, and do not omit a finding the "
    "image shows merely because it was predicted absent. Include laterality, "
    "location, severity, extent and support-device position only when supported "
    "by the available evidence, and never infer them from a finding name alone."
)

EVIDENCE_NEUTRAL = (
    "Use the image and the structured predictions together to write the findings. "
    "Include laterality, location, severity, extent and support-device position "
    "only when supported by the available evidence."
)

EVIDENCE_UNCERTAIN = (
    "Express possible or uncertain findings cautiously. Do not convert an "
    "uncertain prediction into a definite finding without visual support."
)

NATIVE_ONLY_BODY = (
    "Use only the provided image or images and available study context. Describe "
    "supported positive findings and clinically relevant negative findings. "
    "Include laterality, location, severity, extent and support-device position "
    "only when visible or supported. Use cautious wording for equivocal findings."
)

NO_PRIOR_GUARD = (
    "If prior comparison is unavailable, do not state or imply that a finding is "
    "new, improved, worsened, stable or unchanged."
)

FORBID_META = (
    "Do not mention prediction labels, confidence categories, model outputs, the "
    "prompt, an Impression section, recommendations or the task itself."
)

FORBID_META_NATIVE = (
    "Do not output an Impression section, recommendations, bullet points, patient "
    "history, prediction labels or discussion of the task."
)

# All fragments that participate in the template hash, in a stable order.
_HASH_FRAGMENTS = (
    TEMPLATE_VERSION,
    VISUAL_HEADER,
    STRUCTURED_HEADER,
    CONTEXT_HEADER,
    PRESENT_LABEL,
    POSSIBLE_LABEL,
    ABSENT_LABEL,
    COMPACT_NORMAL_STATEMENT,
    TASK_LINE,
    EVIDENCE_PRIMARY_VISUAL,
    EVIDENCE_NEUTRAL,
    EVIDENCE_UNCERTAIN,
    NATIVE_ONLY_BODY,
    NO_PRIOR_GUARD,
    FORBID_META,
    FORBID_META_NATIVE,
)


def sentence_constraint(min_sentences: int, max_sentences: int) -> str:
    if min_sentences == max_sentences:
        count = f"{min_sentences} sentence" + ("" if min_sentences == 1 else "s")
        return f"Return one concise clinical paragraph of {count}."
    return (
        "Return one concise clinical paragraph of "
        f"{min_sentences} to {max_sentences} sentences."
    )


def join_or_none(names: tuple[str, ...]) -> str:
    return ", ".join(names) if names else NONE_TOKEN


def template_hash(visual_mode: VisualMode, length: int = 16) -> str:
    payload = "\x1f".join((*_HASH_FRAGMENTS, visual_mode.value))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]
