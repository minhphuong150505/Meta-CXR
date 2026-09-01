"""Explicit pipeline-mode names for Stage 2.

The older ``--image-mode {qformer,native}`` flag named an implementation detail
rather than an architecture, which made it easy to describe the Q-Former hybrid
as "native MedGemma" in write-ups. Each mode here states which visual pathway is
used and whether Stage 1 is required at all.

``medgemma_direct`` is the default pipeline: MedGemma's own image tower and
multimodal projector, no Stage-1 checkpoint, no Q-Former, no MHCAC, and no
structured abnormality text in the prompt. The image is the only clinical
evidence at inference.

This module is intentionally stdlib-only so mode resolution stays testable on a
CPU box without importing torch, transformers or LAVIS.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineMode:
    """Static description of one Stage-2 architecture."""

    name: str
    # Storage-level key. Kept as the legacy "native"/"qformer" strings so
    # existing adapter directories, meta.json and manifest.json stay loadable.
    image_mode: str
    requires_stage1: bool
    uses_mhcac_prompt: bool
    description: str
    #: Whether the model must genuinely consume pixels. Only the deliberate
    #: language-prior ablation sets this False; every other mode fails closed if
    #: the loaded model turns out to be text-only.
    requires_multimodal: bool = True


MEDGEMMA_DIRECT = PipelineMode(
    name="medgemma_direct",
    image_mode="native",
    requires_stage1=False,
    uses_mhcac_prompt=False,
    description=(
        "Native MedGemma image tower + multimodal projector + Gemma decoder. "
        "No Stage-1 checkpoint, Q-Former, MHCAC or structured findings in the prompt."
    ),
)

META_CXR_QFORMER = PipelineMode(
    name="meta_cxr_qformer",
    image_mode="qformer",
    requires_stage1=True,
    uses_mhcac_prompt=False,
    description=(
        "Hybrid ablation: META-CXR encoders + Q-Former 32 query tokens projected "
        "into MedGemma's embedding space as soft tokens. Not native MedGemma."
    ),
)

META_CXR_QFORMER_WITH_MHCAC_PROMPT = PipelineMode(
    name="meta_cxr_qformer_with_mhcac_prompt",
    image_mode="qformer",
    requires_stage1=True,
    uses_mhcac_prompt=True,
    description=(
        "Hybrid ablation as above, plus MHCAC positive/negative/uncertain "
        "findings injected into the prompt as text."
    ),
)

META_CXR_NATIVE_QFORMER_GUIDED = PipelineMode(
    name="meta_cxr_native_qformer_guided",
    image_mode="native_qformer",
    requires_stage1=True,
    uses_mhcac_prompt=True,
    description=(
        "The originally-designed architecture: MedGemma keeps its OWN vision "
        "tower on the anchor image AND additionally receives the 32 Q-Former "
        "soft tokens, with MHCAC positive/negative/uncertain findings in the "
        "text. The two meta_cxr_qformer* modes above SUBSTITUTE soft tokens for "
        "the image; this one supplements it, so the comparison against "
        "medgemma_direct measures what the soft tokens and cues ADD on top of "
        "MedGemma's own vision -- not whether they can replace it."
    ),
)

TEXT_ONLY_LANGUAGE_PRIOR_ABLATION = PipelineMode(
    name="text_only_language_prior_ablation",
    image_mode="text_only",
    requires_stage1=False,
    uses_mhcac_prompt=False,
    requires_multimodal=False,
    description=(
        "DIAGNOSTIC ABLATION, NOT A VISION PIPELINE. Gemma decoder with no image "
        "input at all: measures how much of the report is recoverable from the "
        "language prior and the prompt alone. This is the floor that a real "
        "vision pipeline must beat. It must never be reported as 'native "
        "MedGemma' or compared as though it used the image."
    ),
)

PRETRAINED_MEDGEMMA_FINDINGS_FIRST = PipelineMode(
    name="pretrained_medgemma_findings_first",
    image_mode="native",
    requires_stage1=False,
    uses_mhcac_prompt=False,
    description=(
        "Inference only over the EXTERNAL erjui/medgemma-4b-srrg-findings "
        "checkpoint. Fine-tuned by a third party from google/medgemma-4b-it on "
        "MIMIC-CXR + CheXpert+ derived data, not by this project and not on "
        "this repository's splits. Generates FINDINGS only; Impression is a "
        "separate unbudgeted phase. Run via "
        "medgemma_inference.run_pretrained_findings."
    ),
)

PRETRAINED_MEDGEMMA_IMPRESSION_PHASE2 = PipelineMode(
    name="pretrained_medgemma_impression_phase2",
    image_mode="native",
    requires_stage1=False,
    uses_mhcac_prompt=False,
    description=(
        "DECLARED BUT DISABLED. Phase-2 Impression generation over the external "
        "erjui/medgemma-4b-srrg-impression checkpoint. Blocked by a runtime "
        "guard until the Findings pilot cost is measured and the additional "
        "spend is explicitly approved. There is no implementation behind this "
        "name yet."
    ),
)

PIPELINE_MODES = {
    mode.name: mode
    for mode in (
        MEDGEMMA_DIRECT,
        META_CXR_QFORMER,
        META_CXR_QFORMER_WITH_MHCAC_PROMPT,
        META_CXR_NATIVE_QFORMER_GUIDED,
        TEXT_ONLY_LANGUAGE_PRIOR_ABLATION,
        PRETRAINED_MEDGEMMA_FINDINGS_FIRST,
        PRETRAINED_MEDGEMMA_IMPRESSION_PHASE2,
    )
}

DEFAULT_PIPELINE_MODE = MEDGEMMA_DIRECT.name

# Runs both the primary pipeline and the hybrid ablation, in that order.
ABLATION_MODE = "both_for_ablation"

# Legacy ``--image-mode`` values, retained so existing scripts and runbooks keep
# working. They map onto the explicit names rather than being interpreted.
LEGACY_IMAGE_MODE_ALIASES = {
    "native": MEDGEMMA_DIRECT.name,
    "qformer": META_CXR_QFORMER.name,
    "both": ABLATION_MODE,
}

#: Modes served by ``medgemma_inference/``, not by this Stage-2 entrypoint.
#: They are registered above so the architecture list is complete and
#: documented in one place, but they are deliberately excluded from ``CHOICES``
#: and rejected by ``resolve_pipeline_modes``: an external inference run must
#: not be reachable through a fine-tuning CLI.
EXTERNAL_INFERENCE_MODES = frozenset(
    {
        PRETRAINED_MEDGEMMA_FINDINGS_FIRST.name,
        PRETRAINED_MEDGEMMA_IMPRESSION_PHASE2.name,
    }
)

CHOICES = (
    *(name for name in PIPELINE_MODES if name not in EXTERNAL_INFERENCE_MODES),
    ABLATION_MODE,
)


def resolve_pipeline_modes(selection: str) -> list[PipelineMode]:
    """Return the ordered pipeline modes for a CLI selection.

    ``both_for_ablation`` runs the primary pipeline first so a crash during the
    ablation still leaves the primary result on disk.
    """
    if selection in EXTERNAL_INFERENCE_MODES:
        raise ValueError(
            f"{selection!r} is an external-checkpoint inference mode and is not "
            "runnable from this entrypoint. Use: python -m "
            "medgemma_inference.run_pretrained_findings --config "
            "configs/experiments/pretrained_medgemma_findings_first.yaml"
        )
    if selection == ABLATION_MODE:
        return [MEDGEMMA_DIRECT, META_CXR_QFORMER]
    if selection in PIPELINE_MODES:
        return [PIPELINE_MODES[selection]]
    if selection in LEGACY_IMAGE_MODE_ALIASES:
        return resolve_pipeline_modes(LEGACY_IMAGE_MODE_ALIASES[selection])
    raise ValueError(
        f"unknown pipeline mode {selection!r}; expected one of {', '.join(CHOICES)}"
    )


def requires_stage1(modes: list[PipelineMode]) -> bool:
    """True when any selected mode needs a Stage-1 checkpoint and config."""
    return any(mode.requires_stage1 for mode in modes)


def requires_multimodal(mode: PipelineMode) -> bool:
    """True when the loaded model must genuinely accept pixels.

    False only for ``text_only_language_prior_ablation``, which a user has to
    name explicitly. There is no path by which a mode silently becomes
    text-only: ``both_for_ablation`` does not include it, and it is not the
    default.
    """
    return mode.requires_multimodal
