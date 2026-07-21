"""Failure modes for the external MedGemma checkpoints.

Every error here is raised instead of degrading to a weaker pipeline. A loader
that cannot produce a multimodal model must not fall back to a text-only one:
the resulting reports would look plausible while never having seen a pixel,
which is the single most expensive failure this project can make.
"""

from __future__ import annotations


class PretrainedMedGemmaError(RuntimeError):
    """Base class for external-checkpoint failures."""


class FindingsModelLoadError(PretrainedMedGemmaError):
    """The Findings checkpoint, processor or dtype could not be prepared."""


class NotMultimodalError(PretrainedMedGemmaError):
    """The loaded artifact lacks a vision tower or an image processor.

    Raised before any generation happens. Silently continuing would produce a
    language-prior report presented as an image-conditioned one.
    """


class ImpressionPhaseDisabledError(PretrainedMedGemmaError):
    """Impression generation was requested during the Findings-first phase.

    Impression is a second GPU-hour budget that has not been approved. This is
    a hard stop with no override flag: enabling it is a config change plus an
    explicit command in a later session.
    """
