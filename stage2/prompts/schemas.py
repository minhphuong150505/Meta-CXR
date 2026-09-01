"""Typed inputs/outputs for the Stage-2 prompt builder.

Torch-free by design: the builder must import on a CPU box so its correctness,
parity and leakage invariants are unit-testable without MedGemma weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class VisualMode(str, Enum):
    """The visual channel and whether structured cues are included.

    Making these explicit fixes the previously-dead ``uses_mhcac_prompt``:
    ``qformer_visual_only`` and ``qformer_guided`` are now distinct, so the
    "visual only" experiment can no longer silently carry Stage-1 labels.
    """

    NATIVE_ANCHOR_ONLY = "native_anchor_only"
    NATIVE_ANCHOR_GUIDED = "native_anchor_guided"
    NATIVE_MULTIVIEW = "native_multiview"
    QFORMER_VISUAL_ONLY = "qformer_visual_only"
    QFORMER_GUIDED = "qformer_guided"
    #: The originally-designed architecture: MedGemma keeps its own vision tower
    #: AND additionally receives the 32 Q-Former soft tokens, with MHCAC cues in
    #: the text. The `qformer_*` modes above SUBSTITUTE soft tokens for the
    #: image; this one supplements it, which is a different claim -- "the soft
    #: tokens add something the vision tower missed" rather than "the soft
    #: tokens are the visual channel".
    NATIVE_QFORMER_GUIDED = "native_qformer_guided"

    @property
    def uses_soft_tokens(self) -> bool:
        return self in (
            VisualMode.QFORMER_VISUAL_ONLY,
            VisualMode.QFORMER_GUIDED,
            VisualMode.NATIVE_QFORMER_GUIDED,
        )

    @property
    def uses_native_image(self) -> bool:
        return self in (
            VisualMode.NATIVE_ANCHOR_ONLY,
            VisualMode.NATIVE_ANCHOR_GUIDED,
            VisualMode.NATIVE_MULTIVIEW,
            VisualMode.NATIVE_QFORMER_GUIDED,
        )

    @property
    def includes_structured(self) -> bool:
        """Whether Stage-1 P/N/U cues are placed in the prompt at all."""
        return self in (
            VisualMode.NATIVE_ANCHOR_GUIDED,
            VisualMode.NATIVE_MULTIVIEW,
            VisualMode.QFORMER_GUIDED,
            VisualMode.NATIVE_QFORMER_GUIDED,
        )

    @property
    def includes_aux_image(self) -> bool:
        return self is VisualMode.NATIVE_MULTIVIEW

    @property
    def image_mode(self) -> str:
        """Storage key for the adapter directory and the runner's branch.

        The first two values are legacy and are kept exactly so existing adapter
        dirs stay loadable. ``native_qformer`` is the combined channel and is
        deliberately a THIRD value rather than a flag on one of the others: the
        runner branches on this string in a dozen places, and reusing "native"
        or "qformer" for a mode that does both would have made every one of
        those branches silently half-right.
        """
        if self.uses_native_image and self.uses_soft_tokens:
            return "native_qformer"
        return "qformer" if self.uses_soft_tokens else "native"


@dataclass(frozen=True)
class PromptContext:
    """Everything the builder needs about one study. Absent facts are None/empty.

    Findings are the *Stage-1 predicted* class lists for this sample (auxiliary
    cues), never the ground-truth label. Probability maps are optional and, when
    present, key a finding name to a calibrated probability in [0, 1].
    """

    study_id: str
    visual_mode: VisualMode
    positive_findings: tuple[str, ...] = ()
    uncertain_findings: tuple[str, ...] = ()
    negative_findings: tuple[str, ...] = ()
    positive_probabilities: dict[str, float] = field(default_factory=dict)
    uncertain_probabilities: dict[str, float] = field(default_factory=dict)
    negative_probabilities: dict[str, float] = field(default_factory=dict)
    anchor_view: str | None = None
    auxiliary_views: tuple[str, ...] = ()
    prior_available: bool = False
    comparison_available: bool = False
    indication: str | None = None
    technique: str | None = None
    has_support_devices: bool = False
    qformer_token_count: int | None = None
    prompt_version: str | None = None

    @property
    def is_structurally_normal(self) -> bool:
        """No positive and no uncertain finding was predicted.

        This is a statement about the Stage-1 *prediction*, not about the study.
        The prompt must still let visual evidence override it (Stage-1 false
        negative), which is why normal policies never assert "No Finding" as fact.
        """
        return not self.positive_findings and not self.uncertain_findings


class PartKind(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    SOFT_TOKENS = "soft_tokens"


@dataclass(frozen=True)
class PromptPart:
    """One model-agnostic piece of the user turn.

    ``SOFT_TOKENS`` carries a count; the integration layer expands it to that
    many copies of the tokenizer's single special token. ``IMAGE`` carries an
    optional role tag (``anchor``/``auxiliary``); the caller supplies the pixels.
    """

    kind: PartKind
    text: str | None = None
    count: int | None = None
    image_role: str | None = None
    #: Truncation order. 0 = never drop (visual tokens, instruction, present /
    #: possible cues). Higher drops first, so the negative list (2) is cut before
    #: auxiliary metadata (1), and neither before the assistant target.
    budget_priority: int = 0


@dataclass(frozen=True)
class RenderedPrompt:
    """Builder output. Carries reproducibility metadata but no patient text hash."""

    parts: tuple[PromptPart, ...]
    prompt_version: str
    template_hash: str
    config_hash: str
    prompt_hash: str

    def user_text(self, soft_token: str = "<qformer_soft_token>") -> str:
        """Flatten to a single user string (Q-Former / Vicuna string path).

        Image parts render to nothing (pixels are supplied out of band); soft
        tokens expand to ``count`` copies of ``soft_token``.
        """
        chunks: list[str] = []
        for part in self.parts:
            if part.kind is PartKind.TEXT and part.text:
                chunks.append(part.text)
            elif part.kind is PartKind.SOFT_TOKENS and part.count:
                chunks.append(" ".join([soft_token] * part.count))
        return "\n\n".join(chunks)
