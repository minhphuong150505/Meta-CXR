"""Multimodal capability validation for the native MedGemma pipeline.

The bug this module exists to prevent: loading MedGemma used to fall back from
``AutoModelForImageTextToText`` to ``AutoModelForCausalLM`` inside a bare
``except Exception``. If the multimodal class failed for *any* reason -- a
missing processor file, an auth error, an OOM, a transformers version skew --
the run continued as a **text-only language model** and produced reports from
the language prior alone.

Nothing downstream could tell. The loss fell, generation succeeded, and the NLG
metrics looked normal, because MIMIC-CXR findings text is repetitive enough
that a prior-only model scores respectably. The "native MedGemma image tower"
row of the ablation table would have been a language-prior baseline wearing a
vision baseline's name -- and it would have been compared against the Q-Former
row as though the only difference were the visual pathway.

That is a research-validity failure, not a robustness nicety. So:

* ``medgemma_direct`` requires a genuinely multimodal model and fails loudly.
* A text-only run is still available, but only by explicitly selecting
  ``text_only_language_prior_ablation``, which is named for what it measures.
* Validation inspects the processor, the model config and the accepted forward
  signature -- not just the class name, which a subclass or a wrapper can
  satisfy without having a vision tower.

Depends on nothing but the standard library, so it is testable on CPU with
fakes and never imports transformers at module scope.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any

#: Config attributes that indicate a vision tower is present.
VISION_CONFIG_ATTRS = ("vision_config", "vision_tower_config", "image_encoder_config")

#: Model attributes that indicate multimodal components were constructed.
MULTIMODAL_MODULE_ATTRS = (
    "vision_tower",
    "vision_model",
    "multi_modal_projector",
    "image_tower",
)

#: Batch keys that carry actual pixels.
PIXEL_KEYS = ("pixel_values", "image", "images", "pixel_values_videos")


class MultimodalModelLoadError(RuntimeError):
    """Native MedGemma could not be loaded as a multimodal model.

    Carries everything needed to diagnose the failure without re-running, and
    -- critically -- replaces what used to be a silent downgrade to a text-only
    model.
    """

    def __init__(
        self,
        *,
        model_id: str,
        revision: str | None,
        transformers_version: str | None,
        original: BaseException | None,
        detail: str = "",
    ):
        self.model_id = model_id
        self.revision = revision
        self.transformers_version = transformers_version
        self.original = original
        message = (
            f"could not load {model_id!r} as a multimodal model.\n"
            f"  revision:             {revision or '<default>'}\n"
            f"  transformers version: {transformers_version or '<unknown>'}\n"
            f"  original error:       {type(original).__name__ if original else 'n/a'}: "
            f"{original if original else 'n/a'}\n"
        )
        if detail:
            message += f"  detail:               {detail}\n"
        message += (
            "\nnative MedGemma (pipeline mode 'medgemma_direct') REQUIRES multimodal "
            "capability: the image is its only clinical input. Falling back to a "
            "text-only causal LM would silently turn this run into a language-prior "
            "baseline while still labelling it a vision baseline, invalidating any "
            "comparison against the Q-Former pipeline.\n"
            "\nIf you intended to measure the language prior, select the pipeline mode "
            "'text_only_language_prior_ablation' explicitly.\n"
            "\nOtherwise check: transformers is recent enough to expose "
            "AutoModelForImageTextToText for this architecture; HF_TOKEN grants access "
            "to the gated repo; and the processor files downloaded completely."
        )
        super().__init__(message)


@dataclass
class CapabilityReport:
    """Result of inspecting a loaded model/processor pair."""

    multimodal: bool
    checks: dict[str, bool] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)

    def as_metadata(self) -> dict[str, Any]:
        """The block written into run manifests and result JSON."""
        return {
            "multimodal": self.multimodal,
            "capability_checks": dict(self.checks),
            "capability_failures": list(self.failures),
        }


def _has_image_processor(processor: Any) -> bool:
    """True when the processor can actually turn a PIL image into pixels."""
    if processor is None:
        return False
    for attr in ("image_processor", "feature_extractor"):
        component = getattr(processor, attr, None)
        if component is not None:
            return True
    # Some processors expose no named sub-component but do accept `images=`.
    if not callable(processor):
        return False
    try:
        return "images" in inspect.signature(processor).parameters
    except (TypeError, ValueError):
        # Builtin/C-implemented callables have no introspectable signature.
        return False


def _has_vision_config(model: Any) -> bool:
    config = getattr(model, "config", None)
    if config is None:
        return False
    return any(getattr(config, attr, None) is not None for attr in VISION_CONFIG_ATTRS)


def _has_multimodal_modules(model: Any) -> bool:
    return any(getattr(model, attr, None) is not None for attr in MULTIMODAL_MODULE_ATTRS)


def _forward_accepts_pixels(model: Any) -> bool:
    forward = getattr(model, "forward", None)
    if forward is None:
        return False
    try:
        parameters = inspect.signature(forward).parameters
    except (TypeError, ValueError):
        return False
    if any(key in parameters for key in PIXEL_KEYS):
        return True
    # A **kwargs forward cannot be shown to reject pixels by signature alone;
    # the config and module checks carry the decision in that case.
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())


def _is_pure_causal_lm(model: Any) -> bool:
    """A text-only causal LM: no vision config, no multimodal submodules."""
    return not _has_vision_config(model) and not _has_multimodal_modules(model)


class MultimodalCapabilityValidator:
    """Checks that a loaded model really can consume images.

    Class-name checks are deliberately not sufficient: a wrapper, a PEFT-wrapped
    model or a subclass can present the right name with no vision tower behind
    it, which is exactly the situation the old fallback produced.
    """

    def inspect(self, model: Any, processor: Any) -> CapabilityReport:
        checks = {
            "processor_has_image_processor": _has_image_processor(processor),
            "model_has_vision_config": _has_vision_config(model),
            "model_has_multimodal_modules": _has_multimodal_modules(model),
            "forward_accepts_pixel_values": _forward_accepts_pixels(model),
            "not_pure_causal_lm": not _is_pure_causal_lm(model),
        }
        failures = [name for name, passed in checks.items() if not passed]
        return CapabilityReport(multimodal=not failures, checks=checks, failures=failures)

    def require_multimodal(
        self,
        model: Any,
        processor: Any,
        *,
        model_id: str,
        revision: str | None = None,
        transformers_version: str | None = None,
    ) -> CapabilityReport:
        """Return the report, or raise ``MultimodalModelLoadError``."""
        report = self.inspect(model, processor)
        if not report.multimodal:
            raise MultimodalModelLoadError(
                model_id=model_id,
                revision=revision,
                transformers_version=transformers_version,
                original=None,
                detail=(
                    "the model loaded, but failed capability checks: "
                    + ", ".join(report.failures)
                ),
            )
        return report


def validate_multimodal_capability(
    model: Any,
    processor: Any,
    *,
    model_id: str,
    revision: str | None = None,
    transformers_version: str | None = None,
) -> CapabilityReport:
    """Module-level entry point wrapping :class:`MultimodalCapabilityValidator`."""
    return MultimodalCapabilityValidator().require_multimodal(
        model,
        processor,
        model_id=model_id,
        revision=revision,
        transformers_version=transformers_version,
    )
