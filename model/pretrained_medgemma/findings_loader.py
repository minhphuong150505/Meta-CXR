"""Loader for the external Findings checkpoint.

``erjui/medgemma-4b-srrg-findings`` is a third-party checkpoint fine-tuned from
``google/medgemma-4b-it`` on the csrrg_ift dataset (MIMIC-CXR + CheXpert+). This
project does not train it and does not claim it.

The published repository ships *merged* weights -- ``model-0000X-of-00002.safetensors``
plus an index, with no ``adapter_config.json`` -- so PEFT is not needed to load
it. The ``lora`` tag on the model card describes how the authors trained it, not
how it is distributed.

The processor is read from the checkpoint repository itself rather than from
``google/medgemma-4b-it``: the repo ships its own tokenizer, ``added_tokens.json``
and ``preprocessor_config.json``, and the base repo is gated. Loading locally
keeps the processor consistent with the weights and avoids a gated dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from model.pretrained_medgemma.errors import FindingsModelLoadError, NotMultimodalError
from runtime.device import DevicePlan, plan_device

FINDINGS_MODEL_ID = "erjui/medgemma-4b-srrg-findings"
#: Architecture the checkpoint declares in config.json. Used as an assertion,
#: not as a lookup: a checkpoint that stopped being multimodal must not load.
EXPECTED_ARCHITECTURE = "Gemma3ForConditionalGeneration"


@dataclass
class FindingsModelBundle:
    """A loaded, eval-mode multimodal model plus its processor and provenance."""

    model: Any
    processor: Any
    model_id: str
    revision: str
    resolved_revision: str
    device_plan: DevicePlan

    def provenance(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "resolved_revision": self.resolved_revision,
            "device": self.device_plan.device,
            "torch_dtype": self.device_plan.torch_dtype,
            "load_in_4bit": self.device_plan.load_in_4bit,
            "external_checkpoint": True,
            "fine_tuned_by_this_project": False,
        }


def _assert_has_image_processor(processor: Any, model_id: str) -> None:
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        raise NotMultimodalError(
            f"processor for {model_id!r} exposes no image_processor, so images "
            "could not be encoded. Refusing to continue: text-only generation "
            "would produce reports that never saw the X-ray."
        )


def _assert_has_vision_tower(model: Any, model_id: str) -> None:
    config = getattr(model, "config", None)
    vision_config = getattr(config, "vision_config", None)
    has_module = any(
        getattr(model, name, None) is not None
        for name in ("vision_tower", "vision_model", "visual")
    )
    if vision_config is None and not has_module:
        raise NotMultimodalError(
            f"{model_id!r} loaded without a vision config or vision module. "
            "Refusing to continue: this pipeline requires the image to be "
            "genuine evidence, not a language prior."
        )


class PretrainedFindingsLoader:
    """Loads the Findings checkpoint, and nothing else.

    This class has no knowledge of the Impression checkpoint. There is no code
    path by which calling it can download, construct or allocate VRAM for a
    second model.
    """

    def __init__(
        self,
        *,
        model_id: str = FINDINGS_MODEL_ID,
        revision: str = "main",
        device: str = "auto",
        dtype: str = "auto",
        load_in_4bit: bool = False,
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self.device = device
        self.dtype = dtype
        self.load_in_4bit = load_in_4bit

    def _quantization_config(self, plan: DevicePlan):
        if not plan.load_in_4bit:
            return None
        import torch
        from transformers import BitsAndBytesConfig

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=getattr(torch, plan.torch_dtype),
        )

    def _model_class(self):
        """Return a multimodal auto-class, never a text-only one.

        ``AutoModelForImageTextToText`` is the current name for this task;
        ``AutoModelForVision2Seq`` is the older alias the model card uses. Both
        are multimodal, so preferring one over the other is a transformers
        version detail, not a capability downgrade. ``AutoModelForCausalLM`` is
        deliberately absent.
        """
        import transformers

        for name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq"):
            cls = getattr(transformers, name, None)
            if cls is not None:
                return cls
        raise FindingsModelLoadError(
            "installed transformers exposes neither AutoModelForImageTextToText "
            "nor AutoModelForVision2Seq. Gemma3 multimodal loading needs "
            "transformers>=4.50; upgrade rather than falling back to a "
            "text-only class."
        )

    def load(self) -> FindingsModelBundle:
        """Download (if needed), load and verify the Findings model."""
        plan = plan_device(
            device=self.device, dtype=self.dtype, load_in_4bit=self.load_in_4bit
        )
        try:
            import torch
            from transformers import AutoProcessor
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise FindingsModelLoadError(
                "transformers and torch are required to load the Findings model."
            ) from exc

        model_class = self._model_class()
        load_kwargs: dict[str, Any] = {
            "revision": self.revision,
            "dtype": getattr(torch, plan.torch_dtype),
        }
        quantization = self._quantization_config(plan)
        if quantization is not None:
            load_kwargs["quantization_config"] = quantization
        # device_map comes from the resolved plan, so nothing pins cuda:0.
        load_kwargs["device_map"] = plan.device

        try:
            processor = AutoProcessor.from_pretrained(
                self.model_id, revision=self.revision
            )
        except Exception as exc:
            raise FindingsModelLoadError(
                f"could not load the processor for {self.model_id!r}."
            ) from exc

        try:
            model = model_class.from_pretrained(self.model_id, **load_kwargs)
        except Exception as exc:
            raise FindingsModelLoadError(
                f"could not load the Findings model {self.model_id!r} "
                f"(revision {self.revision!r}) onto {plan.device}."
            ) from exc

        _assert_has_image_processor(processor, self.model_id)
        _assert_has_vision_tower(model, self.model_id)

        # Inference only. The model is never put back into train mode anywhere
        # in this package.
        model.eval()

        return FindingsModelBundle(
            model=model,
            processor=processor,
            model_id=self.model_id,
            revision=self.revision,
            resolved_revision=str(
                getattr(getattr(model, "config", None), "_commit_hash", None)
                or self.revision
            ),
            device_plan=plan,
        )
