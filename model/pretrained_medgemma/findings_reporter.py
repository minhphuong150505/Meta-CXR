"""FINDINGS generation against the external MedGemma checkpoint.

The prompt asks for FINDINGS only. If the model volunteers an IMPRESSION
anyway, the postprocessor drops it and records a warning -- Phase 1 never keeps
impression text, because no impression evaluation has been budgeted or run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from model.pretrained_medgemma.findings_loader import FindingsModelBundle
from model.pretrained_medgemma.output_schema import postprocess_findings

SYSTEM_PROMPT = "You are an expert radiologist."

FINDINGS_INSTRUCTION = (
    "Analyze the chest X-ray image and write the FINDINGS section of a "
    "structured radiology report. Describe only observations supported by the "
    "image. Include relevant observations for lungs, pleura, cardiomediastinal "
    "silhouette, bones, and support devices. Do not write an IMPRESSION section."
)


@dataclass(frozen=True)
class GenerationSettings:
    """Deterministic by default: a cost estimate must be reproducible."""

    max_new_tokens: int = 512
    do_sample: bool = False
    num_beams: int = 1

    def to_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "max_new_tokens": int(self.max_new_tokens),
            "do_sample": bool(self.do_sample),
            "num_beams": int(self.num_beams),
        }
        return kwargs


@dataclass(frozen=True)
class FindingsGeneration:
    """One generation with its normalised text and timing."""

    findings: str
    warnings: list[str]
    elapsed_seconds: float


class PretrainedFindingsReporter:
    """Generates FINDINGS for one image at a time.

    Holds a reference to an already-loaded bundle; it never loads a model
    itself, and it has no reference to the Impression checkpoint.
    """

    def __init__(
        self,
        bundle: FindingsModelBundle,
        settings: GenerationSettings | None = None,
    ) -> None:
        self.bundle = bundle
        self.settings = settings or GenerationSettings()

    def build_messages(self, image: Any) -> list[dict[str, Any]]:
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": FINDINGS_INSTRUCTION},
                ],
            },
        ]

    def generate(self, image: Any) -> FindingsGeneration:
        """Generate FINDINGS for a single chest X-ray."""
        import torch

        processor = self.bundle.processor
        model = self.bundle.model
        started = time.monotonic()

        inputs = processor.apply_chat_template(
            self.build_messages(image),
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        prompt_length = int(inputs["input_ids"].shape[-1])

        # inference_mode, not no_grad: no autograd bookkeeping is needed
        # anywhere in this pipeline.
        with torch.inference_mode():
            outputs = model.generate(**inputs, **self.settings.to_kwargs())

        # Decode only the newly generated tokens. Slicing off the prompt is
        # exact, unlike string-matching the prompt back out of a full decode.
        new_tokens = outputs[0][prompt_length:]
        raw = processor.decode(new_tokens, skip_special_tokens=True)

        findings, warnings = postprocess_findings(raw)
        return FindingsGeneration(
            findings=findings,
            warnings=warnings,
            elapsed_seconds=time.monotonic() - started,
        )
