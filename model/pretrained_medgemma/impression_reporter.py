"""Impression phase -- DISABLED for the Findings-first budget phase.

This module is intentionally inert. Importing it must not download a
checkpoint, construct a processor, allocate VRAM, or import transformers. It
exists so the Phase-2 interface is settled and reviewable, not so it can run.

Phase 2 requires, in order:
  1. Findings inference complete on the controlled split.
  2. Findings metrics and the safety/counterfactual audit reviewed.
  3. Real measured runtime cost from the pilot.
  4. Explicit user approval of the additional spend.

There is no flag that bypasses the guard below. Enabling Impression is a config
change plus an explicit command, in a separate session.
"""

from __future__ import annotations

from typing import Any

from model.pretrained_medgemma.errors import ImpressionPhaseDisabledError

IMPRESSION_MODEL_ID = "erjui/medgemma-4b-srrg-impression"


def assert_impression_disabled(
    *, model_enabled: bool, run_impression: bool
) -> None:
    """Hard stop when anything asks for Impression during Phase 1.

    Called by the runner before a model is loaded, so a misconfigured run costs
    nothing rather than paying to download a second 4B checkpoint first.
    """
    if model_enabled or run_impression:
        raise ImpressionPhaseDisabledError(
            "Impression generation is disabled for the Findings-first budget "
            "phase. Use a separate explicit Phase-2 command after budget "
            "approval."
        )


class PretrainedImpressionReporter:
    """Phase-2 placeholder. Constructing it is an error.

    The signature records the intended interface (image + the findings text
    produced by Phase 1) so the Phase-2 design is reviewable now, without any
    of it being runnable.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise ImpressionPhaseDisabledError(
            "PretrainedImpressionReporter cannot be constructed during the "
            "Findings-first phase. Impression is a second GPU budget that has "
            "not been approved."
        )

    def generate(self, *, image: Any, findings: str) -> str:
        raise ImpressionPhaseDisabledError("Impression generation is disabled.")
