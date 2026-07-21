"""Phase-1 prediction record and generation postprocessing.

The record deliberately carries provenance flags (``external_checkpoint``,
``fine_tuned_by_this_project``) so a results file can never be mistaken for the
output of a model this project trained. It carries no MIMIC identifiers and no
reference report: ``sample_key`` is the salted digest built by
``training.dataio.manifest.build_records``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from training.dataio.manifest import split_generated_report

PIPELINE_MODE = "pretrained_medgemma_findings_first"

#: The model was asked for FINDINGS only but emitted an IMPRESSION section too.
#: The impression is dropped rather than kept, so Phase 1 never reports an
#: impression score sourced from a model that was not evaluated for it.
WARN_UNEXPECTED_IMPRESSION = "unexpected_impression_generated"
#: Generation came back blank after stripping the prompt echo and headers.
WARN_EMPTY_FINDINGS = "empty_findings"


def postprocess_findings(raw: str) -> tuple[str, list[str]]:
    """Normalise one raw generation into (findings, warnings).

    Returns only the FINDINGS body. A model-emitted IMPRESSION is discarded and
    recorded as a warning -- keeping it would smuggle un-audited Phase-2 content
    into a Phase-1 results file.
    """
    findings, impression = split_generated_report(raw or "")
    warnings: list[str] = []
    if impression.strip():
        warnings.append(WARN_UNEXPECTED_IMPRESSION)
    if not findings.strip():
        warnings.append(WARN_EMPTY_FINDINGS)
    return findings.strip(), warnings


@dataclass
class FindingsPrediction:
    """One Phase-1 prediction, ready to be written as a JSONL line."""

    sample_key: str
    findings: str
    model_id: str
    model_revision: str
    elapsed_seconds: float = 0.0
    estimated_cost_usd: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_key": self.sample_key,
            "findings": self.findings,
            # Phase 1 generates neither of these. They are present and null so
            # downstream readers can rely on a stable key set across phases.
            "impression": None,
            "full_report": None,
            "pipeline_mode": PIPELINE_MODE,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "impression_enabled": False,
            "fine_tuned_by_this_project": False,
            "external_checkpoint": True,
            "warnings": list(self.warnings),
            "runtime": {
                "elapsed_seconds": round(float(self.elapsed_seconds), 4),
                "estimated_cost_usd": round(float(self.estimated_cost_usd), 6),
            },
        }
