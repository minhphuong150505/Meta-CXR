"""External, pre-fine-tuned MedGemma checkpoints used in inference only.

These checkpoints were fine-tuned by a third party (erjui) from
``google/medgemma-4b-it`` on the csrrg_ift dataset, which is derived from
MIMIC-CXR and CheXpert+. This project does not fine-tune them and must not
describe them as its own models, nor as having been trained on this
repository's splits.

Phase 1 loads the Findings checkpoint only. The Impression checkpoint is
declared but disabled; see ``impression_reporter``.
"""

from model.pretrained_medgemma.errors import (
    FindingsModelLoadError,
    ImpressionPhaseDisabledError,
    NotMultimodalError,
    PretrainedMedGemmaError,
)
from model.pretrained_medgemma.findings_loader import (
    FINDINGS_MODEL_ID,
    FindingsModelBundle,
    PretrainedFindingsLoader,
)
from model.pretrained_medgemma.findings_reporter import (
    GenerationSettings,
    PretrainedFindingsReporter,
)
from model.pretrained_medgemma.output_schema import (
    PIPELINE_MODE,
    FindingsPrediction,
    postprocess_findings,
)

__all__ = [
    "FINDINGS_MODEL_ID",
    "PIPELINE_MODE",
    "FindingsModelBundle",
    "FindingsModelLoadError",
    "FindingsPrediction",
    "GenerationSettings",
    "ImpressionPhaseDisabledError",
    "NotMultimodalError",
    "PretrainedFindingsLoader",
    "PretrainedFindingsReporter",
    "PretrainedMedGemmaError",
    "postprocess_findings",
]
