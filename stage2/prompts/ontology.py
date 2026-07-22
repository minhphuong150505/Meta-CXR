"""Canonical abnormality ontology for Stage-2 prompt construction.

Source of truth is ``ABNORMALITIES_14`` in
``training/train_eval_figure9_llm_variants_200.py`` (No Finding + 13 findings).
That module imports torch/transformers and cannot be loaded on a CPU box, so the
names are mirrored here and kept torch-free. ``tests/test_stage2_prompts.py``
pins this list so a future ontology change to the classifier fails loudly here.

Names must match the classifier exactly: the prompt is the only place these
strings surface to the LLM, and a mismatch would silently drop a finding from a
policy (e.g. ``critical_only``) rather than crash.
"""

from __future__ import annotations

NO_FINDING = "No Finding"

#: The 13 modeled findings, in classifier order, excluding ``No Finding``.
MODELED_FINDINGS: tuple[str, ...] = (
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
)

MODELED_FINDINGS_SET = frozenset(MODELED_FINDINGS)

#: Findings whose absence is clinically worth stating even on a normal study.
#: Overridable via ``PromptConfig.critical_findings``. Every name here must be a
#: modeled finding; ``validate_critical_findings`` enforces that.
DEFAULT_CRITICAL_FINDINGS: tuple[str, ...] = (
    "Pneumothorax",
    "Pleural Effusion",
    "Consolidation",
    "Edema",
)


def validate_critical_findings(names: tuple[str, ...]) -> tuple[str, ...]:
    """Drop names that are not modeled findings, preserving order.

    A critical-negative list that references an un-modeled pathology must not
    crash the builder; it simply cannot be selected, so it is filtered out.
    """
    return tuple(name for name in names if name in MODELED_FINDINGS_SET)
