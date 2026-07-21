"""Counterfactual audit: does the report actually depend on the image?

A report generator conditioned on a strong language prior can score well on
BLEU/CIDEr while barely looking at the radiograph -- MIMIC-CXR findings text is
repetitive enough that "no acute cardiopulmonary process" is a good guess before
any pixel is read. NLG metrics cannot detect this. Perturbing the image and
measuring how much the report moves can.

The evaluator is model-agnostic: it talks to a ``ReportGenerationBackend``
protocol, so it runs against a real MedGemma on a GPU, or against a
deterministic fake on CPU in the test suite.

**On the meaning of the numbers.** ``text_change`` is a *lexical* measure -- a
token-level Jaccard distance. It is named ``lexical_text_change`` in the output
schema for that reason. It is NOT a clinical measure: two reports can be
lexically different and clinically identical, or lexically near-identical and
differ on the one word that matters ("no pneumothorax" vs "pneumothorax").
Clinical change requires a clinical backend (see
``training/evaluation/clinical.py``); when none is configured the corresponding
fields are ``None`` and the reason is recorded in ``notes``. They are never
filled in with the lexical number under a clinical-sounding name.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch

from training.evaluation import perturbations as pert

SCHEMA_VERSION = 1

#: Never appears in an audit record. Mirrors stage2_utils.SENSITIVE_EVAL_KEYS;
#: audit output is designed to be shareable, so it carries only ``sample_key``.
FORBIDDEN_RECORD_KEYS = frozenset(
    {"dicom_id", "study_id", "subject_id", "image_path", "ref", "reference", "report"}
)


@runtime_checkable
class ReportGenerationBackend(Protocol):
    """Anything that turns (image, prompt) pairs into report text.

    ``images`` may contain ``None`` for the ``no_image`` perturbation; a backend
    that cannot represent "no image" should raise rather than substitute a blank
    one, otherwise ``no_image`` and ``blank_image`` silently measure the same
    thing.
    """

    def generate(
        self, images: Sequence[torch.Tensor | None], prompts: Sequence[str]
    ) -> list[str]: ...


@runtime_checkable
class ClinicalChangeBackend(Protocol):
    """Optional adapter that compares two reports clinically, not lexically.

    Implementations wrap a real labeler (CheXbert, CheXpert) or a graph
    extractor (RadGraph). None ships in this repo -- see
    ``training/evaluation/clinical.py`` for how missing dependencies are
    reported.
    """

    name: str

    def compare(self, original: str, perturbed: str) -> dict[str, Any]: ...


@dataclass(frozen=True)
class AuditSample:
    """One study to audit. Carries no MIMIC identifiers by construction."""

    sample_key: str
    image: torch.Tensor
    prompt: str


@dataclass
class CounterfactualConfig:
    perturbations: tuple[str, ...] = pert.ALL_PERTURBATIONS
    seed: int = 16
    #: Below this mean lexical change the generator is flagged as leaning on the
    #: language prior. 0.10 is a reporting threshold, not a validated cutoff.
    language_prior_threshold: float = 0.10
    hard_negatives: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unknown = set(self.perturbations) - set(pert.ALL_PERTURBATIONS)
        if unknown:
            raise ValueError(f"unknown perturbation(s): {sorted(unknown)}")
        if not self.perturbations:
            raise ValueError("at least one perturbation is required")


def tokenize(text: str) -> list[str]:
    return [token for token in str(text).lower().split() if token]


def lexical_change(original: str, perturbed: str) -> float:
    """Token-level Jaccard distance in [0, 1]. 0 = identical bag of words.

    Deliberately bag-of-words: word order in radiology findings varies freely
    between equivalent reports, so an edit-distance measure would report change
    where none exists clinically.
    """
    left, right = set(tokenize(original)), set(tokenize(perturbed))
    if not left and not right:
        return 0.0
    union = left | right
    return round(1.0 - len(left & right) / len(union), 6)


@dataclass
class CounterfactualEvaluator:
    """Runs perturbations through a backend and scores the output drift."""

    backend: ReportGenerationBackend
    config: CounterfactualConfig = field(default_factory=CounterfactualConfig)
    clinical_backend: ClinicalChangeBackend | None = None

    def _donor_pool(self, samples: Sequence[AuditSample]) -> list[pert.Donor]:
        return [pert.Donor(sample_key=s.sample_key, image=s.image) for s in samples]

    def _donor_for(
        self, name: str, sample: AuditSample, pool: Sequence[pert.Donor], seed: int
    ) -> pert.Donor:
        if name == pert.HARD_NEGATIVE_SWAP:
            target = self.config.hard_negatives.get(sample.sample_key)
            if target is None:
                raise ValueError(
                    f"no hard negative configured for {sample.sample_key!r}; either "
                    "supply CounterfactualConfig.hard_negatives or drop "
                    f"{pert.HARD_NEGATIVE_SWAP!r} from the perturbation list"
                )
            for donor in pool:
                if donor.sample_key == target:
                    return donor
            raise ValueError(
                f"hard negative {target!r} for {sample.sample_key!r} is not in the cohort"
            )
        return pert.pick_random_donor(sample.sample_key, pool, seed)

    def _clinical_fields(self, original: str, perturbed: str) -> tuple[dict, list[str]]:
        if self.clinical_backend is None:
            return (
                {"clinical_change": None, "label_change": None},
                [
                    "no clinical backend configured; clinical_change and label_change "
                    "are unmeasured, not zero"
                ],
            )
        result = self.clinical_backend.compare(original, perturbed)
        return (
            {
                "clinical_change": result.get("clinical_change"),
                "label_change": result.get("label_change"),
                "clinical_backend": self.clinical_backend.name,
            },
            [],
        )

    def evaluate_sample(
        self, sample: AuditSample, pool: Sequence[pert.Donor]
    ) -> list[dict[str, Any]]:
        original = self.backend.generate([sample.image], [sample.prompt])[0]
        rows: list[dict[str, Any]] = []
        for offset, name in enumerate(self.config.perturbations):
            seed = self.config.seed + offset
            donor_key = None
            if name in pert.NEEDS_DONOR:
                donor = self._donor_for(name, sample, pool, seed)
                image = donor.image
                donor_key = donor.sample_key
            else:
                image = pert.apply_self_contained(name, sample.image, seed)

            perturbed = self.backend.generate([image], [sample.prompt])[0]
            clinical, notes = self._clinical_fields(original, perturbed)
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "sample_key": sample.sample_key,
                    "perturbation": name,
                    "donor_sample_key": donor_key,
                    "original_report": original,
                    "perturbed_report": perturbed,
                    "lexical_text_change": lexical_change(original, perturbed),
                    **clinical,
                    "notes": notes,
                }
            )
        return rows

    def evaluate(self, samples: Iterable[AuditSample]) -> dict[str, Any]:
        samples = list(samples)
        if not samples:
            raise ValueError("no samples to audit")
        pool = self._donor_pool(samples)

        rows: list[dict[str, Any]] = []
        for sample in samples:
            rows.extend(self.evaluate_sample(sample, pool))

        per_sample = {}
        for sample in samples:
            changes = [
                row["lexical_text_change"]
                for row in rows
                if row["sample_key"] == sample.sample_key
            ]
            per_sample[sample.sample_key] = round(sum(changes) / len(changes), 6)

        overall = round(sum(per_sample.values()) / len(per_sample), 6)
        by_perturbation = {}
        for name in self.config.perturbations:
            values = [row["lexical_text_change"] for row in rows if row["perturbation"] == name]
            by_perturbation[name] = round(sum(values) / len(values), 6)

        return {
            "schema_version": SCHEMA_VERSION,
            "records": rows,
            "visual_reliance_score": per_sample,
            "mean_visual_reliance_score": overall,
            "mean_change_by_perturbation": by_perturbation,
            "language_prior_dependent": overall < self.config.language_prior_threshold,
            "language_prior_threshold": self.config.language_prior_threshold,
            "score_definition": (
                "mean token-level Jaccard distance between the original report and "
                "each perturbed report. Lexical only -- not a clinical measure."
            ),
            "clinical_backend": (
                self.clinical_backend.name if self.clinical_backend else None
            ),
        }


def privacy_violations(payload: Any, path: str = "") -> list[str]:
    """Recursively find forbidden identifier keys anywhere in an audit artifact.

    Top-level-only checking is not enough: audit output nests per-perturbation
    records inside lists inside dicts, so an identifier three levels down would
    pass a shallow check and then be written to a shared artifact.
    """
    found: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            here = f"{path}.{key}" if path else str(key)
            if key in FORBIDDEN_RECORD_KEYS:
                found.append(here)
            found.extend(privacy_violations(value, here))
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            found.extend(privacy_violations(value, f"{path}[{index}]"))
    return found


def assert_shareable(payload: Any) -> None:
    """Raise unless the artifact is free of MIMIC identifiers at every depth."""
    violations = privacy_violations(payload)
    if violations:
        raise ValueError(
            "refusing to write a counterfactual artifact containing MIMIC "
            f"identifiers at: {', '.join(sorted(violations))}"
        )
