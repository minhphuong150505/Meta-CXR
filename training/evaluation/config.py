"""Validation for the ``evaluation:`` config block.

Config errors must surface **before** a run starts, not after it has trained for
hours and discovered that its ``selection_metric`` names a metric nothing
computes. Every field is checked against the registries in the modules that
implement it, so adding a metric in one place cannot leave the validator stale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from training.evaluation.bootstrap import (
    DEFAULT_CONFIDENCE,
    DEFAULT_SAMPLES,
    DEFAULT_SEED,
)
from training.evaluation.generation_metrics import LEXICAL_METRICS
from training.evaluation.threshold_calibration import (
    DEFAULT_OBJECTIVE,
    OBJECTIVES,
)
from training.evaluation.uncertain_policy import DEFAULT_POLICY, POLICIES

#: Aggregate metric names a Stage-1 checkpoint may be selected on.
CLASSIFICATION_METRICS = (
    "accuracy",
    "binary_accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "micro_precision",
    "micro_recall",
    "micro_f1",
    "weighted_precision",
    "weighted_recall",
    "weighted_f1",
    "positive_macro_precision",
    "positive_macro_recall",
    "positive_macro_f1",
    "positive_micro_precision",
    "positive_micro_recall",
    "positive_micro_f1",
    "macro_auroc",
    "macro_auprc",
    "micro_auroc",
    "micro_auprc",
    "macro_specificity",
    "macro_npv",
)

#: Stage-1 selection metrics, including the legacy key the runner already uses.
STAGE1_SELECTION_METRICS = (
    "f1_positive_macro",  # legacy key, still the runner default
    "f1_positive_macro_defined_only",
    "positive_macro_f1",
    "macro_auroc",
    "macro_auprc",
    "validation_loss",
    "loss",
)

#: Stage-2 selection metrics.
STAGE2_SELECTION_METRICS = (
    "validation_loss",
    "loss",
    "rouge_l",
    "bleu_4",
    "cider",
    "bertscore_f1",
    "radgraph_f1",
    "composite",
)

CLINICAL_METRICS = ("chexbert", "radgraph", "chexpert_labeler")

THRESHOLD_MODES = ("default", "calibrated")

#: Default composite weights. Documented rather than hard-coded silently:
#: clinical correctness is weighted above surface overlap because a report that
#: reads well and states the wrong finding is worse than an awkward correct one.
#: Override in config; the weights must sum to 1.
DEFAULT_COMPOSITE_WEIGHTS = {
    "radgraph_f1": 0.4,
    "chexbert_macro_f1": 0.2,
    "rouge_l": 0.2,
    "bertscore_f1": 0.2,
}


class EvaluationConfigError(ValueError):
    """The evaluation config block is invalid."""


@dataclass
class BootstrapConfig:
    enabled: bool = True
    samples: int = DEFAULT_SAMPLES
    confidence: float = DEFAULT_CONFIDENCE
    seed: int = DEFAULT_SEED

    def validate(self) -> None:
        if self.samples < 0:
            raise EvaluationConfigError(
                f"evaluation.bootstrap.samples must be >= 0, got {self.samples}"
            )
        if not 0 < self.confidence < 1:
            raise EvaluationConfigError(
                "evaluation.bootstrap.confidence must be in (0, 1), got "
                f"{self.confidence}"
            )


@dataclass
class EvaluationConfig:
    """The validated ``evaluation:`` block."""

    uncertain_policy: str = DEFAULT_POLICY
    selection_metric: str = "f1_positive_macro"
    threshold_mode: str = "default"
    threshold_objective: str = DEFAULT_OBJECTIVE
    include_meta_labels: bool = False
    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)
    classification_metrics: tuple[str, ...] = ()
    generation_metrics: tuple[str, ...] = ()
    clinical_metrics: tuple[str, ...] = ()
    composite_weights: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_COMPOSITE_WEIGHTS)
    )
    save_predictions: bool = True
    save_plots: bool = True
    save_per_sample_results: bool = True

    def validate(self, *, stage: str = "stage1") -> EvaluationConfig:
        if self.uncertain_policy not in POLICIES:
            raise EvaluationConfigError(
                f"evaluation.uncertain_policy={self.uncertain_policy!r} is not "
                f"one of {', '.join(POLICIES)}"
            )

        allowed = (
            STAGE1_SELECTION_METRICS if stage == "stage1" else STAGE2_SELECTION_METRICS
        )
        if self.selection_metric not in allowed:
            raise EvaluationConfigError(
                f"evaluation.selection_metric={self.selection_metric!r} cannot be "
                f"computed for {stage}. Valid options: {', '.join(allowed)}. "
                "Validating this before training starts is deliberate: a run must "
                "not train for hours and then find it has nothing to select on."
            )

        if self.threshold_mode not in THRESHOLD_MODES:
            raise EvaluationConfigError(
                f"evaluation.threshold_mode={self.threshold_mode!r} is not one of "
                f"{', '.join(THRESHOLD_MODES)}"
            )
        if self.threshold_objective not in OBJECTIVES:
            raise EvaluationConfigError(
                f"evaluation.threshold_objective={self.threshold_objective!r} is "
                f"not one of {', '.join(OBJECTIVES)}"
            )

        unknown = set(self.classification_metrics) - set(CLASSIFICATION_METRICS)
        if unknown:
            raise EvaluationConfigError(
                f"evaluation.classification_metrics contains unknown metric(s): "
                f"{', '.join(sorted(unknown))}. "
                f"Available: {', '.join(CLASSIFICATION_METRICS)}"
            )

        unknown = set(self.generation_metrics) - set(LEXICAL_METRICS)
        if unknown:
            raise EvaluationConfigError(
                f"evaluation.generation_metrics contains unknown metric(s): "
                f"{', '.join(sorted(unknown))}. "
                f"Available: {', '.join(LEXICAL_METRICS)}"
            )

        unknown = set(self.clinical_metrics) - set(CLINICAL_METRICS)
        if unknown:
            raise EvaluationConfigError(
                f"evaluation.clinical_metrics contains unknown metric(s): "
                f"{', '.join(sorted(unknown))}. "
                f"Available: {', '.join(CLINICAL_METRICS)}"
            )

        if self.selection_metric == "composite":
            total = sum(self.composite_weights.values())
            if abs(total - 1.0) > 1e-6:
                raise EvaluationConfigError(
                    f"evaluation.composite_weights must sum to 1, got {total:.6f} "
                    f"from {self.composite_weights}"
                )

        self.bootstrap.validate()
        return self

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> EvaluationConfig:
        payload = dict(payload or {})
        bootstrap_payload = payload.pop("bootstrap", {}) or {}
        known = {f for f in cls.__dataclass_fields__ if f != "bootstrap"}

        unknown = set(payload) - known
        if unknown:
            raise EvaluationConfigError(
                f"unknown key(s) in the evaluation config block: "
                f"{', '.join(sorted(unknown))}. Known keys: {', '.join(sorted(known))}"
            )

        for key in ("classification_metrics", "generation_metrics", "clinical_metrics"):
            if key in payload and payload[key] is not None:
                payload[key] = tuple(payload[key])

        return cls(bootstrap=BootstrapConfig(**bootstrap_payload), **payload)


def composite_score(
    metrics: dict[str, float], weights: dict[str, float] | None = None
) -> float:
    """Weighted sum of the configured metrics.

    Raises when a weighted metric is missing rather than treating it as 0: a
    composite silently computed over half its terms is not the score it claims
    to be.
    """
    weights = weights or DEFAULT_COMPOSITE_WEIGHTS
    missing = [name for name in weights if name not in metrics]
    if missing:
        raise EvaluationConfigError(
            f"composite score needs {', '.join(missing)}, which "
            f"{'is' if len(missing) == 1 else 'are'} not available. Either "
            "provide the metric or change evaluation.composite_weights; the "
            "missing terms are not treated as zero."
        )
    return sum(metrics[name] * weight for name, weight in weights.items())
