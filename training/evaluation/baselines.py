"""Trivial three-class baselines, scored with the same paper protocol.

The paper's weighted P/R/F1 average the three classes by their support, so on
MIMIC-CXR -- where most cells are Negative under the blank-as-negative policy --
a model that always answers "Negative" already scores high. A metric table
without these rows cannot tell learning from class imbalance.

Every baseline is a probability array fed through
:func:`training.evaluation.classification_metrics.evaluate_classification`,
so it is scored by exactly the code that scores the model.

* ``all_negative`` / ``all_positive`` / ``all_uncertain`` -- one class always.
* ``majority_class`` -- each finding's most frequent class in this split.
* ``prior_random`` -- each finding's class drawn from its class frequencies.

⚠ ``majority_class`` and ``prior_random`` read the evaluated split's own class
frequencies, so they are an optimistic floor, not a deployable model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from training.evaluation.classification_metrics import (
    NUM_CLASSES,
    evaluate_classification,
)
from training.evaluation.schemas import MISSING, ClassificationPredictions

logger = logging.getLogger(__name__)

BASELINES = ("all_negative", "all_positive", "all_uncertain", "majority_class", "prior_random")

#: Columns of the baseline table: the paper's headline numbers.
BASELINE_COLUMNS = ("weighted_precision", "weighted_recall", "weighted_f1", "mean_weighted_f1_5")


@dataclass
class BaselineRow:
    name: str
    metrics: dict[str, float]
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {"baseline": self.name, "description": self.description, **self.metrics}


def _one_hot(classes: np.ndarray) -> np.ndarray:
    probabilities = np.zeros(classes.shape + (NUM_CLASSES,), dtype=np.float64)
    np.put_along_axis(probabilities, classes[..., None], 1.0, axis=-1)
    return probabilities


def _class_frequencies(labels: np.ndarray) -> np.ndarray:
    """``[P, 3]`` class frequencies per finding, over non-missing cells."""
    freqs = np.zeros((labels.shape[1], NUM_CLASSES), dtype=np.float64)
    for p in range(labels.shape[1]):
        column = labels[:, p]
        column = column[column != MISSING]
        if column.size:
            freqs[p] = np.bincount(column.astype(int), minlength=NUM_CLASSES) / column.size
        else:
            freqs[p, 0] = 1.0
    return freqs


def compute_baselines(
    predictions: ClassificationPredictions, *, seed: int = 42
) -> list[BaselineRow]:
    labels = predictions.labels
    n, p = labels.shape
    freqs = _class_frequencies(labels)
    rng = np.random.default_rng(seed)

    candidates = {
        "all_negative": (np.zeros((n, p), dtype=int), "Negative for every finding"),
        "all_positive": (np.ones((n, p), dtype=int), "Positive for every finding"),
        "all_uncertain": (np.full((n, p), 2, dtype=int), "Uncertain for every finding"),
        "majority_class": (
            np.broadcast_to(freqs.argmax(axis=1), (n, p)).copy(),
            "each finding's most frequent class in this split",
        ),
        "prior_random": (
            np.stack(
                [rng.choice(NUM_CLASSES, size=n, p=freqs[j]) for j in range(p)], axis=1
            ),
            f"class drawn from this split's frequencies (seed {seed})",
        ),
    }
    rows = []
    for name, (classes, description) in candidates.items():
        fake = ClassificationPredictions(
            labels=labels,
            probabilities=_one_hot(classes),
            pathology_names=predictions.pathology_names,
            sample_keys=predictions.sample_keys,
        )
        aggregates = evaluate_classification(fake).aggregates
        rows.append(
            BaselineRow(
                name=name,
                metrics={k: aggregates[k] for k in BASELINE_COLUMNS},
                description=description,
            )
        )
    return rows


def baseline_table(rows: list[BaselineRow], model_row: BaselineRow) -> str:
    def fmt(value: float) -> str:
        return "n/a" if value is None or np.isnan(value) else f"{value:.4f}"

    header = (
        "| | " + " | ".join(BASELINE_COLUMNS) + " |\n"
        "| --- |" + " ---: |" * len(BASELINE_COLUMNS) + "\n"
    )
    body = "".join(
        f"| {'**' + row.name + '**' if row is model_row else row.name} | "
        + " | ".join(fmt(row.metrics[c]) for c in BASELINE_COLUMNS)
        + " |\n"
        for row in [model_row, *rows]
    )
    return header + body
