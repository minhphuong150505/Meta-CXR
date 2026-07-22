"""Synthetic Stage-2 records so the prompt scripts run with no data or GPU.

These are NOT MIMIC-CXR. They carry the same record shape the real pipeline emits
(``pred_groups``, views, prior flags, a ``ref`` target) so scripts are exercisable
end to end. Any numbers produced from them are illustrative, never model results.
"""

from __future__ import annotations

import random
from typing import Any

_FINDINGS = [
    "Cardiomegaly",
    "Lung Opacity",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Support Devices",
]
_VIEWS = [("PA", ()), ("AP", ()), ("PA", ("lateral",)), ("AP", ("lateral",))]
_NORMAL_REF = "The lungs are clear without focal consolidation, effusion or pneumothorax. The cardiomediastinal silhouette is normal."
_TEMPORAL_REF = "Stable cardiomegaly. The pleural effusion is unchanged compared to the prior study. No new consolidation."
_POSITIVE_REF = "There is a moderate right pleural effusion with adjacent atelectasis. Mild cardiomegaly is present."


def synthetic_records(n: int = 50, seed: int = 16) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    for i in range(n):
        roll = rng.random()
        pool = rng.sample(_FINDINGS, k=rng.randint(0, 4))
        positive, uncertain, negative = [], [], []
        for name in _FINDINGS:
            if name in pool:
                (positive if rng.random() < 0.6 else uncertain).append(name)
            else:
                negative.append(name)
        anchor, aux = rng.choice(_VIEWS)
        prior = rng.random() < 0.4
        if roll < 0.35:  # normal-ish study
            positive, uncertain = [], []
            ref = _NORMAL_REF
        elif prior:
            ref = _TEMPORAL_REF
        else:
            ref = _POSITIVE_REF
        records.append(
            {
                "study_id": f"synthetic-{i:04d}",
                "sample_key": f"synthetic-{i:04d}",
                "pred_groups": {
                    "positive": positive,
                    "uncertain": uncertain,
                    "negative": negative,
                },
                "negative_probabilities": {name: round(rng.random(), 3) for name in negative},
                "uncertain_probabilities": {name: round(0.5 + rng.random() / 2, 3) for name in uncertain},
                "anchor_view": anchor,
                "auxiliary_views": list(aux),
                "prior_available": prior,
                "comparison_available": prior,
                "indication": rng.choice([None, "Shortness of breath", "Chest pain", "Follow-up"]),
                "technique": rng.choice([None, "PA and lateral chest radiograph"]),
                "qformer_token_count": 32,
                "ref": ref,
            }
        )
    return records
