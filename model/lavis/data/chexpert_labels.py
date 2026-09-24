"""CheXpert label mapping for Stage-1 classification, kept free of the training stack.

Only numpy and pandas, so the CPU test suite can pin the exact mapping
ReportDataset applies without importing torchvision, transformers or LAVIS.

Class indices: 0 = negative, 1 = positive, 2 = uncertain, ``IGNORE_LABEL`` =
dropped from the loss per cell.

Blank-cell policy (``model.mhcac.blank_label_policy``)
------------------------------------------------------
A blank CheXpert cell means the labeler found no mention of the finding in the
report. Two readings are supported, and the choice changes what the
classification head is trained on:

``negative``
    Blank -> 0. This is what the original META-CXR paper does ("missing (NaN)
    values were treated as the negative class"; upstream uses ``fillna(0.0)``)
    and what the ``study_presence`` evaluation framing already assumes. Shipped
    in ``mimic_cxr_full.yaml`` as of 2026-09-24.

``ignore``
    Blank -> ``IGNORE_LABEL``: the cell is masked. This was the policy from
    2026-08-13 to 2026-09-24, and it is still what a config WITHOUT the key
    gets, so older configs and ablations reproduce exactly.

Under both policies, a study with NO CheXpert information -- no record at all,
or a record whose fourteen cells are all blank -- keeps ``IGNORE_LABEL`` on
every cell and stays out of ``classification_valid``. Filling those with
fourteen zeros would invent a fully-normal study from nothing.

Mention targets are taken from the raw export BEFORE the fill, and
``excluded_labels`` are applied AFTER it, under both policies.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

IGNORE_LABEL = -100

BLANK_AS_NEGATIVE = "negative"
BLANK_IGNORED = "ignore"
BLANK_LABEL_POLICIES = (BLANK_AS_NEGATIVE, BLANK_IGNORED)
# What a config without `model.mhcac.blank_label_policy` gets. Deliberately the
# historical policy, not the shipped one: an old config must not change meaning.
DEFAULT_BLANK_LABEL_POLICY = BLANK_IGNORED


def resolve_blank_label_policy(value) -> str:
    """Validate a configured policy; ``None`` (key absent) means the default."""
    if value is None:
        return DEFAULT_BLANK_LABEL_POLICY
    if value not in BLANK_LABEL_POLICIES:
        raise ValueError(
            f"unknown blank_label_policy {value!r}; expected one of "
            f"{', '.join(BLANK_LABEL_POLICIES)}"
        )
    return value


def mention_column(label: str) -> str:
    return f"_mention_{label}"


def map_chexpert_labels(
    frame: pd.DataFrame, cols: Sequence[str], blank_policy: str
) -> pd.DataFrame:
    """Map raw CheXpert values (1.0 / 0.0 / -1.0 / NaN) to int8 class indices.

    Rows with no non-blank cell at all keep ``IGNORE_LABEL`` everywhere, whatever
    the policy.
    """
    blank_policy = resolve_blank_label_policy(blank_policy)
    cols = list(cols)
    mapped = frame[cols].replace(-1, 2)
    if blank_policy == BLANK_AS_NEGATIVE:
        has_any = mapped.notna().any(axis=1).to_numpy()
        mapped = mapped.mask(mapped.isna().to_numpy() & has_any[:, None], 0)
    return mapped.fillna(IGNORE_LABEL).astype("int8")


def prepare_chexpert_labels(
    chexpert: pd.DataFrame,
    cols: Sequence[str],
    *,
    blank_policy: str,
    excluded_labels: Sequence[str] = (),
) -> pd.DataFrame:
    """Return a copy of ``chexpert`` carrying mapped labels and the row flags.

    Adds ``_has_chexpert_label_raw`` (the export held anything for this study),
    one ``_mention_<label>`` target per label (taken before the fill), and
    ``_has_usable_label`` (a kept column holds a class after the fill and the
    exclusion).
    """
    cols = list(cols)
    unknown = sorted(set(excluded_labels) - set(cols))
    if unknown:
        raise ValueError(
            f"excluded_labels names no such pathology: {unknown}; "
            f"expected a subset of {cols}"
        )
    out = chexpert.copy()
    out["_has_chexpert_label_raw"] = out[cols].notna().any(axis=1)
    for column in cols:
        out[mention_column(column)] = out[column].notna().astype("int8")
    out[cols] = map_chexpert_labels(out, cols, blank_policy)
    for column in excluded_labels:
        out[column] = np.int8(IGNORE_LABEL)
    kept = [c for c in cols if c not in set(excluded_labels)]
    out["_has_usable_label"] = (out[kept] >= 0).any(axis=1)
    return out


def attach_chexpert_labels(
    annotation: pd.DataFrame,
    prepared: pd.DataFrame,
    cols: Sequence[str],
    *,
    label_key: Sequence[str] = ("subject_id", "study_id"),
    processed_has_label=None,
) -> pd.DataFrame:
    """Left-join ``prepared`` onto ``annotation`` and derive the validity flags.

    ``processed_has_label`` is the split CSV's own ``has_chexpert_label`` (already
    coerced to bool) when it carries one. Adds ``classification_valid`` and
    ``mention_valid``; a row that matched no record gets ``IGNORE_LABEL`` on
    every label and zero on every mention target, and both flags False.
    """
    cols = list(cols)
    label_key = list(label_key)
    mention_cols = [mention_column(c) for c in cols]
    labels = prepared[
        label_key + cols + mention_cols + ["_has_chexpert_label_raw", "_has_usable_label"]
    ]
    merged = annotation.merge(
        labels,
        how="left",
        on=label_key,
        validate="many_to_one",
        indicator="_chexpert_merge",
    )
    raw_has_label = merged["_has_chexpert_label_raw"].eq(True)
    # Post-exclusion. The guard below still uses raw_has_label, so it keeps
    # catching a broken join rather than firing on an intentional exclusion.
    usable_label = merged["_has_usable_label"].eq(True)
    if processed_has_label is not None:
        processed = pd.Series(np.asarray(processed_has_label, dtype=bool), index=merged.index)
        inconsistent = processed & ~raw_has_label
        if inconsistent.any():
            raise ValueError(
                f"{int(inconsistent.sum())} processed rows claim CheXpert labels "
                "but no non-null source labels were found."
            )
        merged["classification_valid"] = processed & usable_label
    else:
        merged["classification_valid"] = usable_label
    # Rows that matched no CheXpert record land here with NaN across every
    # label. They are already excluded by classification_valid, but they must
    # not read back as negatives either -- under EITHER blank policy.
    merged[cols] = merged[cols].fillna(IGNORE_LABEL).astype("int8")
    # A study that matched no CheXpert record tells us nothing about what the
    # report mentioned, so it must not train the gate as fourteen zeros. That
    # is a different question from classification_valid, which asks whether
    # any usable P/N/U cell survived.
    merged["mention_valid"] = merged["_chexpert_merge"].eq("both")
    merged[mention_cols] = merged[mention_cols].fillna(0).astype("int8")
    return merged
