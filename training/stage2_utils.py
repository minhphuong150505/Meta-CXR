"""Small, dependency-free helpers shared by the Stage-2 training scripts.

Keep this module importable without PyTorch/Transformers so its correctness and
privacy invariants can be checked on CPU-only development machines.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 3
SENSITIVE_EVAL_KEYS = frozenset(
    {"dicom_id", "study_id", "subject_id", "ref", "reference", "report", "image_path"}
)


def stable_fingerprint(payload: Mapping[str, Any], length: int = 16) -> str:
    """Return a deterministic cache/cohort identifier for JSON-compatible data."""
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:length]


def file_identity(path: str | Path) -> dict[str, Any]:
    """Cheap identity for large checkpoints without hashing multi-GB files."""
    file_path = Path(path)
    if not file_path.exists():
        return {"path_name": file_path.name, "missing": True}
    stat = file_path.stat()
    return {
        "path_name": file_path.name,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def select_threshold_class(
    probabilities: Sequence[float],
    thresholds: Mapping[str, float] | None,
    class_names: Sequence[str] = ("negative", "positive", "uncertain"),
) -> str:
    """Apply calibrated thresholds and always fall back to probability argmax.

    Older Stage-2 code silently omitted an abnormality when no class cleared its
    threshold. That reduced structured-prompt recall and made its behaviour
    differ from inference. The fallback keeps every abnormality classified.
    """
    if len(probabilities) != len(class_names):
        raise ValueError("probabilities and class_names must have equal length")
    if not probabilities:
        raise ValueError("probabilities must not be empty")
    thresholds = thresholds or {}
    passing = [
        (float(probabilities[i]), name)
        for i, name in enumerate(class_names)
        if float(probabilities[i]) >= float(thresholds.get(name, 0.5))
    ]
    if passing:
        return max(passing, key=lambda item: item[0])[1]
    return class_names[max(range(len(probabilities)), key=lambda i: float(probabilities[i]))]


def safe_prediction_row(
    *, sample_key: str, index: int, prediction: str, generation_ok: bool
) -> dict[str, Any]:
    """Build the default resumable evaluation row without MIMIC identifiers/text."""
    return {
        "sample_key": str(sample_key),
        "index": int(index),
        "pred": str(prediction),
        "generation_ok": bool(generation_ok),
    }


def contains_sensitive_eval_fields(row: Mapping[str, Any]) -> bool:
    return bool(SENSITIVE_EVAL_KEYS.intersection(row))


def adapter_is_complete(path: str | Path, image_mode: str) -> bool:
    """Validate a resumable/evaluable Stage-2 artifact, not one sentinel file."""
    adapter_dir = Path(path)
    adapter_weights = (
        adapter_dir / "adapter_model.safetensors",
        adapter_dir / "adapter_model.bin",
    )
    required = [
        adapter_dir / "adapter_config.json",
        adapter_dir / "meta.json",
        adapter_dir / "manifest.json",
        adapter_dir / "trainer_state.pt",
    ]
    if image_mode == "qformer":
        required.append(adapter_dir / "img_proj.pt")
    if not all(item.is_file() for item in required) or not any(
        item.is_file() for item in adapter_weights
    ):
        return False
    try:
        manifest = json.loads((adapter_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    return (
        manifest.get("schema_version") == SCHEMA_VERSION
        and manifest.get("status") == "complete"
        and manifest.get("image_mode") == image_mode
    )


def masked_label_ids(
    full_ids: Sequence[int], prompt_ids: Sequence[int], pad_token_id: int | None = None
) -> list[int]:
    """Mask the exact chat-template prompt prefix and optional padding tokens."""
    prefix = 0
    limit = min(len(full_ids), len(prompt_ids))
    while prefix < limit and int(full_ids[prefix]) == int(prompt_ids[prefix]):
        prefix += 1
    # A mismatch before the prompt ends is almost always a BOS/template bug. Do
    # not train on user/image tokens silently.
    if prefix != len(prompt_ids):
        raise ValueError(
            f"full conversation does not start with prompt tokens ({prefix}/{len(prompt_ids)})"
        )
    labels = [int(token_id) for token_id in full_ids]
    labels[:prefix] = [-100] * prefix
    if pad_token_id is not None:
        labels = [
            -100 if token_id == int(pad_token_id) else label
            for token_id, label in zip(full_ids, labels)
        ]
    return labels


def validate_soft_token_batch(
    num_embeddings: int, batch_size: int, tokens_per_row: int, expected_tokens: int
) -> None:
    """Guard soft-token injection against silent cross-sample contamination.

    The previous implementation indexed projected embeddings with
    ``min(batch_idx, n - 1)``. When the embedding count did not match the batch
    it clamped instead of failing, feeding one study's image features to a
    different study's report with no error.
    """
    if num_embeddings != batch_size:
        raise ValueError(
            "projected image embeddings do not match the batch: "
            f"{num_embeddings} embeddings for {batch_size} sequences"
        )
    if tokens_per_row != expected_tokens:
        raise ValueError(
            f"expected {expected_tokens} image tokens, got {tokens_per_row}"
        )


def accumulation_window_size(batch_index: int, total_batches: int, grad_accum: int) -> int:
    """Actual micro-batch count in the current accumulation window.

    Dividing the final partial window by this value avoids under-scaling its
    gradients when ``total_batches % grad_accum != 0``.
    """
    if total_batches <= 0 or grad_accum <= 0 or not 0 <= batch_index < total_batches:
        raise ValueError("invalid accumulation arguments")
    window_start = (batch_index // grad_accum) * grad_accum
    return min(grad_accum, total_batches - window_start)


def language_lora_target_names(module_names: Sequence[str]) -> list[str]:
    """Select only language-transformer projections from a multimodal model.

    Selection intentionally uses module names rather than ``nn.Linear`` type
    checks. Quantized QLoRA replaces those layers with bitsandbytes
    ``Linear4bit`` modules, so a type check made the previous implementation
    find no targets at all.
    """
    suffixes = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
    language_markers = ("language_model", "text_model")
    return sorted(
        {
            name
            for name in module_names
            if name.rsplit(".", 1)[-1] in suffixes
            and any(marker in name for marker in language_markers)
        }
    )


def section_omission_rate(
    predictions: Sequence[str], references: Sequence[str]
) -> float:
    """Fraction of rows where the reference has a section but the prediction does not.

    Corpus NLG scores hide omission: a model that writes a good FINDINGS and no
    IMPRESSION is barely penalised by BLEU over the joined report, because the
    reference IMPRESSION is short. Reporting the omission rate separately makes
    that failure visible.

    Rows whose reference section is itself empty are excluded from the
    denominator -- there is nothing to omit -- so an all-empty reference cohort
    reports 0.0 rather than a spurious 1.0.
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions and references differ in length: "
            f"{len(predictions)} vs {len(references)}"
        )
    expected = 0
    omitted = 0
    for prediction, reference in zip(predictions, references):
        if not str(reference).strip():
            continue
        expected += 1
        if not str(prediction).strip():
            omitted += 1
    if expected == 0:
        return 0.0
    return omitted / expected


def prefix_metric_keys(metrics: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    """Namespace one section's metrics so section blocks cannot collide."""
    return {f"{prefix}/{key}": value for key, value in metrics.items()}


def native_findings_instruction() -> str:
    """Image-only instruction for the native MedGemma ablation.

    Keeping this function independent of Stage-1 predictions prevents Q-Former
    or MHCAC information from leaking into the native-image baseline prompt.
    """
    return (
        "Act as an expert radiologist. Based only on the provided chest "
        "radiograph, write only the Findings section as one concise clinical "
        "paragraph. Describe visible observations precisely, use cautious "
        "language when appropriate, and do not invent facts, add an Impression "
        "section, use bullet points, or discuss the task itself."
    )


def private_bucket_violations(
    bucket_metadata: Mapping[str, Any], iam_policy: Mapping[str, Any]
) -> list[str]:
    """Return reasons a GCS bucket cannot be proven private.

    Upload is allowed only when Public Access Prevention is explicitly enforced
    and the IAM policy has no public principals. Missing metadata is treated as
    unsafe instead of silently assuming a bucket is private.
    """

    def nested(mapping: Mapping[str, Any], *names: str) -> Any:
        for name in names:
            if name in mapping:
                return mapping[name]
        return None

    iam_config = nested(bucket_metadata, "iamConfiguration", "iam_configuration")
    if not isinstance(iam_config, Mapping):
        iam_config = {}
    prevention = nested(
        iam_config,
        "publicAccessPrevention",
        "public_access_prevention",
    )
    if prevention is None:
        prevention = nested(
            bucket_metadata,
            "publicAccessPrevention",
            "public_access_prevention",
        )

    violations = []
    if str(prevention).strip().lower() != "enforced":
        violations.append("Public Access Prevention is not enforced")

    public_members = {"allUsers", "allAuthenticatedUsers"}
    exposed = sorted(
        {
            str(member)
            for binding in iam_policy.get("bindings", [])
            if isinstance(binding, Mapping)
            for member in binding.get("members", [])
            if str(member) in public_members
        }
    )
    if exposed:
        violations.append(f"public IAM principals present: {', '.join(exposed)}")
    return violations
