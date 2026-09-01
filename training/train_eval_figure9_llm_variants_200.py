#!/usr/bin/env python3
"""Train/evaluate Figure 9 LLM variants on 200 META-CXR samples.

This script builds separate three-line Figure 9 outputs for Vicuna and
MedGemma:

* Base LLM: base model, no LoRA.
* Fine-Tuned LLM: new LoRA trained on 200 train samples.
* Instruction-Tuned LLM: existing domain/instruction LoRA metrics already saved
  in GCS.

The fine-tuned training and all missing evaluations use the 07_all_three stage-1
checkpoint and Q-Former embeddings injected through soft image tokens.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("DISABLE_TORCH_COMPILE", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import transformers
from bert_score import score as bert_score_fn
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from nltk.translate.meteor_score import meteor_score
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)

try:
    from stage2_utils import (
        SCHEMA_VERSION,
        accumulation_window_size,
        adapter_is_complete,
        file_identity,
        language_lora_target_names,
        masked_label_ids,
        native_findings_instruction,
        prefix_metric_keys,
        private_bucket_violations,
        safe_prediction_row,
        section_omission_rate,
        select_threshold_class,
        stable_fingerprint,
    )
except ImportError:  # ``python -m training...``
    from training.stage2_utils import (
        SCHEMA_VERSION,
        accumulation_window_size,
        adapter_is_complete,
        file_identity,
        language_lora_target_names,
        masked_label_ids,
        native_findings_instruction,
        prefix_metric_keys,
        private_bucket_violations,
        safe_prediction_row,
        section_omission_rate,
        select_threshold_class,
        stable_fingerprint,
    )

try:
    from medgemma.capabilities import (
        MultimodalModelLoadError,
        validate_multimodal_capability,
    )
    from medgemma.soft_tokens import SoftTokenEmbeddingWrapper, soft_token_bad_words_ids
    from run_context import Stage1Context
    from torch_io import load_torch_checkpoint
except ImportError:  # ``python -m training...``
    from training.medgemma.capabilities import (
        MultimodalModelLoadError,
        validate_multimodal_capability,
    )
    from training.medgemma.soft_tokens import (
        SoftTokenEmbeddingWrapper,
        soft_token_bad_words_ids,
    )
    from training.run_context import Stage1Context
    from training.torch_io import load_torch_checkpoint

try:
    from dataio.manifest import (
        FINDINGS_AND_IMPRESSION,
        FINDINGS_ONLY,
        split_generated_report,
    )
except ImportError:  # ``python -m training...``
    from training.dataio.manifest import (
        FINDINGS_AND_IMPRESSION,
        FINDINGS_ONLY,
        split_generated_report,
    )

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

# NOTE: LAVIS, the vision encoders, the Q-Former and MHCAC are imported ONLY by
# training/stage1/lavis_loader.py, and only from inside the Stage-1 branch of
# build_stage1_records(). Do not add a module-scope Stage-1 import here: it
# would make `medgemma_direct` -- which by definition uses none of that -- fail
# to start on a machine that has no Stage-1 stack installed.
from local_config import (  # noqa: E402
    PROCESSED_TEST_CSV,
    PROCESSED_TRAIN_CSV,
    PROCESSED_VAL_CSV,
    VIS_ROOT,
)

# Torch-free shared prompt builder. Optional: when a VariantLLM is constructed
# with prompt_config=None the legacy build_prompt/build_native_instruction path is
# used unchanged, so existing runs and checkpoints are unaffected.
from stage2.prompts import (  # noqa: E402
    PromptBuilder,
    PromptConfig,
    context_from_record,
)
from stage2.prompts.templates import template_hash as _prompt_template_hash  # noqa: E402

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)

SEED = 16
DEFAULT_RUN_NAME = "07_all_three"
NUM_IMG_TOKENS = 32
BERTSCORE_MODEL = "microsoft/deberta-xlarge-mnli"
SMOOTH = SmoothingFunction().method1

VICUNA_MODEL_ID = "lmsys/vicuna-7b-v1.3"
#: ``image_mode`` values in which MedGemma's OWN vision tower receives pixels.
#: ``native_qformer`` appears in BOTH of these sets on purpose -- it is the
#: originally-designed architecture, where the 32 soft tokens SUPPLEMENT the
#: image rather than replace it. Branching on these sets rather than on
#: ``== "native"`` / ``== "qformer"`` is what keeps the combined mode from being
#: silently half-configured: every site that used to test one string now tests
#: the capability it actually cares about.
NATIVE_PIXEL_MODES = frozenset({"native", "native_qformer"})
#: ``image_mode`` values in which projected Q-Former vectors are substituted at
#: ``<qformer_soft_token>`` positions.
SOFT_TOKEN_MODES = frozenset({"qformer", "native_qformer"})
ALL_IMAGE_MODES = frozenset({"qformer", "native", "native_qformer", "text_only"})

MEDGEMMA_MODEL_ID = "google/medgemma-1.5-4b-it"
VICUNA_EXISTING_LORA = PROJECT_DIR / "checkpoints/lora-vicuna-7b-report-20250621"
MEDGEMMA_EXISTING_LORA = "DeepRadiology/medgemma1.5-CXR"

INSTRUCTION_TABLES = {
    "vicuna": "gs://meta-cxr-checkpoint/eval/nlg_metrics_table.csv",
    "medgemma": "gs://meta-cxr-checkpoint/eval/MedGemma_QFormer/metrics/nlg_metrics_table_medgemma_qformer.csv",
}

ABNORMALITIES_14 = [
    "No Finding",
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
]
CLASS_MAP = {"negative": 0, "positive": 1, "uncertain": 2}

# Image-only argmax is the safe default. Historical ``threshold.json`` values
# have no Stage-1 checkpoint/validation provenance and must never be loaded
# implicitly. A caller may opt in to a separately calibrated file.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["vicuna", "medgemma"], choices=["vicuna", "medgemma"])
    parser.add_argument("--checkpoint-root", default="checkpoints")
    parser.add_argument("--output-dir", default="output/figure9_llm_variants_200")
    parser.add_argument("--gcs-output", help="Opt-in private gs:// output prefix")
    parser.add_argument(
        "--threshold-path",
        type=Path,
        help="Optional thresholds calibrated on this Stage-1 checkpoint's validation split; default is argmax.",
    )
    parser.add_argument("--sample-limit", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--train-epochs", type=int, default=1)
    parser.add_argument("--train-lr", type=float, default=2e-4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-existing-eval", action="store_true")
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_cmd(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(args, check=check, text=True, capture_output=True)


def load_thresholds(path: str | Path | None) -> dict[str, dict[str, float]]:
    """Load an explicitly selected calibration artifact, or use argmax."""
    if path is None:
        return {}
    threshold_path = Path(path)
    payload = json.loads(threshold_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"threshold file must contain a non-empty JSON object: {threshold_path}")
    calibrated: dict[str, dict[str, float]] = {}
    for abnormality, values in payload.items():
        if not isinstance(values, dict):
            raise ValueError(f"threshold entry for {abnormality!r} must be an object")
        class_thresholds = {}
        for class_name, value in values.items():
            if class_name not in CLASS_MAP:
                raise ValueError(f"unknown threshold class {class_name!r} for {abnormality!r}")
            numeric = float(value)
            if not 0.0 <= numeric <= 1.0:
                raise ValueError(f"threshold for {abnormality!r}/{class_name!r} is outside [0, 1]")
            class_thresholds[class_name] = numeric
        calibrated[str(abnormality)] = class_thresholds
    return calibrated


def assert_private_gcs_destination(gcs_path: str) -> str:
    """Fail closed unless a destination bucket is demonstrably private."""
    if not gcs_path.startswith("gs://") or not gcs_path[5:].split("/", 1)[0]:
        raise ValueError("Stage-2 artifacts may only be uploaded to a private gs:// destination")
    bucket_uri = "gs://" + gcs_path[5:].split("/", 1)[0]
    describe = run_cmd(
        ["gcloud", "storage", "buckets", "describe", bucket_uri, "--format=json"],
        check=False,
    )
    policy = run_cmd(
        ["gcloud", "storage", "buckets", "get-iam-policy", bucket_uri, "--format=json"],
        check=False,
    )
    if describe.returncode != 0 or policy.returncode != 0:
        details = (describe.stderr or policy.stderr or "gcloud verification failed").strip()
        raise RuntimeError(f"cannot verify that {bucket_uri} is private; refusing upload: {details}")
    try:
        metadata = json.loads(describe.stdout or "{}")
        iam_policy = json.loads(policy.stdout or "{}")
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"cannot parse privacy metadata for {bucket_uri}; refusing upload") from exc
    violations = private_bucket_violations(metadata, iam_policy)
    if violations:
        raise RuntimeError(f"refusing upload to {bucket_uri}: {'; '.join(violations)}")
    return bucket_uri


def upload_path(path: Path, gcs_dir: str) -> None:
    run_cmd(["gcloud", "storage", "cp", str(path), gcs_dir.rstrip("/") + "/"])


def upload_dir(path: Path, gcs_dir: str) -> None:
    run_cmd(["gcloud", "storage", "cp", "-r", str(path), gcs_dir.rstrip("/") + "/"])


def gcs_exists(gcs_path: str) -> bool:
    proc = run_cmd(["gcloud", "storage", "ls", gcs_path], check=False)
    return proc.returncode == 0


def read_gcs_csv(gcs_path: str, local_dir: Path) -> pd.DataFrame:
    local = local_dir / Path(gcs_path).name
    run_cmd(["gcloud", "storage", "cp", gcs_path, str(local)])
    return pd.read_csv(local)


def _stage1():
    """Import the Stage-1 stack, lazily and loudly.

    Kept out of module scope so that a ``medgemma_direct`` run -- which uses no
    Stage-1 checkpoint, config, encoder, Q-Former or MHCAC -- never imports
    LAVIS. Only the Q-Former branch of ``build_stage1_records`` calls this.
    """
    from training.stage1 import lavis_loader

    return lavis_loader


def default_stage1_config_path(run_name: str) -> Path:
    # Mirrors lavis_loader.default_stage1_config_path without importing it, so
    # that fingerprinting a cohort does not drag in the Stage-1 stack.
    return PROJECT_DIR / "pretraining/configs/encoder_comparison" / f"{run_name}.yaml"


def build_cfg(context: Stage1Context):
    return _stage1().build_cfg(context)


def stage1_checkpoint_path(context: Stage1Context, checkpoint_root: Path) -> Path:
    return context.resolve_checkpoint_path(checkpoint_root)


def build_stage1_model(context: Stage1Context, checkpoint_root: Path, device: torch.device):
    return _stage1().build_stage1_model(context, checkpoint_root, device)


def make_stage1_loader(cfg, split: str, sample_limit: int | None, num_workers: int):
    return _stage1().make_stage1_loader(cfg, split, sample_limit, num_workers)


def field_value(field, index: int = 0) -> str:
    if isinstance(field, (list, tuple)):
        return str(field[index])
    try:
        return str(field[index])
    except Exception:
        return str(field)


def classify_with_thresholds(context: Stage1Context, logits: torch.Tensor) -> dict[str, list[str]]:
    probs = torch.softmax(logits, dim=-1).tolist()
    out = {"positive": [], "negative": [], "uncertain": []}
    for abn, p in zip(ABNORMALITIES_14, probs):
        if abn == "No Finding":
            continue
        best_cls = select_threshold_class(
            p,
            context.threshold_for(abn),
            tuple(CLASS_MAP),
        )
        out[best_cls].append(abn)
    return out


def format_findings(groups: dict[str, list[str]]) -> str:
    parts = []
    for key in ["positive", "negative", "uncertain"]:
        if groups.get(key):
            parts.append(f"{key.capitalize()} findings: {', '.join(groups[key])}")
    return ". ".join(parts) if parts else "no common findings"


def image_block(img_token: str) -> str:
    return " ".join([img_token] * NUM_IMG_TOKENS)


def build_instruction(groups: dict[str, list[str]], prompt_style: str = "fine") -> str:
    findings = format_findings(groups)
    if prompt_style == "fine":
        return (
            f"Abnormality information: {findings}\n\n"
            "Act as an expert radiologist. Write only the Findings section of a chest "
            "X-ray report as one concise clinical paragraph. Do not invent facts, add "
            "an Impression section, or repeat the structured findings."
        )
    return (
        f"Abnormality information: {findings}\n\n"
        "Act as an expert radiologist. Using only the structured abnormality information and the image-derived features above, "
        "write the *Findings* section of a chest X-ray report.\n\n"
        "- Do not invent findings. Only describe abnormalities explicitly provided in the 'Abnormality information'.\n"
        "- Do not repeat the same information using different wording.\n"
        "- Use a single, fluent paragraph in formal radiological style.\n"
        "- Use cautious and precise language if uncertain abnormalities are present.\n"
        "- Avoid enumeration, bullet points, and speculative phrases.\n"
        "- The report should reflect the clinical tone and structure of professionally written reports.\n\n"
        "Return only the generated findings text."
    )


def build_native_instruction() -> str:
    """Prompt the native MedGemma baseline from pixels alone."""
    return native_findings_instruction()


def build_prompt(groups: dict[str, list[str]], img_token: str, prompt_style: str) -> str:
    """Legacy/plain-text prompt used by Vicuna and soft-token MedGemma."""
    return f"Image information: {image_block(img_token)}.\n\n{build_instruction(groups, prompt_style)}"


def _split_csv_for(split: str) -> str:
    return {
        "train": PROCESSED_TRAIN_CSV,
        "val": PROCESSED_VAL_CSV,
        "test": PROCESSED_TEST_CSV,
    }[split]


def data_object_identity(path: str | Path) -> dict[str, Any]:
    value = str(path)
    if not value.startswith("gs://"):
        return file_identity(value)
    proc = run_cmd(
        ["gcloud", "storage", "objects", "describe", value, "--format=json"],
        check=False,
    )
    if proc.returncode == 0:
        try:
            metadata = json.loads(proc.stdout)
            return {
                "uri_hash": stable_fingerprint({"uri": value}),
                "generation": metadata.get("generation"),
                "size": metadata.get("size"),
                "md5_hash": metadata.get("md5Hash"),
                "update_time": metadata.get("updateTime"),
            }
        except (TypeError, ValueError):
            pass
    # Still distinguish different cohorts if gcloud is unavailable. The cache
    # manifest records no credentialed bucket URI in plaintext.
    return {"uri_hash": stable_fingerprint({"uri": value}), "metadata_unavailable": True}


def stage1_cohort_fingerprint(
    context: Stage1Context, checkpoint_root: Path, split: str, sample_limit: int | None
) -> tuple[str, dict[str, Any]]:
    ckpt_path = stage1_checkpoint_path(context, checkpoint_root)
    cfg_path = context.resolve_config_path(
        default_stage1_config_path(context.run_name)
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_name": context.run_name,
        "split": split,
        "sample_limit": sample_limit if sample_limit and sample_limit > 0 else "all",
        "checkpoint": file_identity(ckpt_path),
        "stage1_config": file_identity(cfg_path),
        "split_csv": data_object_identity(_split_csv_for(split)),
        "thresholds": stable_fingerprint(context.fingerprint_payload()["thresholds"], length=32),
        "vis_root_name": Path(VIS_ROOT).name,
    }
    return stable_fingerprint(payload), payload


@torch.no_grad()
def build_stage1_records(
    context: Stage1Context,
    checkpoint_root: Path,
    output_dir: Path,
    split: str,
    sample_limit: int | None,
    num_workers: int,
) -> list[dict]:
    """Build Q-Former records. Stage-1 only -- native MedGemma must not call this.

    ``medgemma_direct`` records come from ``training/dataio/manifest.py`` via
    ``run_medgemma_qlora.build_native_records``, which reads the split CSVs
    directly. This function loads a Stage-1 checkpoint and iterates
    ``MIMIC_CXR_Dataset``, so routing native mode through it would reintroduce
    exactly the Stage-1 coupling the pipeline split exists to remove.
    """
    cohort_id, cohort = stage1_cohort_fingerprint(
        context, checkpoint_root, split, sample_limit
    )
    limit_name = str(sample_limit) if sample_limit and sample_limit > 0 else "all"
    # This local-only cache necessarily contains target report text and image
    # paths. Upload functions intentionally never include this directory.
    cache_dir = output_dir / ".sensitive_stage1_cache"
    cache_path = cache_dir / f"{context.run_name}_{split}_qformer_{limit_name}_{cohort_id}.pt"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        print(f"[stage1] reusing {cache_path}")
        cached = load_torch_checkpoint(cache_path)
        if cached.get("cohort_id") == cohort_id:
            return cached["records"]
        print("[stage1] cache manifest mismatch; rebuilding")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, model = build_stage1_model(context, checkpoint_root, device)
    loader = make_stage1_loader(cfg, split, sample_limit, num_workers)
    records = []
    skipped_invalid_targets = 0
    image_input_keys = {
        "image",
        "aux_image",
        "aux_mask",
        "anchor_view_id",
        "aux_view_ids",
        "biovil_feat",
        "pubmedclip_feat",
        "swin_feat",
        "raddino_feat",
        "aux_biovil_feat",
        "aux_pubmedclip_feat",
        "aux_swin_feat",
        "aux_raddino_feat",
    }
    for batch in tqdm(loader, desc=f"stage1 {split}"):
        generation_mask = batch.get("generation_mask")
        if not torch.is_tensor(generation_mask) or generation_mask.numel() != 1:
            raise RuntimeError("Stage-2 requires one scalar generation_mask per Stage-1 batch")
        if not bool(generation_mask.reshape(-1)[0].item()):
            skipped_invalid_targets += 1
            continue
        target = field_value(batch["text_output"]).strip()
        if not target:
            raise RuntimeError("generation_mask=true but FINDINGS target is blank")

        record = {
            "index": len(records),
            "sample_key": stable_fingerprint(
                {"cohort": cohort_id, "dicom": field_value(batch.get("dicom_id", ""))},
                length=24,
            ),
            "ref": target,
            "image_path": field_value(batch.get("image_path", "")),
        }
        model_inputs = {
            key: value.to(device, non_blocking=True)
            for key, value in batch.items()
            if key in image_input_keys and torch.is_tensor(value)
        }
        logits, qformer = model.forward_image(model_inputs)
        record["pred_groups"] = classify_with_thresholds(context, logits[0].detach().cpu())
        record["qformer_embs"] = qformer[0].detach().cpu().to(torch.float16)
        records.append(record)
    tmp_path = cache_path.with_suffix(".tmp")
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "cohort_id": cohort_id,
            "cohort": cohort,
            "contains_sensitive_data": True,
            "records": records,
        },
        tmp_path,
    )
    tmp_path.replace(cache_path)
    print(
        f"[stage1] wrote {cache_path} ({len(records)} valid FINDINGS records; "
        f"skipped {skipped_invalid_targets} invalid targets)"
    )
    del model
    clear_memory()
    return records


def hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def hf_kwargs() -> dict:
    token = hf_token()
    return {"token": token} if token else {}


def preferred_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


class RecordDataset(Dataset):
    def __init__(self, records: list[dict]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        return self.records[index]


class VariantLLM:
    def __init__(
        self,
        family: str,
        adapter: str | Path | None = None,
        train_adapter: bool = False,
        quantize_4bit: bool = False,
        image_mode: str = "qformer",
        lora_rank: int = 8,
        lora_alpha: int = 16,
        gradient_checkpointing: bool = True,
        prompt_config: PromptConfig | None = None,
    ):
        if image_mode not in ALL_IMAGE_MODES:
            raise ValueError(f"image_mode must be one of {sorted(ALL_IMAGE_MODES)}")
        if image_mode in {"native", "native_qformer", "text_only"} and family != "medgemma":
            raise ValueError(f"{image_mode} image mode is only supported for MedGemma")
        if image_mode == "native_qformer" and prompt_config is None:
            # The soft tokens live in the prompt text, and the legacy
            # build_native_instruction() emits none. Without a v2 prompt config
            # the substitution would find zero positions and silently return the
            # embeddings untouched -- a run that looks exactly like a working
            # combined mode while being plain native MedGemma.
            raise ValueError(
                "native_qformer requires --prompt-config: the soft-token "
                "placeholders come from the v2 prompt builder, and the legacy "
                "instruction emits none"
            )
        # Opt-in v2 prompt builder. None keeps the exact legacy prompt strings.
        if prompt_config is not None:
            if family != "medgemma":
                raise ValueError("prompt_config is only supported for MedGemma")
            if prompt_config.visual_mode.image_mode != image_mode:
                raise ValueError(
                    f"prompt_config.visual_mode={prompt_config.visual_mode.value} "
                    f"maps to image_mode {prompt_config.visual_mode.image_mode!r}, "
                    f"which does not match image_mode={image_mode!r}"
                )
        self.prompt_config = prompt_config
        # ``text_only`` is reachable only via the explicitly-named
        # text_only_language_prior_ablation pipeline mode. It is never a
        # fallback and never a default.
        self.capability = None
        self.family = family
        self.adapter = adapter
        self.train_adapter = train_adapter
        self.image_mode = image_mode
        self.quantize_4bit = quantize_4bit and family != "vicuna"
        self.dtype = preferred_dtype()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = None
        self.img_token = "<IMG>" if family == "vicuna" else "<qformer_soft_token>"

        if family == "vicuna":
            self.model_id = VICUNA_MODEL_ID
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                use_fast=False,
                truncation_side="right",
                padding_side="right",
                **hf_kwargs(),
            )
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.unk_token or self.tokenizer.eos_token
            self.tokenizer.add_special_tokens({"additional_special_tokens": [self.img_token]})
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
                **hf_kwargs(),
            )
            self.model.resize_token_embeddings(len(self.tokenizer))
        else:
            self.model_id = MEDGEMMA_MODEL_ID
            self.processor = AutoProcessor.from_pretrained(self.model_id, **hf_kwargs())
            self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
            self.tokenizer.truncation_side = "right"
            self.tokenizer.padding_side = "right"
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            load_kwargs = dict(
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
                attn_implementation="eager",
                **hf_kwargs(),
            )
            if self.quantize_4bit:
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=self.dtype,
                    bnb_4bit_use_double_quant=True,
                )
                # Single-process, single-GPU by design.
                # Pin to the current CUDA device rather than literal 0 so
                # CUDA_VISIBLE_DEVICES selection is honoured. Multi-GPU would
                # need DDP, not a wider device_map.
                load_kwargs["device_map"] = {
                    "": torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
                }
            # NO SILENT FALLBACK. This used to be a try/except that dropped to
            # AutoModelForCausalLM on any failure, which turned a vision run
            # into a language-prior run under a vision run's name. See
            # training/medgemma/capabilities.py for why that invalidates the
            # ablation rather than merely degrading it.
            if self.image_mode == "text_only":
                # Reached only when the user explicitly selected
                # text_only_language_prior_ablation.
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id, **load_kwargs
                )
            else:
                try:
                    self.model = AutoModelForImageTextToText.from_pretrained(
                        self.model_id, **load_kwargs
                    )
                except Exception as exc:
                    raise MultimodalModelLoadError(
                        model_id=self.model_id,
                        revision=load_kwargs.get("revision"),
                        transformers_version=getattr(transformers, "__version__", None),
                        original=exc,
                    ) from exc
                self.capability = validate_multimodal_capability(
                    self.model,
                    self.processor,
                    model_id=self.model_id,
                    revision=load_kwargs.get("revision"),
                    transformers_version=getattr(transformers, "__version__", None),
                )
            if self.image_mode in SOFT_TOKEN_MODES and self.img_token not in self.tokenizer.get_vocab():
                self.tokenizer.add_special_tokens({"additional_special_tokens": [self.img_token]})
                self.model.resize_token_embeddings(len(self.tokenizer))
            if self.quantize_4bit:
                self.model = prepare_model_for_kbit_training(
                    self.model, use_gradient_checkpointing=gradient_checkpointing and train_adapter
                )
                self._align_output_head_dtype()

        if not self.quantize_4bit:
            self.model.to(self.device)
        self.img_token_id = None
        self.img_proj = None
        if self.image_mode in SOFT_TOKEN_MODES:
            self.img_token_id = self.tokenizer.convert_tokens_to_ids(self.img_token)
            if self.img_token_id is None or self.img_token_id < 0:
                raise RuntimeError(f"could not register image token {self.img_token}")
            hidden = int(self.model.get_input_embeddings().weight.shape[-1])
            # Keep the newly initialized bridge in fp32; it is much smaller than
            # the LLM and benefits from stable updates at its higher learning rate.
            self.img_proj = nn.Linear(768, hidden).to(self.device, dtype=torch.float32)
        if adapter:
            self.model = PeftModel.from_pretrained(self.model, str(adapter), is_trainable=train_adapter)
        elif train_adapter:
            targets = self._language_lora_targets()
            cfg = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=targets,
            )
            self.model = get_peft_model(self.model, cfg)

        if hasattr(self.model, "config"):
            self.model.config.use_cache = False
        if getattr(self.model, "generation_config", None) is not None:
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
            self.model.generation_config.eos_token_id = self.tokenizer.eos_token_id

        if not self.quantize_4bit:
            self.model.to(self.device)
        else:
            self._align_output_head_dtype()

    def _language_lora_targets(self) -> list[str]:
        suffixes = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
        if self.family == "vicuna":
            return sorted(suffixes)
        # QLoRA replaces ``nn.Linear`` with bitsandbytes Linear4bit modules,
        # so selecting by module type yields an empty target list.  Restrict by
        # full module name instead, while retaining the guard against adapting
        # MedGemma's vision tower.
        names = language_lora_target_names(
            [name for name, _module in self.model.named_modules()]
        )
        if not names:
            raise RuntimeError(
                "could not identify MedGemma language-layer LoRA targets; refusing to "
                "fall back to all-linear because that also adapts the vision tower"
            )
        return names

    def _align_output_head_dtype(self) -> None:
        candidates = [
            self.model,
            getattr(self.model, "base_model", None),
            getattr(getattr(self.model, "base_model", None), "model", None),
        ]
        for module in candidates:
            head = getattr(module, "lm_head", None)
            if head is not None:
                head.to(device=self.device, dtype=self.dtype)

    def parameter_report(self) -> dict:
        """Trainable/frozen accounting for the run manifest.

        Under NF4 the base weights are packed, so ``total`` is the storage
        element count rather than the dense parameter count. The trainable and
        LoRA figures are exact because adapters are never quantized.
        """
        vision_markers = ("vision_tower", "vision_model", "image_tower", "multi_modal_projector")
        total = trainable = lora = vision = trainable_vision = 0
        for name, param in self.model.named_parameters():
            count = param.numel()
            is_vision = any(marker in name for marker in vision_markers)
            total += count
            vision += count if is_vision else 0
            if param.requires_grad:
                trainable += count
                lora += count if "lora_" in name else 0
                trainable_vision += count if is_vision else 0
        projector = (
            sum(p.numel() for p in self.img_proj.parameters())
            if self.img_proj is not None
            else 0
        )
        return {
            "total_parameters": total + projector,
            "trainable_parameters": trainable + projector,
            "lora_parameters": lora,
            "projector_parameters": projector,
            "vision_parameters": vision,
            "trainable_vision_parameters": trainable_vision,
            "trainable_fraction": round(
                (trainable + projector) / max(total + projector, 1), 6
            ),
            "note": "NF4-packed base weights; trainable/LoRA counts are exact",
        }

    def assert_vision_tower_frozen(self) -> None:
        """LoRA must adapt language layers only unless explicitly configured.

        ``target_modules="all-linear"`` would also wrap the image tower, which
        silently changes what the ablation is measuring.
        """
        report = self.parameter_report()
        if report["trainable_vision_parameters"]:
            raise RuntimeError(
                "LoRA targets reached MedGemma's vision tower "
                f"({report['trainable_vision_parameters']} trainable vision params); "
                "language-only targeting is required"
            )

    def _prompt_metadata(self) -> dict:
        """Reproducibility record for the prompt used by this run."""
        if self.prompt_config is None:
            return {
                "builder": "legacy",
                "version": "legacy_build_instruction",
                "image_mode": self.image_mode,
                "num_img_tokens": NUM_IMG_TOKENS,
            }
        config = self.prompt_config
        return {
            "builder": "stage2.prompts.PromptBuilder",
            "version": config.version,
            "visual_mode": config.visual_mode.value,
            "normal_policy": config.normal_policy.value,
            "negative_policy": config.negative_policy.value,
            "uncertainty_policy": config.uncertainty_policy.value,
            "temporal_target_policy": config.temporal_target_policy.value,
            "config_hash": config.config_hash(),
            "template_hash": _prompt_template_hash(config.visual_mode),
            "num_img_tokens": NUM_IMG_TOKENS,
            "tokenizer": self.model_id,
            "processor": self.model_id,
        }

    def save_adapter(
        self,
        out_dir: Path,
        *,
        status: str = "complete",
        trainer_state: dict | None = None,
        training_config: dict | None = None,
    ) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(out_dir)
        if self.img_proj is not None:
            torch.save(self.img_proj.state_dict(), out_dir / "img_proj.pt")
        prompt_meta = self._prompt_metadata()
        meta = {
            "family": self.family,
            "model_id": self.model_id,
            "img_token": self.img_token,
            "img_token_id": self.img_token_id,
            "num_img_tokens": NUM_IMG_TOKENS,
            "image_mode": self.image_mode,
            "prompt": prompt_meta,
            # Recorded so a checkpoint can never be mistaken later for a vision
            # run when it was in fact the language-prior ablation.
            **(
                self.capability.as_metadata()
                if self.capability is not None
                else {"multimodal": False, "capability_checks": {}, "capability_failures": []}
            ),
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "family": self.family,
            "model_id": self.model_id,
            "image_mode": self.image_mode,
            "prompt": prompt_meta,
            "training_config": training_config or {},
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if trainer_state is not None:
            torch.save(trainer_state, out_dir / "trainer_state.pt")

    def load_img_proj_if_present(self, adapter_dir: Path | str | None) -> None:
        if not adapter_dir:
            return
        p = Path(adapter_dir) / "img_proj.pt"
        if p.exists() and self.img_proj is not None:
            self.img_proj.load_state_dict(load_torch_checkpoint(p))

    def _render_prompt_text(self, record: dict) -> str:
        """User-turn text from the shared v2 builder (opt-in via prompt_config).

        For qformer modes the returned string already contains the 32
        ``<qformer_soft_token>`` placeholders; for native modes it is the
        instruction text only (pixels are attached separately). Train and
        inference both call this, so the user turn is identical by construction.
        """
        context = context_from_record(
            record,
            visual_mode=self.prompt_config.visual_mode,
            qformer_token_count=NUM_IMG_TOKENS,
            prompt_version=self.prompt_config.version,
        )
        return PromptBuilder(self.prompt_config).build(context).user_text(self.img_token)

    def _chat_texts(self, record: dict, prompt_style: str) -> tuple[str, str]:
        target = str(record["ref"]).strip()
        if self.family != "medgemma":
            prompt = build_prompt(record["pred_groups"], self.img_token, prompt_style) + "\nASSISTANT:"
            return prompt, prompt + " " + target + (self.tokenizer.eos_token or "")
        content: list[dict[str, Any]] = []
        if self.image_mode in NATIVE_PIXEL_MODES:
            # Any mode that feeds MedGemma's own vision tower must go through
            # _native_chat_inputs, which embeds the real PIL image so the
            # processor expands the image tokens identically for prompt and
            # full chat. Rendering to a string here would produce a different
            # image-token prefix and unmask prompt tokens during training.
            raise RuntimeError(
                f"_chat_texts is the string path; image_mode={self.image_mode!r} "
                "must use _native_chat_inputs"
            )
        if self.prompt_config is not None:
            instruction = self._render_prompt_text(record)
        else:
            instruction = build_prompt(record["pred_groups"], self.img_token, prompt_style)
        content.append({"type": "text", "text": instruction})
        prompt_messages = [{"role": "user", "content": content}]
        full_messages = prompt_messages + [
            {"role": "assistant", "content": [{"type": "text", "text": target}]}
        ]
        prompt = self.processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        full = self.processor.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )
        return prompt, full

    @staticmethod
    def _load_rgb(record: dict) -> Image.Image:
        image_path = Path(str(record.get("image_path", "")))
        if not image_path.is_file():
            raise FileNotFoundError(f"native-image input is unavailable: {image_path.name}")
        with Image.open(image_path) as image:
            return image.convert("RGB").copy()

    def _native_messages(self, record: dict, *, include_target: bool) -> list[dict[str, Any]]:
        """Build MedGemma messages with the real image object embedded.

        ``AutoProcessor.apply_chat_template`` is the supported MedGemma path:
        it expands image tokens consistently for both prompt and full chat.
        Rendering text first and then calling the processor can produce a
        different image-token prefix, which would accidentally unmask prompt
        tokens during supervised fine tuning.
        """
        instruction = (
            self._render_prompt_text(record)
            if self.prompt_config is not None
            else build_native_instruction()
        )
        content: list[dict[str, Any]] = [
            {"type": "image", "image": self._load_rgb(record)},
            {"type": "text", "text": instruction},
        ]
        messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
        if include_target:
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": str(record["ref"]).strip()}],
                }
            )
        return messages

    def _native_chat_inputs(
        self,
        record: dict,
        *,
        include_target: bool,
        add_generation_prompt: bool,
        max_length: int,
    ):
        return self.processor.apply_chat_template(
            self._native_messages(record, include_target=include_target),
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_dict=True,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )

    def encode_train_example(self, record: dict, prompt_style: str, max_length: int = 768) -> dict:
        tokenize_kwargs = dict(
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )
        if self.image_mode in NATIVE_PIXEL_MODES:
            prompt_encoded = self._native_chat_inputs(
                record,
                include_target=False,
                add_generation_prompt=True,
                max_length=max_length,
            )
            encoded = self._native_chat_inputs(
                record,
                include_target=True,
                add_generation_prompt=False,
                max_length=max_length,
            )
            prompt_ids = prompt_encoded["input_ids"][0].tolist()
        else:
            prompt, full = self._chat_texts(record, prompt_style)
            encoded = self.tokenizer(full, **tokenize_kwargs)
            prompt_ids = self.tokenizer(
                prompt,
                truncation=True,
                max_length=max_length,
                add_special_tokens=False,
            ).input_ids
        full_ids = encoded["input_ids"][0].tolist()
        # This sequence is not padded yet. In Gemma-family tokenizers PAD may
        # equal EOS, so masking by token value here would remove the target EOS.
        labels = masked_label_ids(full_ids, prompt_ids)
        if not any(label != -100 for label in labels):
            raise ValueError("target was completely truncated; increase max_length")
        item = {key: value[0] for key, value in encoded.items() if torch.is_tensor(value)}
        item["labels"] = torch.tensor(labels, dtype=torch.long)
        if self.image_mode in SOFT_TOKEN_MODES:
            item["qformer_embs"] = record["qformer_embs"].float()
        return item

    def collate_train(self, records: list[dict], max_length: int = 768) -> dict[str, torch.Tensor]:
        items = [self.encode_train_example(record, "fine", max_length) for record in records]
        sequence_keys = {"input_ids", "attention_mask", "token_type_ids", "labels"}
        max_len = max(item["input_ids"].shape[0] for item in items)
        batch: dict[str, torch.Tensor] = {}
        for key in sequence_keys:
            if key not in items[0]:
                continue
            pad_value = -100 if key == "labels" else (self.tokenizer.pad_token_id if key == "input_ids" else 0)
            padded = []
            for item in items:
                tensor = item[key]
                padding = torch.full((max_len - tensor.shape[0],), pad_value, dtype=tensor.dtype)
                padded.append(torch.cat([tensor, padding]))
            batch[key] = torch.stack(padded)
        # Stack every non-sequence tensor, whatever it is. The combined mode
        # carries BOTH pixel_values and qformer_embs, so the old
        # qformer-or-everything-else branch would have dropped one of them.
        for key in items[0]:
            if key not in sequence_keys:
                batch[key] = torch.stack([item[key] for item in items])
        return batch

    def _forward_batch(self, batch: dict[str, torch.Tensor]):
        moved = {
            key: (
                value.to(self.device, dtype=self.dtype, non_blocking=True)
                if value.is_floating_point()
                else value.to(self.device, non_blocking=True)
            )
            for key, value in batch.items()
        }
        qformer = moved.pop("qformer_embs", None)
        old_embedding = None
        if self.image_mode in SOFT_TOKEN_MODES:
            projected = self.img_proj(qformer.float())
            old_embedding = self.model.get_input_embeddings()
            self.model.set_input_embeddings(
                SoftTokenEmbeddingWrapper(
                    old_embedding, self.img_token_id, projected, NUM_IMG_TOKENS
                )
            )
        try:
            return self.model(**moved)
        finally:
            if old_embedding is not None:
                self.model.set_input_embeddings(old_embedding)

    @torch.no_grad()
    def evaluate_loss(self, records: list[dict], batch_size: int, max_length: int) -> float:
        if not records:
            return float("nan")
        self.model.eval()
        if self.img_proj is not None:
            self.img_proj.eval()
        loader = DataLoader(
            RecordDataset(records),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda rows: self.collate_train(rows, max_length),
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        weighted_loss, examples = 0.0, 0
        for batch in tqdm(loader, desc=f"{self.family} {self.image_mode} val loss"):
            output = self._forward_batch(batch)
            current = int(batch["input_ids"].shape[0])
            weighted_loss += float(output.loss.detach().float().cpu()) * current
            examples += current
        return weighted_loss / max(examples, 1)

    def train_fine(
        self,
        records: list[dict],
        out_dir: Path,
        epochs: int,
        lr: float | None = None,
        grad_accum: int = 8,
        *,
        val_records: list[dict] | None = None,
        batch_size: int = 2,
        lora_lr: float | None = None,
        projector_lr: float = 1e-3,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.03,
        max_grad_norm: float = 1.0,
        max_length: int = 768,
        patience: int = 2,
        seed: int = SEED,
        resume_state: str | Path | None = None,
        save_every_updates: int = 0,
    ) -> dict:
        if not records:
            raise ValueError("training records are empty")
        lora_lr = float(lora_lr if lora_lr is not None else (lr if lr is not None else 1e-4))
        model_params = [p for p in self.model.parameters() if p.requires_grad]
        param_groups = [{"params": model_params, "lr": lora_lr, "weight_decay": weight_decay}]
        if self.img_proj is not None:
            param_groups.append(
                {"params": list(self.img_proj.parameters()), "lr": projector_lr, "weight_decay": weight_decay}
            )
        params = [param for group in param_groups for param in group["params"]]
        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999))
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(
            RecordDataset(records),
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            collate_fn=lambda rows: self.collate_train(rows, max_length),
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        updates_per_epoch = math.ceil(len(loader) / grad_accum)
        total_updates = max(1, updates_per_epoch * epochs)
        warmup_steps = int(round(total_updates * warmup_ratio))
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_updates)
        start_epoch, global_step, best_val, bad_epochs = 0, 0, float("inf"), 0
        if resume_state:
            state_path = Path(resume_state)
            if state_path.is_dir():
                state_path = state_path / "trainer_state.pt"
            if state_path.is_file():
                state = load_torch_checkpoint(state_path)
                optimizer.load_state_dict(state["optimizer"])
                scheduler.load_state_dict(state["scheduler"])
                start_epoch = int(state.get("epoch", -1)) + 1
                global_step = int(state.get("global_step", 0))
                best_val = float(state.get("best_val_loss", float("inf")))
                bad_epochs = int(state.get("bad_epochs", 0))
                if state.get("data_generator_state") is not None:
                    generator.set_state(state["data_generator_state"])
                if state.get("torch_rng_state") is not None:
                    torch.set_rng_state(state["torch_rng_state"])
                if torch.cuda.is_available() and state.get("cuda_rng_state") is not None:
                    torch.cuda.set_rng_state_all(state["cuda_rng_state"])
                print(f"[train] resumed optimizer/scheduler at epoch={start_epoch} step={global_step}")

        self.assert_vision_tower_frozen()
        parameters = self.parameter_report()
        print(
            f"[params] trainable={parameters['trainable_parameters']:,} "
            f"lora={parameters['lora_parameters']:,} "
            f"projector={parameters['projector_parameters']:,} "
            f"({parameters['trainable_fraction'] * 100:.4f}% of {parameters['total_parameters']:,})",
            flush=True,
        )
        training_config = {
            "parameters": parameters,
            "epochs": epochs,
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "effective_batch_size": batch_size * grad_accum,
            "lora_lr": lora_lr,
            "projector_lr": projector_lr if self.img_proj is not None else None,
            "weight_decay": weight_decay,
            "warmup_ratio": warmup_ratio,
            "max_grad_norm": max_grad_norm,
            "max_length": max_length,
            "train_samples": len(records),
            "val_samples": len(val_records or []),
        }
        last_dir = out_dir / "checkpoints" / "last"
        for epoch in range(start_epoch, epochs):
            self.model.train()
            if self.img_proj is not None:
                self.img_proj.train()
            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0
            progress = tqdm(loader, desc=f"{self.family} {self.image_mode} train {epoch + 1}/{epochs}")
            for batch_index, batch in enumerate(progress):
                output = self._forward_batch(batch)
                raw_loss = output.loss
                divisor = accumulation_window_size(batch_index, len(loader), grad_accum)
                (raw_loss / divisor).backward()
                running_loss += float(raw_loss.detach().float().cpu())
                end_window = (batch_index + 1) % grad_accum == 0 or batch_index + 1 == len(loader)
                if end_window:
                    torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    if save_every_updates and global_step % save_every_updates == 0:
                        # RECOVERY ARTIFACT, not a resume point. save_adapter is
                        # otherwise called only after the batch loop ends, so a
                        # single epoch over the full 176k-record cohort runs
                        # ~70 h with nothing on disk, and a hang at hour 69
                        # loses all of it. This machine has a documented
                        # history of hard hangs, so that is not a theoretical
                        # risk.
                        #
                        # ⚠ Resuming from this still restarts the epoch:
                        # ``resume_state`` sets start_epoch = epoch + 1 and
                        # nothing skips the batches already seen. What it
                        # saves is the trained ADAPTER, which is the expensive
                        # part; status="in_progress" marks it so a partial
                        # adapter is never mistaken for a finished run.
                        self.save_adapter(
                            last_dir,
                            status="in_progress",
                            trainer_state={
                                "epoch": epoch,
                                "global_step": global_step,
                                "best_val_loss": best_val,
                                "bad_epochs": bad_epochs,
                                "optimizer": optimizer.state_dict(),
                                "scheduler": scheduler.state_dict(),
                                "data_generator_state": generator.get_state(),
                                "torch_rng_state": torch.get_rng_state(),
                                "cuda_rng_state": (
                                    torch.cuda.get_rng_state_all()
                                    if torch.cuda.is_available() else None
                                ),
                            },
                            training_config=training_config,
                        )
                        print(
                            f"[train] recovery checkpoint at update {global_step} "
                            f"(epoch {epoch + 1}, batch {batch_index + 1}/{len(loader)})",
                            flush=True,
                        )
                progress.set_postfix(loss=f"{float(raw_loss.detach().float().cpu()):.4f}")

            train_loss = running_loss / max(len(loader), 1)
            val_loss = self.evaluate_loss(val_records or [], batch_size, max_length)
            score = val_loss if math.isfinite(val_loss) else train_loss
            improved = score < best_val
            if improved:
                best_val, bad_epochs = score, 0
            else:
                bad_epochs += 1
            trainer_state = {
                "epoch": epoch,
                "global_step": global_step,
                "best_val_loss": best_val,
                "bad_epochs": bad_epochs,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "data_generator_state": generator.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            }
            self.save_adapter(
                last_dir,
                status="resumable",
                trainer_state=trainer_state,
                training_config=training_config,
            )
            if improved:
                self.save_adapter(
                    out_dir,
                    status="complete",
                    trainer_state=trainer_state,
                    training_config=training_config,
                )
            print(
                f"[epoch {epoch + 1}] train_loss={train_loss:.5f} "
                f"val_loss={val_loss:.5f} best={best_val:.5f}"
            )
            if bad_epochs >= patience:
                print(f"[train] early stopping after {bad_epochs} non-improving epoch(s)")
                break
        if not adapter_is_complete(out_dir, self.image_mode):
            # Covers a resume taken after the last-epoch checkpoint was written
            # but before the best-checkpoint promotion completed.
            recovery_state = {
                "epoch": max(start_epoch - 1, 0),
                "global_step": global_step,
                "best_val_loss": best_val,
                "bad_epochs": bad_epochs,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "data_generator_state": generator.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            }
            self.save_adapter(
                out_dir,
                status="complete",
                trainer_state=recovery_state,
                training_config=training_config,
            )
        return {"global_step": global_step, "best_val_loss": best_val, **training_config}

    @torch.no_grad()
    def generate(self, record: dict, prompt_style: str, max_new_tokens: int) -> str:
        self.model.eval()
        if self.img_proj is not None:
            self.img_proj.eval()
        if self.image_mode in NATIVE_PIXEL_MODES:
            encoded = self._native_chat_inputs(
                record,
                include_target=False,
                add_generation_prompt=True,
                max_length=768,
            )
        else:
            prompt, _ = self._chat_texts(record, prompt_style)
            encoded = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=768,
                add_special_tokens=False,
            )
        input_ids = encoded["input_ids"].to(self.device)
        model_inputs = {
            key: (
                value.to(self.device, dtype=self.dtype)
                if value.is_floating_point()
                else value.to(self.device)
            )
            for key, value in encoded.items()
            if torch.is_tensor(value)
        }
        old_embedding = None
        if self.image_mode in SOFT_TOKEN_MODES:
            qformer = record["qformer_embs"].unsqueeze(0).to(self.device, dtype=torch.float32)
            projected = self.img_proj(qformer)
            old_embedding = self.model.get_input_embeddings()
            self.model.set_input_embeddings(
                SoftTokenEmbeddingWrapper(
                    old_embedding, self.img_token_id, projected, NUM_IMG_TOKENS
                )
            )
        try:
            generate_kwargs = {
                **model_inputs,
                "max_new_tokens": max_new_tokens,
                "num_beams": 1,
                "do_sample": False,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "use_cache": True,
                "return_dict_in_generate": False,
            }
            # Stop the model spending probability mass on the soft-token
            # placeholder; None (native/text-only) leaves generation unchanged.
            bad_words = soft_token_bad_words_ids(
                self.img_token_id if self.image_mode in SOFT_TOKEN_MODES else None
            )
            if bad_words is not None:
                generate_kwargs["bad_words_ids"] = bad_words
            seq = self.model.generate(**generate_kwargs)[0]
        finally:
            if old_embedding is not None:
                self.model.set_input_embeddings(old_embedding)
        gen_ids = seq[input_ids.shape[1] :] if seq.shape[0] > input_ids.shape[1] else seq
        return clean_text(self.tokenizer.decode(gen_ids, skip_special_tokens=True))


def clean_text(text: str) -> str:
    text = str(text).strip()
    for marker in ["ASSISTANT:", "Assistant:", "assistant:", "model:", "model\n"]:
        if marker in text:
            text = text.split(marker)[-1].strip()
    return text


def tokenize(text: str):
    return re.findall(r"\w+", str(text).lower())


def compute_nlg(preds: list[str], refs: list[str]) -> dict:
    meteor_vals = []
    tokenized_predictions = [tokenize(pred) for pred in preds]
    tokenized_references = [[tokenize(ref)] for ref in refs]
    for pred, ref in zip(preds, refs):
        pred_tok = tokenize(pred)
        ref_tok = [tokenize(ref)]
        if not pred_tok:
            meteor_vals.append(0.0)
            continue
        try:
            meteor_vals.append(meteor_score(ref_tok, pred_tok) if ref_tok and isinstance(ref_tok[0], list) else meteor_score(ref_tok, pred_tok))
        except Exception:
            try:
                meteor_vals.append(meteor_score([tokenize(ref)], pred_tok))
            except Exception:
                meteor_vals.append(0.0)
    gts = {i: [r] for i, r in enumerate(refs)}
    res = {i: [p] for i, p in enumerate(preds)}
    try:
        _, rouge_scores = Rouge().compute_score(gts, res)
        rouge_l = float(np.mean(rouge_scores))
    except Exception:
        rouge_l = 0.0
    try:
        cider, _ = Cider().compute_score(gts, res)
        cider = float(cider) if cider else 0.0
    except Exception:
        cider = 0.0
    _p, _r, f1 = bert_score_fn(
        preds,
        refs,
        lang="en",
        model_type=BERTSCORE_MODEL,
        rescale_with_baseline=False,
        verbose=False,
        device="cpu",
    )
    bleu = {}
    for n in range(1, 5):
        weights = tuple(1.0 / n if i < n else 0.0 for i in range(4))
        bleu[n] = corpus_bleu(
            tokenized_references,
            tokenized_predictions,
            weights=weights,
            smoothing_function=SMOOTH,
        )
    return {
        "BLEU-1": round(float(bleu[1]), 4),
        "BLEU-2": round(float(bleu[2]), 4),
        "BLEU-3": round(float(bleu[3]), 4),
        "BLEU-4": round(float(bleu[4]), 4),
        "METEOR": round(float(np.mean(meteor_vals)), 4),
        "ROUGE-L": round(rouge_l, 4),
        "CIDEr": round(cider, 4),
        "BERTScore": round(float(f1.mean().item()), 4),
    }


def _unzip_sections(texts: list[str]) -> tuple[list[str], list[str]]:
    """Split each text into (findings, impression), keeping row alignment."""
    findings: list[str] = []
    impression: list[str] = []
    for text in texts:
        section_findings, section_impression = split_generated_report(text)
        findings.append(section_findings)
        impression.append(section_impression)
    return findings, impression


def compute_sectioned_nlg(preds: list[str], refs: list[str], section_mode: str) -> dict:
    """Score the full report and, when both sections are targets, each section.

    ``split_generated_report`` treats an unheadered generation as FINDINGS with an
    empty IMPRESSION rather than duplicating it into both, so a model that never
    emits an IMPRESSION header scores 0 on the impression block instead of being
    silently credited with its findings text.
    """
    metrics = compute_nlg(preds, refs)
    if section_mode != FINDINGS_AND_IMPRESSION:
        return metrics

    pred_findings, pred_impression = _unzip_sections(preds)
    ref_findings, ref_impression = _unzip_sections(refs)
    for label, pred_section, ref_section in (
        ("Findings", pred_findings, ref_findings),
        ("Impression", pred_impression, ref_impression),
    ):
        metrics.update(
            prefix_metric_keys(
                compute_nlg(list(pred_section), list(ref_section)), label
            )
        )
    metrics["FindingsOmissionRate"] = round(
        section_omission_rate(list(pred_findings), list(ref_findings)), 6
    )
    metrics["ImpressionOmissionRate"] = round(
        section_omission_rate(list(pred_impression), list(ref_impression)), 6
    )
    return metrics


def evaluate_variant(
    family: str,
    variant: str,
    llm: VariantLLM,
    records: list[dict],
    out_dir: Path,
    max_new_tokens: int,
    prompt_style: str,
    *,
    cohort_id: str | None = None,
    include_sensitive_fields: bool = False,
    section_mode: str = FINDINGS_ONLY,
    context: Stage1Context,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cohort_id = cohort_id or stable_fingerprint(
        {
            "sample_keys": [record.get("sample_key", record.get("index")) for record in records],
            "count": len(records),
        }
    )
    adapter_path = Path(llm.adapter) if llm.adapter else None
    adapter_manifest = file_identity(adapter_path / "manifest.json") if adapter_path else {"base": True}
    eval_id = stable_fingerprint(
        {
            "schema_version": SCHEMA_VERSION,
            "family": family,
            "variant": variant,
            "run": context.run_name,
            "image_mode": llm.image_mode,
            "cohort_id": cohort_id,
            "adapter": adapter_manifest,
            "max_new_tokens": max_new_tokens,
            "prompt_style": prompt_style,
        }
    )
    stem = f"{family}_{variant}_{llm.image_mode}_{context.run_name}_{eval_id}"
    # Generated text is a MIMIC derivative. Keep it in an explicitly sensitive
    # local resume cache; upload callers only copy aggregate files in out_dir.
    prediction_cache = out_dir / ".sensitive_predictions"
    prediction_cache.mkdir(parents=True, exist_ok=True)
    jsonl_path = prediction_cache / f"predictions_{stem}.jsonl"
    metrics_path = out_dir / f"metrics_{stem}.json"
    if metrics_path.exists() and jsonl_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    preds: list[str] = []
    failures = 0
    if jsonl_path.exists():
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                preds.append(str(item.get("pred", "")))
                failures += int(not bool(item.get("generation_ok", True)))
        if len(preds) > len(records):
            preds, failures = [], 0
        elif preds:
            print(f"[{family} {variant}] resuming eval from {len(preds)}/{len(records)} records")
    refs = [str(record["ref"]) for record in records]
    mode = "a" if preds else "w"
    with open(jsonl_path, mode, encoding="utf-8") as f:
        for record in tqdm(records[len(preds) :], desc=f"{family} {variant} eval"):
            generation_ok = True
            try:
                pred = llm.generate(record, prompt_style, max_new_tokens)
            except Exception as exc:
                print(f"generate failed at {record.get('index')}: {exc}")
                pred = ""
                generation_ok = False
                failures += 1
            preds.append(pred)
            row = safe_prediction_row(
                sample_key=record.get("sample_key", stable_fingerprint({"index": record.get("index")})),
                index=record.get("index", len(preds) - 1),
                prediction=pred,
                generation_ok=generation_ok,
            )
            if include_sensitive_fields:
                row.update({"ref": str(record["ref"]), "image_path": str(record.get("image_path", ""))})
            f.write(json.dumps(row) + "\n")
            f.flush()
    metrics = compute_sectioned_nlg(preds, refs, section_mode)
    metrics.update(
        {
            "Family": family,
            "SectionMode": section_mode,
            "Variant": variant,
            "Run": context.run_name,
            "ImageMode": llm.image_mode,
            "N": len(records),
            "MaxNewTokens": max_new_tokens,
            "CohortId": cohort_id,
            "EvalId": eval_id,
            "GenerationFailures": failures,
            "GenerationFailureRate": round(failures / max(len(records), 1), 6),
            "PredictionsFile": jsonl_path.name,
        }
    )
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def instruction_metrics(context: Stage1Context, family: str, work_dir: Path) -> dict:
    table = read_gcs_csv(INSTRUCTION_TABLES[family], work_dir)
    row = table[table["Run"] == context.run_name].iloc[0]
    return {
        "Family": family,
        "Variant": "instruction",
        "Run": context.run_name,
        "N": 200,
        "BLEU-1": float(row["BLEU-1"]),
        "BLEU-2": float(row["BLEU-2"]),
        "BLEU-3": float(row["BLEU-3"]),
        "BLEU-4": float(row["BLEU-4"]),
        "METEOR": float(row["METEOR"]),
        "ROUGE-L": float(row["ROUGE-L"]),
        "CIDEr": float(row["CIDEr"]),
        "BERTScore": float(row["BERTScore"]),
        "Source": INSTRUCTION_TABLES[family],
    }


def plot_family(family: str, metrics_df: pd.DataFrame, out_dir: Path) -> dict:
    labels = ["BERTScore", "BLEU-4", "ROUGE-L", "METEOR", "CIDEr"]
    names = {
        "base": "Base LLM",
        "fine": "Fine-Tuned LLM",
        "instruction": "Instruction-Tuned LLM",
    }
    colors = {"base": "#2563eb", "fine": "#16a34a", "instruction": "#dc2626"}
    linestyles = {"base": "--", "fine": "-", "instruction": "-."}
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=(6.8, 6.4), dpi=220)
    ax = fig.add_subplot(111, polar=True)
    for variant in ["base", "fine", "instruction"]:
        row = metrics_df[metrics_df["Variant"] == variant].iloc[0]
        values = [float(row[m]) for m in labels]
        values += values[:1]
        ax.plot(angles, values, label=names[variant], color=colors[variant], linestyle=linestyles[variant], linewidth=2.0)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    max_value = max(float(metrics_df[m].max()) for m in labels)
    rmax = max(0.5, min(1.0, np.ceil((max_value + 0.06) * 10) / 10))
    ax.set_ylim(0, rmax)
    yticks = np.arange(0.1, rmax + 0.001, 0.1)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{x:.1f}" for x in yticks], fontsize=8, color="#555555")
    ax.grid(color="#b8b8b8", linewidth=0.65, alpha=0.85)
    title_name = "Vicuna" if family == "vicuna" else "MedGemma1.5"
    ax.set_title(f"{title_name} LLM Comparison", fontsize=12, pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.12), fontsize=8, frameon=True)
    fig.text(0.08, 0.035, "Figure 9. Effect of Tuned LLMs.", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0.03, 0.07, 0.96, 0.96))
    png = out_dir / f"figure9_{family}_three_llm_types_200.png"
    pdf = out_dir / f"figure9_{family}_three_llm_types_200.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"figure_png": str(png), "figure_pdf": str(pdf)}


def run_family(context: Stage1Context, args: argparse.Namespace, family: str, train_records: list[dict], test_records: list[dict], root: Path) -> dict:
    family_dir = root / family
    adapters_dir = family_dir / "adapters"
    eval_dir = family_dir / "eval"
    family_dir.mkdir(parents=True, exist_ok=True)
    fine_adapter_dir = adapters_dir / f"{family}_fine_lora_200"

    metrics = []

    if (not args.skip_existing_eval) or not (eval_dir / f"metrics_{family}_base_{context.run_name}.json").exists():
        print(f"[{family}] evaluating base")
        llm = VariantLLM(family)
        metrics.append(evaluate_variant(family, "base", llm, test_records, eval_dir, args.max_new_tokens, "fine", context=context))
        del llm
        clear_memory()
    else:
        metrics.append(json.loads((eval_dir / f"metrics_{family}_base_{context.run_name}.json").read_text(encoding="utf-8")))

    if not args.skip_train and (args.force or not adapter_is_complete(fine_adapter_dir, "qformer")):
        print(f"[{family}] training fine LoRA -> {fine_adapter_dir}")
        llm = VariantLLM(family, train_adapter=True)
        llm.train_fine(train_records, fine_adapter_dir, args.train_epochs, args.train_lr, args.grad_accum)
        del llm
        clear_memory()

    if args.skip_train and not adapter_is_complete(fine_adapter_dir, "qformer"):
        raise RuntimeError(
            f"--skip-train requested but adapter is incomplete: {fine_adapter_dir}"
        )

    print(f"[{family}] evaluating fine")
    usable_adapter = fine_adapter_dir if adapter_is_complete(fine_adapter_dir, "qformer") else None
    llm = VariantLLM(family, adapter=usable_adapter)
    llm.load_img_proj_if_present(fine_adapter_dir)
    metrics.append(evaluate_variant(family, "fine", llm, test_records, eval_dir, args.max_new_tokens, "fine", context=context))
    del llm
    clear_memory()

    metrics.append(instruction_metrics(context, family, family_dir))

    df = pd.DataFrame(metrics)
    csv_path = family_dir / f"figure9_{family}_three_llm_types_200_metrics.csv"
    json_path = family_dir / f"figure9_{family}_three_llm_types_200_metrics.json"
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plot_outputs = plot_family(family, df, family_dir)

    summary = {
        "family": family,
        "sample_limit": args.sample_limit,
        "train_samples": len(train_records),
        "test_samples": len(test_records),
        "fine_adapter": str(fine_adapter_dir),
        "metrics_csv": str(csv_path),
        "metrics_json": str(json_path),
        **plot_outputs,
    }
    summary_path = family_dir / f"figure9_{family}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.gcs_output and not args.no_upload:
        safe_outputs = [csv_path, json_path, summary_path]
        safe_outputs.extend(Path(path) for path in plot_outputs.values())
        safe_outputs.extend(eval_dir.glob("metrics_*.json"))
        for path in safe_outputs:
            upload_path(path, f"{args.gcs_output}/{family}")
    return summary


def main() -> None:
    args = parse_args()
    set_seed(SEED)
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    checkpoint_root = Path(args.checkpoint_root)
    context = Stage1Context(
        run_name=DEFAULT_RUN_NAME,
        thresholds=load_thresholds(getattr(args, "threshold_path", None)),
    )

    train_records = build_stage1_records(context, checkpoint_root, root, "train", args.sample_limit, args.num_workers)
    test_records = build_stage1_records(context, checkpoint_root, root, "test", args.sample_limit, args.num_workers)

    summaries = {}
    for family in args.models:
        summaries[family] = run_family(context, args, family, train_records, test_records, root)

    summary_path = root / "figure9_llm_variants_200_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    if args.gcs_output and not args.no_upload:
        upload_path(summary_path, args.gcs_output)
        upload_path(Path(__file__), f"{args.gcs_output}/scripts")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
