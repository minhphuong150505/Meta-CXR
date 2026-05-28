#!/usr/bin/env python3
"""Report generation eval for MedGemma + LoRA with Q-Former embedding injection.

This mirrors the Vicuna paper pipeline more closely than the text-only
MedGemma notebook:

1. Run the META-CXR stage-1 checkpoint to get classification logits and
   32 Q-Former image-query embeddings.
2. Build the same prompt pattern used for Vicuna, with 32 <IMG> tokens.
3. During MedGemma generation, temporarily wrap the token embedding layer and
   replace the embeddings at the <IMG> token positions with
   img_proj_layer(qformer_embs), where img_proj_layer maps 768 -> LLM hidden.
4. Save JSONL predictions and BERTScore summaries for the 7 encoder runs.

The MedGemma projection layer is randomly initialized for evaluation unless a
future MedGemma stage-2 checkpoint provides trained projection weights.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("DISABLE_TORCH_COMPILE", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from bert_score import score as bert_score_fn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor

PROJECT_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "model"))

import model.lavis.tasks as tasks
from local_config import VIS_ROOT
from model.lavis.common.config import Config
from model.lavis.common.registry import registry
from model.lavis.data.ReportDataset import MIMIC_CXR_Dataset

try:
    from peft import PeftModel
except ImportError:  # pragma: no cover - handled at runtime on VM.
    PeftModel = None


registry.mapping["paths"]["cache_root"] = "."

MEDGEMMA_MODEL_ID = "google/medgemma-1.5-4b-it"
MEDGEMMA_LORA_ID = "DeepRadiology/medgemma1.5-CXR"

NUM_IMG_TOKENS = 32
# MedGemma/Gemma tokenizer already contains this image placeholder token.
# Using it avoids resizing token embeddings after PEFT wraps embed_tokens.
IMG_TOKEN = "<image_soft_token>"
IMG_TOKEN_BLOCK = " ".join([IMG_TOKEN] * NUM_IMG_TOKENS)
MAX_NEW_TOKENS = 300
NUM_BEAMS = 1
SEED = 16
EVAL_BATCH_SIZE = 1
NUM_WORKERS = 2
TEST_SAMPLE_LIMIT = 200
BERTSCORE_MODEL = "microsoft/deberta-xlarge-mnli"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

with open(PROJECT_DIR / "threshold.json") as f:
    THRESHOLDS = json.load(f)

TABLE_RUNS = [
    {"run": "01_biovil_only", "RN50": True, "ViT": False, "Swin": False},
    {"run": "02_pubmedclip_only", "RN50": False, "ViT": True, "Swin": False},
    {"run": "03_swin_only", "RN50": False, "ViT": False, "Swin": True},
    {"run": "04_biovil_pubmedclip", "RN50": True, "ViT": True, "Swin": False},
    {"run": "05_biovil_swin", "RN50": True, "ViT": False, "Swin": True},
    {"run": "06_pubmedclip_swin", "RN50": False, "ViT": True, "Swin": True},
    {"run": "07_all_three", "RN50": True, "ViT": True, "Swin": True},
]

PAPER_BERTSCORE = {
    "01_biovil_only": 0.312,
    "02_pubmedclip_only": 0.289,
    "03_swin_only": 0.267,
    "04_biovil_pubmedclip": 0.401,
    "05_biovil_swin": 0.394,
    "06_pubmedclip_swin": None,
    "07_all_three": 0.426,
}

PROMPT_TEMPLATE = (
    "A chat between a curious user and an artificial intelligence assistant."
    "The assistant gives professional, detailed, and polite answers to the user's questions. "
    "USER: Image information: {img_block}.\n\n"
    "Abnormality information: {findings}\n\n"
    "Act as an expert radiologist. Using only the structured abnormality information and the image-derived features above, "
    "write the *Findings* section of a chest X-ray report.\n\n"
    "- Do not invent findings. Only describe abnormalities explicitly provided in the 'Abnormality information'.\n"
    "- Do not repeat the same information using different wording.\n"
    "- Use a single, fluent paragraph in formal radiological style.\n"
    "- Use cautious and precise language if uncertain abnormalities are present.\n"
    "- Avoid enumeration, bullet points, and speculative phrases.\n"
    "- The report should reflect the clinical tone and structure of professionally written reports.\n\n"
    "Return only the generated findings text. ASSISTANT:"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="*", default=[item["run"] for item in TABLE_RUNS])
    parser.add_argument("--sample-limit", type=int, default=TEST_SAMPLE_LIMIT)
    parser.add_argument("--checkpoint-root", default="/mnt/meta-cxr-checkpoint")
    parser.add_argument("--output-dir", default="/home/phuong/META-CXR/output/medgemma_qformer")
    parser.add_argument("--gcs-output", default="gs://meta-cxr-checkpoint/eval/MedGemma_QFormer")
    parser.add_argument("--medgemma-model-id", default=MEDGEMMA_MODEL_ID)
    parser.add_argument("--medgemma-lora-id", default=MEDGEMMA_LORA_ID)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--reuse-stage1-cache", action="store_true")
    parser.add_argument("--device-map-auto", action="store_true")
    return parser.parse_args()


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)


def clear_cuda_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_cfg(run_name: str):
    cfg_path = PROJECT_DIR / "pretraining" / "configs" / "encoder_comparison" / f"{run_name}.yaml"
    args = SimpleNamespace(cfg_path=str(cfg_path), options=None)
    return Config(args)


def load_torch_checkpoint(path: Path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def filter_state_dict_for_model(model, state_dict):
    model_state = model.state_dict()
    filtered = {}
    mismatched = []
    for key, value in state_dict.items():
        if key in model_state and hasattr(value, "shape"):
            ckpt_shape = tuple(value.shape)
            model_shape = tuple(model_state[key].shape)
            if ckpt_shape != model_shape:
                mismatched.append((key, ckpt_shape, model_shape))
                continue
        filtered[key] = value
    return filtered, mismatched


def build_meta_cxr_model(run_name: str, checkpoint_path: Path):
    cfg = build_cfg(run_name)
    task = tasks.setup_task(cfg)
    model = task.build_model(cfg)
    ckpt = load_torch_checkpoint(checkpoint_path)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    state_dict, mismatched = filter_state_dict_for_model(model, state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(
        f"{run_name}: loaded {checkpoint_path}; missing={len(missing)}, "
        f"unexpected={len(unexpected)}, mismatched_skipped={len(mismatched)}"
    )
    if mismatched:
        for key, ckpt_shape, model_shape in mismatched[:5]:
            print(f"  skipped shape mismatch: {key}: checkpoint={ckpt_shape}, model={model_shape}")
        if len(mismatched) > 5:
            print(f"  ... skipped {len(mismatched) - 5} more mismatched tensors")
    model.to(DEVICE)
    model.eval()
    return cfg, model


def make_test_loader(cfg, sample_limit: int | None, num_workers: int):
    dataset = MIMIC_CXR_Dataset(
        vis_processor=None,
        text_processor=None,
        vis_root=VIS_ROOT,
        split="test",
        cfg=cfg,
        truncate=sample_limit,
    )
    return DataLoader(
        dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(DEVICE == "cuda"),
    )


@torch.no_grad()
def forward_image_with_embs(model, batch):
    image = batch["image"].to(DEVICE, non_blocking=True)
    cls_logits, qformer_embs = model.forward_image(image)
    return cls_logits.float().cpu(), qformer_embs.detach().half().cpu()


def classify_with_thresholds(logits: torch.Tensor):
    assert logits.shape == (14, 3), f"expected (14, 3), got {tuple(logits.shape)}"
    probs = torch.softmax(logits, dim=-1).tolist()
    out = {"positive": [], "negative": [], "uncertain": []}
    for abn, p in zip(ABNORMALITIES_14, probs):
        if abn == "No Finding":
            continue
        thresholds_abn = THRESHOLDS.get(abn, {})
        best_cls, best_score = None, 0.0
        for cls_name, cls_idx in CLASS_MAP.items():
            threshold = thresholds_abn.get(cls_name, 0.5)
            prob = p[cls_idx]
            if prob >= threshold and prob > best_score:
                best_cls = cls_name
                best_score = prob
        if best_cls is not None:
            out[best_cls].append(abn)
    return out


def format_findings_dict(classifications):
    parts = []
    for label in ["positive", "negative", "uncertain"]:
        items = classifications[label]
        if items:
            parts.append(f"{label.capitalize()} findings: {', '.join(items)}")
    return ". ".join(parts) if parts else "no common findings"


def build_prompt(classifications):
    return PROMPT_TEMPLATE.format(
        img_block=IMG_TOKEN_BLOCK,
        findings=format_findings_dict(classifications),
    )


def _field_value(field, index: int):
    if isinstance(field, (list, tuple)):
        return field[index]
    try:
        return field[index]
    except Exception:
        return str(field)


def _batch_size(batch) -> int:
    image = batch.get("image")
    if image is not None:
        return int(image.shape[0])
    text = batch.get("text_output")
    return len(text) if isinstance(text, (list, tuple)) else 1


def extract_stage1_run(run_name: str, checkpoint_root: Path, output_dir: Path, sample_limit: int | None, num_workers: int, reuse_cache: bool):
    cache_dir = output_dir / "stage1_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{run_name}_stage1.pt"
    if reuse_cache and cache_path.exists():
        cached_records = torch.load(cache_path, map_location="cpu")
        expected_len = sample_limit
        has_current_placeholder = bool(cached_records) and IMG_TOKEN in cached_records[0].get("prompt", "")
        if (expected_len is None or len(cached_records) == expected_len) and has_current_placeholder:
            print(f">>> Reusing stage-1 cache: {cache_path} ({len(cached_records)} samples)")
            return cached_records
        reasons = []
        if expected_len is not None and len(cached_records) != expected_len:
            reasons.append(f"{len(cached_records)} samples != expected {expected_len}")
        if not has_current_placeholder:
            reasons.append(f"prompt does not contain {IMG_TOKEN}")
        print(f">>> Ignoring stage-1 cache ({'; '.join(reasons)}). Recomputing {run_name}.")

    ckpt_path = checkpoint_root / run_name / "checkpoint_best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    cfg, model = build_meta_cxr_model(run_name, ckpt_path)
    loader = make_test_loader(cfg, sample_limit, num_workers)
    records = []

    for batch in tqdm(loader, desc=f"{run_name} stage1"):
        try:
            cls_logits, qformer_embs = forward_image_with_embs(model, batch)
        except Exception as exc:
            print(f"  stage1 failed around sample {len(records)}: {exc}")
            for i in range(_batch_size(batch)):
                ref_field = batch.get("text_output", "")
                records.append(
                    {
                        "index": len(records),
                        "ref": str(_field_value(ref_field, i)),
                        "pred_groups": {"positive": [], "negative": [], "uncertain": []},
                        "prompt": build_prompt({"positive": [], "negative": [], "uncertain": []}),
                        "qformer_embs": torch.zeros(NUM_IMG_TOKENS, 768, dtype=torch.float16),
                        "stage1_error": str(exc),
                    }
                )
            continue

        for i in range(cls_logits.shape[0]):
            groups = classify_with_thresholds(cls_logits[i])
            ref_field = batch["text_output"]
            dicom_field = batch.get("dicom_id")
            records.append(
                {
                    "index": len(records),
                    "dicom_id": str(_field_value(dicom_field, i)) if dicom_field is not None else "",
                    "ref": str(_field_value(ref_field, i)),
                    "pred_groups": groups,
                    "prompt": build_prompt(groups),
                    "qformer_embs": qformer_embs[i].cpu(),
                }
            )

    torch.save(records, cache_path)
    print(f">>> Wrote stage-1 cache: {cache_path} ({len(records)} samples)")

    del model, loader
    clear_cuda_cache()
    return records


def hf_token():
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def hf_kwargs():
    token = hf_token()
    return {"token": token} if token else {}


def preferred_llm_dtype():
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def tokenizer_from_processor(processor):
    tokenizer = getattr(processor, "tokenizer", processor)
    if tokenizer is None:
        raise RuntimeError("Could not get tokenizer from AutoProcessor.")
    return tokenizer


def model_input_device(model):
    embedding = model.get_input_embeddings()
    for tensor in list(embedding.parameters(recurse=True)):
        if not getattr(tensor, "is_meta", False):
            return tensor.device
    hf_device_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_device_map, dict):
        for value in hf_device_map.values():
            if isinstance(value, int):
                return torch.device(f"cuda:{value}")
            if isinstance(value, str) and value not in {"cpu", "disk", "meta"}:
                return torch.device(value)
    return torch.device(DEVICE)


def load_base_medgemma(model_id: str, dtype: torch.dtype, device_map_auto: bool):
    kwargs = hf_kwargs()
    kwargs["torch_dtype"] = dtype
    if device_map_auto:
        kwargs["device_map"] = "auto"
    else:
        kwargs["low_cpu_mem_usage"] = True

    try:
        return AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    except Exception as image_exc:
        print(f"WARN: AutoModelForImageTextToText failed: {image_exc}")
        print(">>> Retrying with AutoModelForCausalLM ...")
        return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)


def activate_adapter(model, adapter_name):
    if adapter_name and hasattr(model, "set_adapter"):
        model.set_adapter(adapter_name)


def attach_lora(model, lora_id: str, dtype: torch.dtype):
    if not lora_id:
        return model
    print(f">>> Attaching LoRA: {lora_id}")
    if hasattr(model, "load_adapter"):
        try:
            adapter_name = model.load_adapter(lora_id, adapter_name="cxr", **hf_kwargs())
        except TypeError:
            adapter_name = model.load_adapter(lora_id, adapter_name="cxr")
        activate_adapter(model, adapter_name or "cxr")
        return model
    if PeftModel is None:
        raise ImportError("peft is required to attach MedGemma LoRA.")
    kwargs = hf_kwargs()
    kwargs["torch_dtype"] = dtype
    return PeftModel.from_pretrained(model, lora_id, **kwargs)


class QFormerEmbeddingWrapper(nn.Module):
    """Embedding wrapper that replaces <IMG> token embeddings for one prompt."""

    def __init__(self, base_embedding: nn.Module, img_token_id: int, projected_img_embs: torch.Tensor):
        super().__init__()
        self.base_embedding = base_embedding
        self.img_token_id = int(img_token_id)
        self.projected_img_embs = projected_img_embs

    @property
    def weight(self):
        return getattr(self.base_embedding, "weight", None)

    def forward(self, input_ids: torch.Tensor):
        embeds = self.base_embedding(input_ids)
        if input_ids is None:
            return embeds
        mask = input_ids == self.img_token_id
        if not mask.any():
            return embeds

        embeds = embeds.clone()
        for batch_idx in range(input_ids.shape[0]):
            positions = mask[batch_idx].nonzero(as_tuple=False).flatten()
            if len(positions) != NUM_IMG_TOKENS:
                raise RuntimeError(
                    f"Expected {NUM_IMG_TOKENS} {IMG_TOKEN} tokens, got {len(positions)}"
                )
            img_embs = self.projected_img_embs[min(batch_idx, self.projected_img_embs.shape[0] - 1)]
            embeds[batch_idx, positions, :] = img_embs.to(device=embeds.device, dtype=embeds.dtype)
        return embeds


class MedGemmaQFormerGenerator:
    def __init__(self, model_id: str, lora_id: str, device_map_auto: bool):
        self.model_id = model_id
        self.lora_id = lora_id
        self.dtype = preferred_llm_dtype()
        self.processor = AutoProcessor.from_pretrained(model_id, **hf_kwargs())
        self.tokenizer = tokenizer_from_processor(self.processor)

        print(f">>> Loading {model_id} (dtype={self.dtype}, device_map_auto={device_map_auto})")
        self.model = load_base_medgemma(model_id, self.dtype, device_map_auto=device_map_auto)
        self.model = attach_lora(self.model, lora_id, self.dtype)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        added = self.tokenizer.add_special_tokens({"additional_special_tokens": [IMG_TOKEN]})
        if added:
            print(f">>> Added special token {IMG_TOKEN}; resizing embeddings to {len(self.tokenizer)}")
            try:
                self.model.resize_token_embeddings(len(self.tokenizer), mean_resizing=False)
            except TypeError:
                self.model.resize_token_embeddings(len(self.tokenizer))

        if not device_map_auto and DEVICE == "cuda":
            self.model.to("cuda")
        self.model.eval()

        self.img_token_id = self.tokenizer.convert_tokens_to_ids(IMG_TOKEN)
        if self.img_token_id is None or self.img_token_id < 0:
            raise RuntimeError(f"Tokenizer did not register {IMG_TOKEN}.")

        input_device = model_input_device(self.model)
        probe = self.model.get_input_embeddings()(torch.tensor([[self.tokenizer.eos_token_id]], device=input_device))
        hidden_size = int(probe.shape[-1])
        self.img_proj_layer = nn.Linear(768, hidden_size).to(device=input_device, dtype=probe.dtype)
        self.img_proj_layer.eval()
        print(f">>> Q-Former projection: 768 -> {hidden_size}; input_device={input_device}")

        if getattr(self.model, "generation_config", None) is not None:
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

    def _chat_text(self, prompt: str) -> str:
        # Keep the image placeholder tokens in the final token stream. Some
        # chat templates treat image tokens structurally and remove literal
        # occurrences from text-only content, which prevents Q-Former injection.
        return prompt

    @torch.no_grad()
    def generate(self, prompt: str, qformer_embs: torch.Tensor, max_new_tokens: int) -> str:
        text = self._chat_text(prompt)
        encoded = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        input_device = model_input_device(self.model)
        input_ids = encoded["input_ids"].to(input_device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(input_device)

        img_count = int((input_ids == self.img_token_id).sum().item())
        if img_count != NUM_IMG_TOKENS:
            raise RuntimeError(
                f"Prompt tokenization produced {img_count} {IMG_TOKEN} tokens; expected {NUM_IMG_TOKENS}."
            )

        qformer_embs = qformer_embs.unsqueeze(0) if qformer_embs.dim() == 2 else qformer_embs
        qformer_embs = qformer_embs.to(input_device, dtype=self.img_proj_layer.weight.dtype)
        projected = self.img_proj_layer(qformer_embs)

        old_embedding = self.model.get_input_embeddings()
        wrapper = QFormerEmbeddingWrapper(old_embedding, self.img_token_id, projected)
        self.model.set_input_embeddings(wrapper)
        try:
            sequences = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                num_beams=NUM_BEAMS,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=False,
            )
        finally:
            self.model.set_input_embeddings(old_embedding)

        seq = sequences[0]
        gen_ids = seq[input_ids.shape[1] :] if seq.shape[0] > input_ids.shape[1] else seq
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        return clean_generated_text(text)


def clean_generated_text(text: str) -> str:
    text = text.strip()
    markers = ["ASSISTANT:", "Assistant:", "assistant:", "model\n", "model:"]
    for marker in markers:
        if marker in text:
            text = text.split(marker)[-1].strip()
    return text


def compute_bertscore(predictions, references):
    if not predictions:
        return float("nan")
    _p, _r, f1 = bert_score_fn(
        predictions,
        references,
        lang="en",
        model_type=BERTSCORE_MODEL,
        rescale_with_baseline=False,
        verbose=False,
        device="cpu",
    )
    return float(f1.mean().item())


def upload_dir(local_dir: Path, gcs_output: str):
    cmd = f"gcloud storage cp -r {local_dir}/* {gcs_output}/ --quiet"
    print(f">>> Uploading: {cmd}")
    rc = os.system(cmd)
    if rc != 0:
        raise RuntimeError(f"Upload failed with rc={rc}: {cmd}")


def run_generation_for_records(generator: MedGemmaQFormerGenerator, records: list[dict], run_name: str, output_dir: Path, max_new_tokens: int):
    predictions, references = [], []
    out_jsonl = output_dir / f"reports_medgemma_qformer_{run_name}.jsonl"
    with open(out_jsonl, "w") as f:
        for record in tqdm(records, desc=f"{run_name} MedGemma"):
            try:
                pred = generator.generate(record["prompt"], record["qformer_embs"], max_new_tokens)
            except Exception as exc:
                print(f"  generate failed for sample {record.get('index', len(predictions))}: {exc}")
                pred = ""
            ref = str(record["ref"])
            predictions.append(pred)
            references.append(ref)
            row = {
                "pred": pred,
                "ref": ref,
                "index": record.get("index"),
                "dicom_id": record.get("dicom_id", ""),
                "pred_groups": record.get("pred_groups", {}),
                "prompt": record.get("prompt", ""),
            }
            f.write(json.dumps(row) + "\n")

    mean_f1 = compute_bertscore(predictions, references)
    print(f">>> Wrote {out_jsonl} ({len(predictions)} samples, BERTScore F1={mean_f1:.4f})")
    return mean_f1, len(predictions)


def selected_run_items(run_names: Iterable[str]):
    by_name = {item["run"]: item for item in TABLE_RUNS}
    return [by_name[name] for name in run_names]


def main():
    args = parse_args()
    set_seed(SEED)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root = Path(args.checkpoint_root)

    run_items = selected_run_items(args.runs)
    sample_limit = args.sample_limit if args.sample_limit and args.sample_limit > 0 else None

    print(f"DEVICE={DEVICE}")
    print(f"MedGemma={args.medgemma_model_id}")
    print(f"LoRA={args.medgemma_lora_id or '(none)'}")
    print(f"sample_limit={sample_limit or 'full'}")
    print(f"checkpoint_root={checkpoint_root}")
    print(f"output_dir={output_dir}")

    stage1_by_run = {}
    skipped = []
    for item in run_items:
        run_name = item["run"]
        try:
            stage1_by_run[run_name] = extract_stage1_run(
                run_name,
                checkpoint_root,
                output_dir,
                sample_limit,
                args.num_workers,
                args.reuse_stage1_cache,
            )
        except FileNotFoundError as exc:
            print(f"WARNING: skipping {run_name}: {exc}")
            skipped.append(run_name)

    clear_cuda_cache()
    generator = MedGemmaQFormerGenerator(
        args.medgemma_model_id,
        args.medgemma_lora_id,
        device_map_auto=args.device_map_auto,
    )

    rows = []
    for item in run_items:
        run_name = item["run"]
        paper_value = PAPER_BERTSCORE[run_name]
        if run_name in skipped:
            rows.append(
                {
                    "Run": run_name,
                    "RN50": "+" if item["RN50"] else "-",
                    "ViT": "+" if item["ViT"] else "-",
                    "Swin": "+" if item["Swin"] else "-",
                    "BERTScore": "MISSING",
                    "N_samples": 0,
                    "Paper_BERTScore": paper_value if paper_value is not None else "-",
                    "Delta_vs_Paper": "-",
                }
            )
            continue

        bs, n = run_generation_for_records(
            generator,
            stage1_by_run[run_name],
            run_name,
            output_dir,
            args.max_new_tokens,
        )
        rows.append(
            {
                "Run": run_name,
                "RN50": "+" if item["RN50"] else "-",
                "ViT": "+" if item["ViT"] else "-",
                "Swin": "+" if item["Swin"] else "-",
                "BERTScore": round(bs, 4),
                "N_samples": n,
                "Paper_BERTScore": paper_value if paper_value is not None else "-",
                "Delta_vs_Paper": round(bs - paper_value, 4) if paper_value is not None else "-",
            }
        )

    table = pd.DataFrame(rows)
    csv_path = output_dir / "encoder_bertscore_table_medgemma_qformer.csv"
    table.to_csv(csv_path, index=False)
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "llm": args.medgemma_model_id,
                "lora": args.medgemma_lora_id,
                "image_conditioning": "Q-Former embeddings injected at 32 <IMG> token positions",
                "img_projection": "random Linear(768, llm_hidden_size), same evaluation-time pattern as existing Vicuna script",
                "sample_limit": sample_limit,
                "bertscore_model": BERTSCORE_MODEL,
                "runs": rows,
                "skipped": skipped,
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 80)
    print("BERTScore Results (MedGemma + LoRA + Q-Former injection)")
    print("=" * 80)
    print(table.to_string(index=False))

    if not args.skip_upload:
        upload_dir(output_dir, args.gcs_output)
        print(f">>> Uploaded to {args.gcs_output}/")


if __name__ == "__main__":
    main()
