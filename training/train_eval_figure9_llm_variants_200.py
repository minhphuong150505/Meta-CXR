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
import os
import random
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

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
from bert_score import score as bert_score_fn
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from peft import LoraConfig, PeftModel, get_peft_model
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "model"))

import model.lavis.tasks as tasks  # noqa: E402
from local_config import VIS_ROOT  # noqa: E402
from model.lavis.common.config import Config  # noqa: E402
from model.lavis.common.registry import registry  # noqa: E402
from model.lavis.data.ReportDataset import MIMIC_CXR_Dataset  # noqa: E402

registry.mapping["paths"]["cache_root"] = "."

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)

SEED = 16
RUN_NAME = "07_all_three"
NUM_IMG_TOKENS = 32
BERTSCORE_MODEL = "microsoft/deberta-xlarge-mnli"
SMOOTH = SmoothingFunction().method1

VICUNA_MODEL_ID = "lmsys/vicuna-7b-v1.3"
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

with open(PROJECT_DIR / "threshold.json") as f:
    THRESHOLDS = json.load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["vicuna", "medgemma"], choices=["vicuna", "medgemma"])
    parser.add_argument("--checkpoint-root", default="/mnt/meta-cxr-checkpoint")
    parser.add_argument("--output-dir", default="/home/phuong/META-CXR/output/figure9_llm_variants_200")
    parser.add_argument("--gcs-output", default="gs://meta-cxr-checkpoint/eval/paper_figures/figure9_llm_variants_200")
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


def build_cfg(run_name: str) -> Config:
    cfg_path = PROJECT_DIR / "pretraining/configs/encoder_comparison" / f"{run_name}.yaml"
    return Config(SimpleNamespace(cfg_path=str(cfg_path), options=None))


def load_torch_checkpoint(path: Path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def filter_state_dict_for_model(model, state_dict: dict) -> dict:
    model_state = model.state_dict()
    filtered = {}
    for key, value in state_dict.items():
        if key in model_state and hasattr(value, "shape") and tuple(value.shape) != tuple(model_state[key].shape):
            continue
        filtered[key] = value
    return filtered


def build_stage1_model(checkpoint_root: Path, device: torch.device):
    cfg = build_cfg(RUN_NAME)
    task = tasks.setup_task(cfg)
    model = task.build_model(cfg)
    ckpt_path = checkpoint_root / RUN_NAME / "checkpoint_best.pth"
    ckpt = load_torch_checkpoint(ckpt_path)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    state_dict = filter_state_dict_for_model(model, state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[stage1] loaded {ckpt_path}; missing={len(missing)} unexpected={len(unexpected)}")
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return cfg, model


def make_stage1_loader(cfg: Config, split: str, sample_limit: int, num_workers: int) -> DataLoader:
    dataset = MIMIC_CXR_Dataset(
        vis_processor=None,
        text_processor=None,
        vis_root=VIS_ROOT,
        split=split,
        cfg=cfg,
        truncate=sample_limit,
    )
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def field_value(field, index: int = 0) -> str:
    if isinstance(field, (list, tuple)):
        return str(field[index])
    try:
        return str(field[index])
    except Exception:
        return str(field)


def classify_with_thresholds(logits: torch.Tensor) -> dict[str, list[str]]:
    probs = torch.softmax(logits, dim=-1).tolist()
    out = {"positive": [], "negative": [], "uncertain": []}
    for abn, p in zip(ABNORMALITIES_14, probs):
        if abn == "No Finding":
            continue
        best_cls, best_score = None, 0.0
        for cls_name, cls_idx in CLASS_MAP.items():
            threshold = THRESHOLDS.get(abn, {}).get(cls_name, 0.5)
            if p[cls_idx] >= threshold and p[cls_idx] > best_score:
                best_cls = cls_name
                best_score = p[cls_idx]
        if best_cls:
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


def build_prompt(groups: dict[str, list[str]], img_token: str, prompt_style: str) -> str:
    findings = format_findings(groups)
    if prompt_style == "fine":
        return (
            f"Image information: {image_block(img_token)}.\n\n"
            f"Abnormality information: {findings}\n\n"
            "Write the Findings section of a chest X-ray report. Return only the findings text."
        )
    return (
        "A chat between a curious user and an artificial intelligence assistant."
        "The assistant gives professional, detailed, and polite answers to the user's questions. "
        f"USER: Image information: {image_block(img_token)}.\n\n"
        f"Abnormality information: {findings}\n\n"
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


@torch.no_grad()
def build_stage1_records(
    checkpoint_root: Path,
    output_dir: Path,
    split: str,
    sample_limit: int,
    num_workers: int,
) -> list[dict]:
    cache_path = output_dir / "stage1_cache" / f"{RUN_NAME}_{split}_{sample_limit}.pt"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        print(f"[stage1] reusing {cache_path}")
        return torch.load(cache_path, map_location="cpu")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, model = build_stage1_model(checkpoint_root, device)
    loader = make_stage1_loader(cfg, split, sample_limit, num_workers)
    records = []
    for batch in tqdm(loader, desc=f"stage1 {split}"):
        image = batch["image"].to(device, non_blocking=True)
        logits, qformer = model.forward_image(image)
        groups = classify_with_thresholds(logits[0].detach().cpu())
        records.append(
            {
                "index": len(records),
                "dicom_id": field_value(batch.get("dicom_id", "")),
                "ref": field_value(batch["text_output"]),
                "pred_groups": groups,
                "qformer_embs": qformer[0].detach().cpu().to(torch.float16),
            }
        )
    torch.save(records, cache_path)
    print(f"[stage1] wrote {cache_path} ({len(records)} records)")
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


class SoftTokenEmbeddingWrapper(nn.Module):
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
        mask = input_ids == self.img_token_id
        if not mask.any():
            return embeds
        embeds = embeds.clone()
        for batch_idx in range(input_ids.shape[0]):
            positions = mask[batch_idx].nonzero(as_tuple=False).flatten()
            if len(positions) != NUM_IMG_TOKENS:
                raise RuntimeError(f"expected {NUM_IMG_TOKENS} image tokens, got {len(positions)}")
            img = self.projected_img_embs[min(batch_idx, self.projected_img_embs.shape[0] - 1)]
            embeds[batch_idx, positions, :] = img.to(device=embeds.device, dtype=embeds.dtype)
        return embeds


class VariantLLM:
    def __init__(self, family: str, adapter: str | Path | None = None, train_adapter: bool = False):
        self.family = family
        self.adapter = adapter
        self.train_adapter = train_adapter
        self.dtype = preferred_dtype()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if family == "vicuna":
            self.model_id = VICUNA_MODEL_ID
            self.img_token = "<IMG>"
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                use_fast=False,
                truncation_side="left",
                padding_side="left",
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
            self.img_token = "<image_soft_token>"
            self.processor = AutoProcessor.from_pretrained(self.model_id, **hf_kwargs())
            self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            try:
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.model_id,
                    torch_dtype=self.dtype,
                    low_cpu_mem_usage=True,
                    **hf_kwargs(),
                )
            except Exception:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    torch_dtype=self.dtype,
                    low_cpu_mem_usage=True,
                    **hf_kwargs(),
                )
            if self.img_token not in self.tokenizer.get_vocab():
                self.tokenizer.add_special_tokens({"additional_special_tokens": [self.img_token]})
                self.model.resize_token_embeddings(len(self.tokenizer))

        self.model.to(self.device)
        self.img_token_id = self.tokenizer.convert_tokens_to_ids(self.img_token)
        if self.img_token_id is None or self.img_token_id < 0:
            raise RuntimeError(f"could not register image token {self.img_token}")

        hidden = int(self.model.get_input_embeddings().weight.shape[-1])
        self.img_proj = nn.Linear(768, hidden).to(self.device, dtype=self.dtype)
        if adapter:
            self.model = PeftModel.from_pretrained(self.model, str(adapter), is_trainable=train_adapter)
        elif train_adapter:
            targets = ["q_proj", "v_proj"] if family == "vicuna" else "all-linear"
            cfg = LoraConfig(
                r=8,
                lora_alpha=16,
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

        self.model.to(self.device)

    def save_adapter(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(out_dir)
        torch.save(self.img_proj.state_dict(), out_dir / "img_proj.pt")
        meta = {
            "family": self.family,
            "model_id": self.model_id,
            "img_token": self.img_token,
            "img_token_id": self.img_token_id,
            "num_img_tokens": NUM_IMG_TOKENS,
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def load_img_proj_if_present(self, adapter_dir: Path | str | None) -> None:
        if not adapter_dir:
            return
        p = Path(adapter_dir) / "img_proj.pt"
        if p.exists():
            self.img_proj.load_state_dict(torch.load(p, map_location=self.device))

    def encode_train_example(self, record: dict, prompt_style: str, max_length: int = 768):
        prompt = build_prompt(record["pred_groups"], self.img_token, prompt_style)
        target = str(record["ref"]).strip()
        full = prompt + "\n" + target
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        encoded = self.tokenizer(
            full,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )
        labels = encoded["input_ids"].clone()
        prompt_len = min(len(prompt_ids), labels.shape[1])
        labels[:, :prompt_len] = -100
        return encoded["input_ids"].to(self.device), encoded["attention_mask"].to(self.device), labels.to(self.device)

    def train_fine(self, records: list[dict], out_dir: Path, epochs: int, lr: float, grad_accum: int) -> None:
        self.model.train()
        self.img_proj.train()
        params = [p for p in self.model.parameters() if p.requires_grad] + list(self.img_proj.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        step = 0
        opt.zero_grad(set_to_none=True)
        for epoch in range(epochs):
            random.shuffle(records)
            progress = tqdm(records, desc=f"{self.family} fine train epoch {epoch + 1}/{epochs}")
            for record in progress:
                input_ids, attention_mask, labels = self.encode_train_example(record, "fine")
                qformer = record["qformer_embs"].unsqueeze(0).to(self.device, dtype=self.dtype)
                projected = self.img_proj(qformer)
                old_embedding = self.model.get_input_embeddings()
                self.model.set_input_embeddings(
                    SoftTokenEmbeddingWrapper(old_embedding, self.img_token_id, projected)
                )
                try:
                    out = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = out.loss / grad_accum
                    loss.backward()
                finally:
                    self.model.set_input_embeddings(old_embedding)
                if (step + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                step += 1
                progress.set_postfix(loss=f"{float(loss.detach().cpu()) * grad_accum:.4f}")
        if step % grad_accum:
            opt.step()
            opt.zero_grad(set_to_none=True)
        self.save_adapter(out_dir)

    @torch.no_grad()
    def generate(self, record: dict, prompt_style: str, max_new_tokens: int) -> str:
        self.model.eval()
        self.img_proj.eval()
        prompt = build_prompt(record["pred_groups"], self.img_token, prompt_style)
        encoded = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768)
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        qformer = record["qformer_embs"].unsqueeze(0).to(self.device, dtype=self.dtype)
        projected = self.img_proj(qformer)
        old_embedding = self.model.get_input_embeddings()
        self.model.set_input_embeddings(SoftTokenEmbeddingWrapper(old_embedding, self.img_token_id, projected))
        try:
            seq = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=False,
            )[0]
        finally:
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
    bleu_scores = {1: [], 2: [], 3: [], 4: []}
    meteor_vals = []
    for pred, ref in zip(preds, refs):
        pred_tok = tokenize(pred)
        ref_tok = [tokenize(ref)]
        if not pred_tok:
            for n in bleu_scores:
                bleu_scores[n].append(0.0)
            meteor_vals.append(0.0)
            continue
        for n in range(1, 5):
            weights = tuple(1.0 / n if i < n else 0.0 for i in range(4))
            bleu_scores[n].append(sentence_bleu(ref_tok, pred_tok, weights=weights, smoothing_function=SMOOTH))
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
    return {
        "BLEU-1": round(float(np.mean(bleu_scores[1])), 4),
        "BLEU-2": round(float(np.mean(bleu_scores[2])), 4),
        "BLEU-3": round(float(np.mean(bleu_scores[3])), 4),
        "BLEU-4": round(float(np.mean(bleu_scores[4])), 4),
        "METEOR": round(float(np.mean(meteor_vals)), 4),
        "ROUGE-L": round(rouge_l, 4),
        "CIDEr": round(cider, 4),
        "BERTScore": round(float(f1.mean().item()), 4),
    }


def evaluate_variant(
    family: str,
    variant: str,
    llm: VariantLLM,
    records: list[dict],
    out_dir: Path,
    max_new_tokens: int,
    prompt_style: str,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"reports_{family}_{variant}_{RUN_NAME}.jsonl"
    metrics_path = out_dir / f"metrics_{family}_{variant}_{RUN_NAME}.json"
    if metrics_path.exists() and jsonl_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    preds, refs = [], []
    if jsonl_path.exists():
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                preds.append(str(item.get("pred", "")))
                refs.append(str(item.get("ref", "")))
        if len(preds) > len(records):
            preds, refs = [], []
        elif preds:
            print(f"[{family} {variant}] resuming eval from {len(preds)}/{len(records)} records")
    mode = "a" if preds else "w"
    with open(jsonl_path, mode, encoding="utf-8") as f:
        for record in tqdm(records[len(preds) :], desc=f"{family} {variant} eval"):
            try:
                pred = llm.generate(record, prompt_style, max_new_tokens)
            except Exception as exc:
                print(f"generate failed at {record.get('index')}: {exc}")
                pred = ""
            ref = str(record["ref"])
            preds.append(pred)
            refs.append(ref)
            f.write(json.dumps({"pred": pred, "ref": ref, "index": record.get("index"), "dicom_id": record.get("dicom_id")}) + "\n")
    metrics = compute_nlg(preds, refs)
    metrics.update({"Family": family, "Variant": variant, "Run": RUN_NAME, "N": len(records), "MaxNewTokens": max_new_tokens})
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def instruction_metrics(family: str, work_dir: Path) -> dict:
    table = read_gcs_csv(INSTRUCTION_TABLES[family], work_dir)
    row = table[table["Run"] == RUN_NAME].iloc[0]
    return {
        "Family": family,
        "Variant": "instruction",
        "Run": RUN_NAME,
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


def run_family(args: argparse.Namespace, family: str, train_records: list[dict], test_records: list[dict], root: Path) -> dict:
    family_dir = root / family
    adapters_dir = family_dir / "adapters"
    eval_dir = family_dir / "eval"
    family_dir.mkdir(parents=True, exist_ok=True)
    fine_adapter_dir = adapters_dir / f"{family}_fine_lora_200"

    metrics = []

    if (not args.skip_existing_eval) or not (eval_dir / f"metrics_{family}_base_{RUN_NAME}.json").exists():
        print(f"[{family}] evaluating base")
        llm = VariantLLM(family)
        metrics.append(evaluate_variant(family, "base", llm, test_records, eval_dir, args.max_new_tokens, "fine"))
        del llm
        clear_memory()
    else:
        metrics.append(json.loads((eval_dir / f"metrics_{family}_base_{RUN_NAME}.json").read_text(encoding="utf-8")))

    if not args.skip_train and (args.force or not (fine_adapter_dir / "adapter_config.json").exists()):
        print(f"[{family}] training fine LoRA -> {fine_adapter_dir}")
        llm = VariantLLM(family, train_adapter=True)
        llm.train_fine(train_records, fine_adapter_dir, args.train_epochs, args.train_lr, args.grad_accum)
        del llm
        clear_memory()

    print(f"[{family}] evaluating fine")
    llm = VariantLLM(family, adapter=fine_adapter_dir if fine_adapter_dir.exists() else None)
    llm.load_img_proj_if_present(fine_adapter_dir)
    metrics.append(evaluate_variant(family, "fine", llm, test_records, eval_dir, args.max_new_tokens, "fine"))
    del llm
    clear_memory()

    metrics.append(instruction_metrics(family, family_dir))

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
    if not args.no_upload:
        upload_dir(family_dir, args.gcs_output)
    return summary


def main() -> None:
    args = parse_args()
    set_seed(SEED)
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    checkpoint_root = Path(args.checkpoint_root)

    train_records = build_stage1_records(checkpoint_root, root, "train", args.sample_limit, args.num_workers)
    test_records = build_stage1_records(checkpoint_root, root, "test", args.sample_limit, args.num_workers)

    summaries = {}
    for family in args.models:
        summaries[family] = run_family(args, family, train_records, test_records, root)

    summary_path = root / "figure9_llm_variants_200_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    if not args.no_upload:
        upload_path(summary_path, args.gcs_output)
        upload_path(Path(__file__), f"{args.gcs_output}/scripts")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
