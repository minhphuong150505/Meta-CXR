#!/usr/bin/env python3
"""BERTScore evaluation for Vicuna-7B + LoRA across all 7 encoder runs.

Follows 10_table6_bertscore_vicuna_per_combo_checkpoints.ipynb logic. Runs on GCP L4 VM with gcsfuse-mounted data.
Output: CSV table + per-run JSONL files, uploaded to gs://meta-cxr-checkpoint/eval/BERTSCore/Vicuna/
"""

import gc
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import LlamaTokenizer
from peft import PeftModelForCausalLM

PROJECT_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "model"))

import model.lavis.tasks as tasks
from model.lavis.common.config import Config
from model.lavis.common.registry import registry
from model.lavis.data.ReportDataset import MIMIC_CXR_Dataset
from model.lavis.models.blip2_models.modeling_llama_imgemb import LlamaForCausalLM
from local_config import VIS_ROOT

registry.mapping["paths"]["cache_root"] = "."

VICUNA_MODEL_ID = "lmsys/vicuna-7b-v1.3"
LORA_PATH = PROJECT_DIR / "checkpoints" / "lora-vicuna-7b-report-20250621"

NUM_IMG_TOKENS = 32
IMG_TOKEN = "<IMG>"
MAX_NEW_TOKENS = 300
NUM_BEAMS = 1
SEED = 16
EVAL_BATCH_SIZE = 1
NUM_WORKERS = 2
TEST_SAMPLE_LIMIT = 200

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
META_DEVICE = (
    "cuda:1"
    if DEVICE == "cuda" and torch.cuda.device_count() >= 2
    else DEVICE
)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(SEED)

ABNORMALITIES_14 = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices",
]
CLASS_MAP = {"negative": 0, "positive": 1, "uncertain": 2}

with open(PROJECT_DIR / "threshold.json") as f:
    THRESHOLDS = json.load(f)

CHECKPOINT_ROOT = Path("/mnt/meta-cxr-checkpoint")
OUTPUT_DIR = Path("/home/phuong/META-CXR/output/bertscore_vicuna")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GCS_OUTPUT = "gs://meta-cxr-checkpoint/eval/BERTSCore/Vicuna"

TABLE_RUNS = [
    {"run": "01_biovil_only",       "RN50": True,  "ViT": False, "Swin": False},
    {"run": "02_pubmedclip_only",   "RN50": False, "ViT": True,  "Swin": False},
    {"run": "03_swin_only",         "RN50": False, "ViT": False, "Swin": True},
    {"run": "04_biovil_pubmedclip", "RN50": True,  "ViT": True,  "Swin": False},
    {"run": "05_biovil_swin",       "RN50": True,  "ViT": False, "Swin": True},
    {"run": "06_pubmedclip_swin",   "RN50": False, "ViT": True,  "Swin": True},
    {"run": "07_all_three",         "RN50": True,  "ViT": True,  "Swin": True},
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


def build_cfg(run_name: str):
    cfg_path = PROJECT_DIR / "pretraining" / "configs" / "encoder_comparison" / f"{run_name}.yaml"
    args = SimpleNamespace(cfg_path=str(cfg_path), options=None)
    return Config(args)


def load_torch_checkpoint(path: Path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def build_meta_cxr_model(run_name: str, checkpoint_path: Path):
    cfg = build_cfg(run_name)
    task = tasks.setup_task(cfg)
    model = task.build_model(cfg)
    ckpt = load_torch_checkpoint(checkpoint_path)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"  {run_name}: loaded {checkpoint_path.name}; missing={len(missing)}, unexpected={len(unexpected)}")
    model.to(META_DEVICE)
    if getattr(model, "pubmedclip", None) is not None:
        model.pubmedclip.device = META_DEVICE
    model.eval()
    return cfg, model


def make_test_loader(cfg):
    dataset = MIMIC_CXR_Dataset(
        vis_processor=None,
        text_processor=None,
        vis_root=VIS_ROOT,
        split="test",
        cfg=cfg,
        truncate=TEST_SAMPLE_LIMIT,
    )
    return DataLoader(
        dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
    )


@torch.no_grad()
def forward_image_with_embs(model, batch):
    image = batch["image"].to(META_DEVICE, non_blocking=True)
    cls_logits, qformer_embs = model.forward_image(image)
    return cls_logits.float().cpu(), qformer_embs.float().cpu()


def classify_with_thresholds(logits):
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
            prefix = f"{label.capitalize()} findings"
            parts.append(f"{prefix}: {', '.join(items)}")
    return ". ".join(parts) if parts else "no common findings"


IMG_TOKEN_BLOCK = IMG_TOKEN * NUM_IMG_TOKENS

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


def build_prompt(classifications):
    findings = format_findings_dict(classifications)
    return PROMPT_TEMPLATE.format(img_block=IMG_TOKEN_BLOCK, findings=findings)


_LLM_CACHE = {}


def get_vicuna():
    if "model" in _LLM_CACHE:
        return _LLM_CACHE["model"], _LLM_CACHE["tokenizer"]

    print(f">>> Loading {VICUNA_MODEL_ID} ...")
    tokenizer = LlamaTokenizer.from_pretrained(
        VICUNA_MODEL_ID, use_fast=False, truncation_side="left", padding_side="left"
    )
    llm_dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    llm_kwargs = {"torch_dtype": llm_dtype}
    if DEVICE == "cuda":
        llm_kwargs["device_map"] = {"": 0}
    base = LlamaForCausalLM.from_pretrained(VICUNA_MODEL_ID, **llm_kwargs)
    tokenizer.pad_token = tokenizer.unk_token

    base.base_model.img_proj_layer = nn.Linear(768, base.base_model.config.hidden_size).to(
        base.base_model.device
    )
    tokenizer.add_special_tokens({"additional_special_tokens": [IMG_TOKEN]})

    print(f">>> Attaching LoRA from {LORA_PATH} ...")
    llm = PeftModelForCausalLM.from_pretrained(
        base, str(LORA_PATH), torch_dtype=llm_dtype, use_ram_optimized_load=False
    )
    if DEVICE == "cuda":
        llm = llm.half()
    llm.eval()

    _LLM_CACHE["model"] = llm
    _LLM_CACHE["tokenizer"] = tokenizer
    return llm, tokenizer


@torch.no_grad()
def generate_report(prompt, qformer_embs, llm, tokenizer):
    assert qformer_embs.dim() == 3 and qformer_embs.shape[1] == NUM_IMG_TOKENS, (
        f"qformer_embs must be (B, {NUM_IMG_TOKENS}, 768), got {tuple(qformer_embs.shape)}"
    )
    torch.save(qformer_embs.cpu(), "current_chat_img.pt")

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(llm.device)

    out = llm.generate(
        input_ids=input_ids,
        dicom=None,
        use_img=True,
        return_dict_in_generate=True,
        output_scores=False,
        max_new_tokens=MAX_NEW_TOKENS,
        num_beams=NUM_BEAMS,
        do_sample=False,
    )
    preds = tokenizer.batch_decode(out.sequences, skip_special_tokens=True)
    text = preds[0].split("ASSISTANT:")[-1].strip()
    return text


from bert_score import score as bert_score_fn

BERTSCORE_MODEL = "microsoft/deberta-xlarge-mnli"


def compute_bertscore(predictions, references):
    if not predictions:
        return float("nan")
    _P, _R, F1 = bert_score_fn(
        predictions,
        references,
        lang="en",
        model_type=BERTSCORE_MODEL,
        rescale_with_baseline=False,
        verbose=False,
        device="cpu",
    )
    return float(F1.mean().item())


def bertscore_for_run(run_name):
    ckpt_path = CHECKPOINT_ROOT / run_name / "checkpoint_best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    cfg, model = build_meta_cxr_model(run_name, ckpt_path)
    loader = make_test_loader(cfg)
    llm, tokenizer = get_vicuna()

    predictions, references = [], []

    for batch in tqdm(loader, desc=f"{run_name} infer"):
        cls_logits, qformer_embs = forward_image_with_embs(model, batch)
        for i in range(cls_logits.shape[0]):
            classifications = classify_with_thresholds(cls_logits[i])
            prompt = build_prompt(classifications)
            try:
                pred = generate_report(prompt, qformer_embs[i : i + 1], llm, tokenizer)
            except Exception as exc:
                print(f"  generate failed for sample {len(predictions)}: {exc}")
                pred = ""
            ref_field = batch["text_output"]
            ref = ref_field[i] if isinstance(ref_field, (list, tuple)) else str(ref_field[i])
            predictions.append(pred)
            references.append(ref)

    del model, loader
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    mean_f1 = compute_bertscore(predictions, references)

    out_jsonl = OUTPUT_DIR / f"reports_vicuna_{run_name}.jsonl"
    with open(out_jsonl, "w") as f:
        for p, r in zip(predictions, references):
            f.write(json.dumps({"pred": p, "ref": r}) + "\n")
    print(f">>> Wrote {out_jsonl} ({len(predictions)} samples, mean BERTScore F1={mean_f1:.4f})")

    return mean_f1, len(predictions)


def main():
    rows = []
    skipped = []

    for item in TABLE_RUNS:
        run_name = item["run"]
        paper_value = PAPER_BERTSCORE[run_name]
        print(f"\n{'='*60}")
        print(f"Processing {run_name}...")
        print(f"{'='*60}")
        try:
            bs, n = bertscore_for_run(run_name)
            rows.append({
                "Run": run_name,
                "RN50": "+" if item["RN50"] else "-",
                "ViT": "+" if item["ViT"] else "-",
                "Swin": "+" if item["Swin"] else "-",
                "BERTScore": round(bs, 4),
                "N_samples": n,
                "Paper_BERTScore": paper_value if paper_value is not None else "-",
                "Delta_vs_Paper": round(bs - paper_value, 4) if paper_value is not None else "-",
            })
        except FileNotFoundError as exc:
            print(f"WARNING: skipping {run_name}: {exc}")
            skipped.append(run_name)
            rows.append({
                "Run": run_name,
                "RN50": "+" if item["RN50"] else "-",
                "ViT": "+" if item["ViT"] else "-",
                "Swin": "+" if item["Swin"] else "-",
                "BERTScore": "MISSING",
                "N_samples": 0,
                "Paper_BERTScore": paper_value if paper_value is not None else "-",
                "Delta_vs_Paper": "-",
            })

    table = pd.DataFrame(rows)

    csv_path = OUTPUT_DIR / "encoder_bertscore_table_vicuna.csv"
    table.to_csv(csv_path, index=False)
    print(f"\nSaved CSV: {csv_path}")

    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "llm": VICUNA_MODEL_ID,
            "lora": str(LORA_PATH),
            "sample_limit": TEST_SAMPLE_LIMIT,
            "bertscore_model": BERTSCORE_MODEL,
            "runs": rows,
            "skipped": skipped,
        }, f, indent=2)
    print(f"Saved summary: {summary_path}")

    print(f"\nUploading to {GCS_OUTPUT}...")
    os.system(f"gsutil -m cp -r {OUTPUT_DIR}/* {GCS_OUTPUT}/")
    print(f"Uploaded to {GCS_OUTPUT}/")

    print("\n" + "=" * 60)
    print("BERTScore Results (Vicuna-7B + LoRA)")
    print("=" * 60)
    print(table.to_string(index=False))
    print(f"\nSkipped runs: {skipped or 'none'}")

    return table


if __name__ == "__main__":
    main()
