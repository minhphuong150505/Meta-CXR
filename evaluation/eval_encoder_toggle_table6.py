#!/usr/bin/env python3
"""Table 6: report-generation BERTScore under encoder toggling from 07_all_three.

Loads the single ``07_all_three`` checkpoint (same one Table 5 uses), then for
each encoder configuration masks the image streams that feed BOTH the MHCAC
findings and the Q-Former generation tokens -- only the selected encoder's
features pass through to the generation head, while MHCAC / soft-prompt tokens /
LLM weights stay unchanged (paper's stated Table 6 methodology). It generates the
Findings section with Vicuna-7B + LoRA and scores against references with
BERTScore. The test set is subsampled (default 300) to fit one Kaggle session.

This mirrors evaluation/eval_encoder_toggle_07.py (Table 5, classification) for
the toggle/checkpoint logic, and evaluation/eval_bertscore_vicuna.py for the
Vicuna generation + BERTScore logic. No model code is modified: the per-stream
Q-Former projection of _encode_image_streams is replicated here with masking.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
from transformers import LlamaTokenizer
from peft import PeftModelForCausalLM

import model.lavis.tasks as tasks
from model.lavis.common.config import Config
from model.lavis.common.registry import registry

# Registration imports required by the LAVIS registry.
from model.lavis.common.optims import LinearWarmupCosineLRScheduler, LinearWarmupStepLRScheduler  # noqa: F401
from model.lavis.datasets.builders import *  # noqa: F401,F403
from model.lavis.models import *  # noqa: F401,F403
from model.lavis.processors import *  # noqa: F401,F403
from model.lavis.tasks import *  # noqa: F401,F403
from model.lavis.data.ReportDataset import MIMIC_CXR_Dataset
from model.lavis.models.blip2_models.modeling_llama_imgemb import LlamaForCausalLM  # noqa: F401
from local_config import VIS_ROOT

registry.mapping["paths"]["cache_root"] = "."

# --- Vicuna / generation constants (from eval_bertscore_vicuna.py) -----------
VICUNA_MODEL_ID = "lmsys/vicuna-7b-v1.3"
NUM_IMG_TOKENS = 32
IMG_TOKEN = "<IMG>"
IMG_TOKEN_BLOCK = IMG_TOKEN * NUM_IMG_TOKENS
MAX_NEW_TOKENS = 300
NUM_BEAMS = 1
SEED = 16

ABNORMALITIES_14 = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices",
]
CLASS_MAP = {"negative": 0, "positive": 1, "uncertain": 2}

# Encoder configurations (paper Table 6). pubmedclip_swin has no paper value,
# matching Table 5 (eval_encoder_toggle_07.py).
TOGGLE_RUNS = [
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

# raddino is disabled for 07_all_three, so its newly-initialized alignment head
# is never used at inference and is legitimately absent from the checkpoint.
ALLOWED_MISSING_PREFIXES = (
    "visual_encoder.",
    "pubmedclip.model.",
    "swin.model.",
    "mhcac.embedding_alignment.raddino_",
)

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

BERTSCORE_MODEL = "microsoft/deberta-xlarge-mnli"

_LLM_CACHE: dict = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument(
        "--cfg",
        type=Path,
        default=Path("pretraining/configs/encoder_comparison/07_all_three.yaml"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/kaggle/temp/checkpoints/07_all_three/checkpoint_best.pth"),
    )
    parser.add_argument("--limit", type=int, default=300, help="Subsample N test samples.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--bertscore-device",
        default="cpu",
        help="Device for BERTScore (cpu keeps GPU memory for Vicuna).",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("output/encoder_toggle_table6"))
    parser.add_argument("--allow-frozen-missing", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def seed_everything(device: str) -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if device == "cuda":
        torch.cuda.manual_seed_all(SEED)


def load_torch_checkpoint(path: Path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def build_cfg(cfg_path: Path) -> Config:
    args = SimpleNamespace(cfg_path=str(cfg_path), options=None)
    return Config(args)


def validate_load_result(missing, unexpected, allow_frozen_missing) -> None:
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected[:20]}")
    if allow_frozen_missing:
        invalid = [k for k in missing if not k.startswith(ALLOWED_MISSING_PREFIXES)]
        if invalid:
            raise RuntimeError(f"Checkpoint missing non-frozen/non-backbone keys: {invalid[:20]}")
    elif missing:
        raise RuntimeError(f"Checkpoint is missing keys: {missing[:20]}")


def build_model(cfg: Config, checkpoint_path: Path, device: str, allow_frozen_missing: bool):
    task = tasks.setup_task(cfg)
    model = task.build_model(cfg)
    ckpt = load_torch_checkpoint(checkpoint_path)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded {checkpoint_path}; missing={len(missing)}, unexpected={len(unexpected)}")
    validate_load_result(list(missing), list(unexpected), allow_frozen_missing)
    model.to(device)
    # Pubmedclip hardcodes self.device='cuda' (=cuda:0) at init and uses it to
    # place its inputs in forward; realign with the actual device so it works
    # when the model is placed on cuda:1.
    if getattr(model, "pubmedclip", None) is not None:
        model.pubmedclip.device = device
    model.eval()
    return model


def make_test_loader(cfg: Config, batch_size: int, num_workers: int, limit: int | None, device: str):
    dataset = MIMIC_CXR_Dataset(
        vis_processor=None,
        text_processor=None,
        vis_root=VIS_ROOT,
        split="test",
        cfg=cfg,
        truncate=None,
    )
    if limit is not None:
        dataset = Subset(dataset, list(range(min(limit, len(dataset)))))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )


@torch.no_grad()
def encode_toggled(model, image, use_rn50: bool, use_vit: bool, use_swin: bool):
    """Replicate forward_image with per-encoder masking.

    Only the selected encoders' Q-Former projections are concatenated and fed to
    the Q-Former; the same selection masks the MHCAC streams. Returns toggled
    (classification_logits, qformer_embeds).
    """
    streams = []
    cnn_patches = vit_patches = swin_patches = None

    if model.use_biovil:
        cnn_raw = model.visual_encoder(image).projected_patch_embeddings.reshape(
            image.shape[0], -1, 1408
        )
        cnn_patches = model.ln_vision(cnn_raw)
        if use_rn50:
            streams.append(cnn_patches)

    if model.use_pubmedclip:
        vit_patches, pubmed_projection = model.pubmedclip(image, apply_aug=False)
        if use_vit:
            streams.append(pubmed_projection)

    if model.use_swin:
        swin_patches = model.swin(image)
        if use_swin:
            streams.append(model.swin_qformer_proj(swin_patches))

    if not streams:
        raise ValueError("No encoder stream selected for this configuration.")

    concat_image_embeds = torch.cat(streams, dim=1)

    cls_logits, _, _, _, _ = model.mhcac(
        cnn_patches=cnn_patches if use_rn50 else None,
        vit_patches=vit_patches if use_vit else None,
        swin_patches=swin_patches if use_swin else None,
        raddino_patches=None,
        text_embeddings=None,
        labels=None,
    )

    image_atts = torch.ones(concat_image_embeds.size()[:-1], dtype=torch.long, device=image.device)
    query_tokens = model.query_tokens.expand(concat_image_embeds.shape[0], -1, -1)
    query_output = model.Qformer.bert(
        query_embeds=query_tokens,
        encoder_hidden_states=concat_image_embeds,
        encoder_attention_mask=image_atts,
        return_dict=True,
    )
    return cls_logits.float().cpu(), query_output.last_hidden_state.float().cpu()


def classify_with_thresholds(logits, thresholds):
    assert logits.shape == (14, 3), f"expected (14, 3), got {tuple(logits.shape)}"
    probs = torch.softmax(logits, dim=-1).tolist()
    out = {"positive": [], "negative": [], "uncertain": []}
    for abn, p in zip(ABNORMALITIES_14, probs):
        if abn == "No Finding":
            continue
        thresholds_abn = thresholds.get(abn, {})
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
    findings = format_findings_dict(classifications)
    return PROMPT_TEMPLATE.format(img_block=IMG_TOKEN_BLOCK, findings=findings)


def get_vicuna(project_dir: Path):
    if "model" in _LLM_CACHE:
        return _LLM_CACHE["model"], _LLM_CACHE["tokenizer"]

    lora_path = project_dir / "checkpoints" / "lora-vicuna-7b-report-20250621"
    print(f">>> Loading {VICUNA_MODEL_ID} ...")
    tokenizer = LlamaTokenizer.from_pretrained(
        VICUNA_MODEL_ID, use_fast=False, truncation_side="left", padding_side="left"
    )
    # Load the whole LLM on a single GPU (cuda:0). Sharding across 2 GPUs with
    # device_map="auto" puts the image-embedding injection cat() on mixed
    # devices (cuda:0 vs cuda:1) and crashes; single-device matches the original
    # single-GPU (L4) run.
    base = LlamaForCausalLM.from_pretrained(
        VICUNA_MODEL_ID, torch_dtype=torch.float16, device_map={"": 0}
    )
    tokenizer.pad_token = tokenizer.unk_token
    base.base_model.img_proj_layer = nn.Linear(
        768, base.base_model.config.hidden_size
    ).to(base.base_model.device)
    tokenizer.add_special_tokens({"additional_special_tokens": [IMG_TOKEN]})

    print(f">>> Attaching LoRA from {lora_path} ...")
    llm = PeftModelForCausalLM.from_pretrained(
        base, str(lora_path), torch_dtype=torch.float16, use_ram_optimized_load=False
    ).half()
    llm.eval()

    _LLM_CACHE["model"] = llm
    _LLM_CACHE["tokenizer"] = tokenizer
    return llm, tokenizer


@torch.no_grad()
def generate_report(prompt, qformer_embs, llm, tokenizer):
    assert qformer_embs.dim() == 3 and qformer_embs.shape[1] == NUM_IMG_TOKENS, (
        f"qformer_embs must be (B, {NUM_IMG_TOKENS}, 768), got {tuple(qformer_embs.shape)}"
    )
    # modeling_llama_imgemb reads the image embeddings from this side-channel file.
    torch.save(qformer_embs, "current_chat_img.pt")
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
    return preds[0].split("ASSISTANT:")[-1].strip()


def compute_bertscore(predictions, references, device):
    from bert_score import score as bert_score_fn

    if not predictions:
        return float("nan")
    _P, _R, F1 = bert_score_fn(
        predictions,
        references,
        lang="en",
        model_type=BERTSCORE_MODEL,
        rescale_with_baseline=False,
        verbose=False,
        device=device,
    )
    return float(F1.mean().item())


def main() -> None:
    args = parse_args()
    project_dir = args.project_dir.resolve()
    cfg_path = args.cfg if args.cfg.is_absolute() else project_dir / args.cfg
    checkpoint_path = args.checkpoint if args.checkpoint.is_absolute() else project_dir / args.checkpoint
    out_dir = args.out_dir if args.out_dir.is_absolute() else project_dir / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(args.device)
    with open(project_dir / "threshold.json") as f:
        thresholds = json.load(f)

    print("project_dir =", project_dir)
    print("cfg_path    =", cfg_path)
    print("checkpoint  =", checkpoint_path)
    print("device      =", args.device, "| bertscore_device =", args.bertscore_device)
    print("limit       =", args.limit)

    # Vicuna is loaded whole on cuda:0; on 2+ GPUs put the meta-cxr encoder/Q-Former
    # model on cuda:1 to avoid co-location OOM. Its outputs are moved to CPU in
    # encode_toggled, so the LLM device is independent.
    meta_device = args.device
    if args.device == "cuda" and torch.cuda.device_count() >= 2:
        meta_device = "cuda:1"
    print("meta_device =", meta_device, "| vicuna on cuda:0")

    cfg = build_cfg(cfg_path)
    model = build_model(cfg, checkpoint_path, meta_device, args.allow_frozen_missing)
    loader = make_test_loader(cfg, args.batch_size, args.num_workers, args.limit, meta_device)
    n_samples = len(loader.dataset)
    print("test samples =", n_samples)

    llm, tokenizer = get_vicuna(project_dir)

    rows = []
    details = {}
    for item in TOGGLE_RUNS:
        run_name = item["run"]
        print(f"\n{'='*60}\n{run_name}\n{'='*60}")
        predictions, references = [], []
        for batch in tqdm(loader, desc=f"{run_name} gen"):
            image = batch["image"].to(meta_device, non_blocking=True)
            cls_logits, qformer_embs = encode_toggled(
                model, image, item["RN50"], item["ViT"], item["Swin"]
            )
            for i in range(cls_logits.shape[0]):
                classifications = classify_with_thresholds(cls_logits[i], thresholds)
                prompt = build_prompt(classifications)
                try:
                    pred = generate_report(prompt, qformer_embs[i : i + 1], llm, tokenizer)
                except Exception as exc:  # noqa: BLE001 - keep going, record empty pred
                    print(f"  generate failed for sample {len(predictions)}: {exc}")
                    pred = ""
                ref_field = batch["text_output"]
                ref = ref_field[i] if isinstance(ref_field, (list, tuple)) else str(ref_field[i])
                predictions.append(pred)
                references.append(ref)

        mean_f1 = compute_bertscore(predictions, references, args.bertscore_device)
        paper = PAPER_BERTSCORE[run_name]
        rows.append(
            {
                "Run": run_name,
                "RN50": "yes" if item["RN50"] else "no",
                "ViT": "yes" if item["ViT"] else "no",
                "Swin": "yes" if item["Swin"] else "no",
                "BERTScore": round(mean_f1, 4),
                "N_samples": len(predictions),
                "Paper BERTScore": paper if paper is not None else None,
                "Delta vs Paper": round(mean_f1 - paper, 4) if paper is not None else None,
            }
        )
        details[run_name] = {"bertscore_f1": mean_f1, "n_samples": len(predictions)}

        out_jsonl = out_dir / f"reports_{run_name}.jsonl"
        with out_jsonl.open("w", encoding="utf-8") as f:
            for p, r in zip(predictions, references):
                f.write(json.dumps({"pred": p, "ref": r}) + "\n")
        print(f">>> {run_name}: BERTScore F1={mean_f1:.4f} ({len(predictions)} samples) -> {out_jsonl}")

    table = pd.DataFrame(rows)
    table_path = out_dir / "table6_bertscore_table.csv"
    json_path = out_dir / "table6_bertscore_details.json"
    table.to_csv(table_path, index=False)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": str(checkpoint_path),
                "cfg": str(cfg_path),
                "limit": args.limit,
                "num_samples": n_samples,
                "llm": VICUNA_MODEL_ID,
                "bertscore_model": BERTSCORE_MODEL,
                "max_new_tokens": MAX_NEW_TOKENS,
                "num_beams": NUM_BEAMS,
                "details": details,
            },
            f,
            indent=2,
        )

    print("\nTable 6:")
    print(table.to_string(index=False))
    print("\nWrote:", table_path)
    print("Wrote:", json_path)

    del model, loader
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
