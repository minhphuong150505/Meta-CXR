#!/usr/bin/env python3
"""Generate paper-ready Table 4-style metrics and Figure 6-style examples.

The available mounted data in this project is MIMIC-CXR p10, so the default
output is explicitly labelled as MIMIC-CXR test rather than CheXpert
cross-domain. If a CheXpert mount is added later, this script can be extended
with a CheXpert loader without changing the checkpoint/model path logic.
"""

from __future__ import annotations

import argparse
import gc
import html
import json
import os
import random
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from peft import PeftModelForCausalLM
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import LlamaTokenizer

PROJECT_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "model"))

import model.lavis.tasks as tasks  # noqa: E402
from local_config import VIS_ROOT  # noqa: E402
from model.lavis.common.config import Config  # noqa: E402
from model.lavis.common.registry import registry  # noqa: E402
from model.lavis.data.ReportDataset import MIMIC_CXR_Dataset  # noqa: E402
from model.lavis.models.blip2_models.modeling_llama_imgemb import LlamaForCausalLM  # noqa: E402

# Registration imports.
from model.lavis.common.optims import (  # noqa: E402,F401
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from model.lavis.datasets.builders import *  # noqa: E402,F403
from model.lavis.models import *  # noqa: E402,F403
from model.lavis.processors import *  # noqa: E402,F403
from model.lavis.tasks import *  # noqa: E402,F403

registry.mapping["paths"]["cache_root"] = "."

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

COMMON_5 = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]
COMMON_5_IDXS = [ABNORMALITIES_14.index(name) for name in COMMON_5]
CLASS_ID_TO_NAME = {0: "negative", 1: "positive", 2: "uncertain"}
CLASS_NAME_TO_ID = {v: k for k, v in CLASS_ID_TO_NAME.items()}

VICUNA_MODEL_ID = "lmsys/vicuna-7b-v1.3"
IMG_TOKEN = "<IMG>"
NUM_IMG_TOKENS = 32
IMG_TOKEN_BLOCK = IMG_TOKEN * NUM_IMG_TOKENS
PROMPT_TEMPLATE = (
    "A chat between a curious user and an artificial intelligence assistant."
    "The assistant gives professional, detailed, and polite answers to the user's questions. "
    "USER: Image information: {img_block}.\n\n"
    "Abnormality information: {findings}\n\n"
    "Act as an expert radiologist. Using only the structured abnormality information and the image-derived features above, "
    "write the Findings section of a chest X-ray report.\n\n"
    "- Do not invent findings. Only describe abnormalities explicitly provided in the 'Abnormality information'.\n"
    "- Do not repeat the same information using different wording.\n"
    "- Use a single, fluent paragraph in formal radiological style.\n"
    "- Use cautious and precise language if uncertain abnormalities are present.\n"
    "- Avoid enumeration, bullet points, and speculative phrases.\n"
    "- The report should reflect the clinical tone and structure of professionally written reports.\n\n"
    "Return only the generated findings text. ASSISTANT:"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper assets for META-CXR")
    parser.add_argument("--run-name", default="07_all_three")
    parser.add_argument(
        "--cfg-path",
        type=Path,
        default=PROJECT_DIR / "pretraining/configs/encoder_comparison/07_all_three.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/mnt/meta-cxr-checkpoint/07_all_three/checkpoint_best.pth"),
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--figure-samples", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--outputs",
        type=Path,
        default=PROJECT_DIR / "outputs" / "paper_outputs",
    )
    parser.add_argument(
        "--skip-figure",
        action="store_true",
        help="Only compute Table 4-style metrics; skip Vicuna report generation.",
    )
    parser.add_argument(
        "--reuse-table4",
        action="store_true",
        help="Reuse an existing table4_classification_metrics.csv instead of rerunning full-test inference.",
    )
    return parser.parse_args()


def load_thresholds() -> dict:
    path = PROJECT_DIR / "threshold.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def build_cfg(cfg_path: Path) -> Config:
    return Config(SimpleNamespace(cfg_path=str(cfg_path), options=None))


def build_model(cfg: Config, checkpoint_path: Path, device: str) -> torch.nn.Module:
    task = tasks.setup_task(cfg)
    model = task.build_model(cfg)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[load] {checkpoint_path} missing={len(missing)} unexpected={len(unexpected)}")
    model = model.to(device)
    if getattr(model, "pubmedclip", None) is not None:
        model.pubmedclip.device = device
    return model.eval()


def make_loader(cfg: Config, split: str, batch_size: int, num_workers: int, truncate=None) -> DataLoader:
    dataset = make_dataset(cfg, split, truncate=truncate)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def make_dataset(cfg: Config, split: str, truncate=None) -> MIMIC_CXR_Dataset:
    return MIMIC_CXR_Dataset(
        vis_processor=None,
        text_processor=None,
        vis_root=VIS_ROOT,
        split=split,
        cfg=cfg,
        truncate=truncate,
    )


def collect_labels_and_meta_from_dataset(dataset: MIMIC_CXR_Dataset):
    labels = dataset.annotation[dataset.chexpert_cols].values.astype(np.int64)
    meta = []
    for _, ann in dataset.annotation.iterrows():
        raw = str(ann["image_path"]).replace("\\", "/")
        marker = "/mimic-cxr-jpg-lite/"
        rel = raw.split(marker, 1)[-1] if marker in raw else raw.lstrip("/")
        meta.append(
            {
                "dicom_id": str(ann["dicom_id"]),
                "image_path": str(Path(VIS_ROOT) / rel),
                "text_output": str(ann["findings"]),
            }
        )
    return labels, meta


@torch.no_grad()
def forward_with_attention(model: torch.nn.Module, images: torch.Tensor):
    cnn_patches, vit_patches, swin_patches, concat_image_embeds = model._encode_image_streams(
        images, apply_aug=False
    )
    logits, attention, _contrastive, _orth, _sparsity = model.mhcac(
        cnn_patches=cnn_patches,
        vit_patches=vit_patches,
        swin_patches=swin_patches,
        text_embeddings=None,
        labels=None,
    )

    image_atts = torch.ones(concat_image_embeds.size()[:-1], dtype=torch.long).to(images.device)
    query_tokens = model.query_tokens.expand(concat_image_embeds.shape[0], -1, -1)
    query_output = model.Qformer.bert(
        query_embeds=query_tokens,
        encoder_hidden_states=concat_image_embeds,
        encoder_attention_mask=image_atts,
        output_attentions=True,
        return_dict=True,
    )
    return logits, query_output.last_hidden_state, attention


@torch.no_grad()
def collect_predictions(model: torch.nn.Module, loader: DataLoader, device: str):
    logits_all, labels_all, meta = [], [], []
    for batch in tqdm(loader, desc="table4-infer"):
        images = batch["image"].to(device, non_blocking=True)
        logits, _qformer, _attention = forward_with_attention(model, images)
        logits_all.append(logits.float().cpu().numpy())
        labels_all.append(batch["classification_labels"].cpu().numpy())
        batch_size = logits.shape[0]
        for i in range(batch_size):
            meta.append(
                {
                    "dicom_id": str(batch["dicom_id"][i]),
                    "image_path": str(batch["image_path"][i]),
                    "text_output": batch["text_output"][i],
                }
            )
    logits_np = np.concatenate(logits_all, axis=0)
    labels_np = np.concatenate(labels_all, axis=0)
    return logits_np, labels_np, meta


def positive_thresholds(thresholds: dict) -> dict[str, float]:
    return {name: float(thresholds.get(name, {}).get("positive", 0.5)) for name in COMMON_5}


def compute_table4_metrics(logits: np.ndarray, labels: np.ndarray, thresholds: dict):
    probs = softmax(logits)
    preds3 = probs.argmax(axis=-1)
    pos_threshold = positive_thresholds(thresholds)
    rows = []
    aucs, f1s_binary, f1s_weighted = [], [], []

    for name, idx in zip(COMMON_5, COMMON_5_IDXS):
        y_true_pos = (labels[:, idx] == 1).astype(np.int32)
        y_score_pos = probs[:, idx, 1]
        if 0 < y_true_pos.sum() < len(y_true_pos):
            auc_val = float(roc_auc_score(y_true_pos, y_score_pos))
        else:
            auc_val = float("nan")

        y_pred_pos = (y_score_pos >= pos_threshold[name]).astype(np.int32)
        f1_binary = float(f1_score(y_true_pos, y_pred_pos, zero_division=0))
        f1_weighted = float(
            f1_score(labels[:, idx], preds3[:, idx], average="weighted", zero_division=1)
        )
        aucs.append(auc_val)
        f1s_binary.append(f1_binary)
        f1s_weighted.append(f1_weighted)
        rows.append(
            {
                "pathology": name,
                "auc_positive_vs_rest": round(auc_val, 6),
                "f1_positive_threshold": round(f1_binary, 6),
                "f1_weighted_multiclass": round(f1_weighted, 6),
                "positive_threshold": pos_threshold[name],
                "support_positive": int(y_true_pos.sum()),
                "support_total": int(len(y_true_pos)),
            }
        )

    return {
        "mean_auc": round(float(np.nanmean(aucs)), 6),
        "mean_f1_positive_threshold": round(float(np.mean(f1s_binary)), 6),
        "mean_f1_weighted_multiclass": round(float(np.mean(f1s_weighted)), 6),
        "per_pathology": rows,
    }


def softmax(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=-1, keepdims=True)


def format_latex_cell(value) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.3f}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value).replace("_", r"\_")


def dataframe_to_latex(df: pd.DataFrame) -> str:
    align = "l" * len(df.columns)
    lines = [
        rf"\begin{{tabular}}{{{align}}}",
        r"\hline",
        " & ".join(format_latex_cell(col) for col in df.columns) + r" \\",
        r"\hline",
    ]
    for _, row in df.iterrows():
        lines.append(" & ".join(format_latex_cell(row[col]) for col in df.columns) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}", ""])
    return "\n".join(lines)


def metrics_from_table4_csv(path: Path) -> dict:
    df = pd.read_csv(path)
    required = {
        "pathology",
        "auc_positive_vs_rest",
        "f1_positive_threshold",
        "f1_weighted_multiclass",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Cannot reuse {path}; missing columns: {missing}")
    rows = df.to_dict(orient="records")
    return {
        "mean_auc": round(float(df["auc_positive_vs_rest"].mean()), 6),
        "mean_f1_positive_threshold": round(float(df["f1_positive_threshold"].mean()), 6),
        "mean_f1_weighted_multiclass": round(float(df["f1_weighted_multiclass"].mean()), 6),
        "per_pathology": rows,
    }


def save_table4_outputs(metrics: dict, out_dir: Path, dataset_label: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(metrics["per_pathology"])
    df.to_csv(out_dir / "table4_classification_metrics.csv", index=False)

    table_rows = [
        {
            "Metric": "AUC",
            "Model": "META-CXR (Ours)",
            "Mean": metrics["mean_auc"],
            **{r["pathology"]: r["auc_positive_vs_rest"] for r in metrics["per_pathology"]},
        },
        {
            "Metric": "F1-Score",
            "Model": "META-CXR (Ours)",
            "Mean": metrics["mean_f1_positive_threshold"],
            **{r["pathology"]: r["f1_positive_threshold"] for r in metrics["per_pathology"]},
        },
    ]
    paper_df = pd.DataFrame(table_rows)
    paper_df.to_csv(out_dir / "table4_paper_row.csv", index=False)
    (out_dir / "table4_paper_row.tex").write_text(
        dataframe_to_latex(paper_df),
        encoding="utf-8",
    )
    payload = {
        "dataset": dataset_label,
        "metric_definition": {
            "AUC": "positive class probability vs positive label, one-vs-rest",
            "F1-Score": "positive-vs-rest using threshold.json positive threshold per pathology",
            "alternate_f1": "weighted multiclass F1 over negative/positive/uncertain",
        },
        **metrics,
        "paper_rows": table_rows,
    }
    (out_dir / "table4_classification_metrics.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    return paper_df


def labels_to_groups(labels: np.ndarray) -> dict[str, list[str]]:
    out = {"positive": [], "uncertain": [], "negative": []}
    for name, value in zip(ABNORMALITIES_14, labels.tolist()):
        if name == "No Finding":
            continue
        out[CLASS_ID_TO_NAME.get(int(value), "negative")].append(name)
    return out


def classify_from_logits(logits: torch.Tensor, thresholds: dict, mode: str) -> dict[str, list[str]]:
    probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
    out = {"positive": [], "uncertain": [], "negative": []}
    for name, p in zip(ABNORMALITIES_14, probs):
        if name == "No Finding":
            continue
        if mode == "pn":
            threshold = float(thresholds.get(name, {}).get("positive", 0.5))
            if p[1] >= threshold:
                out["positive"].append(name)
            else:
                out["negative"].append(name)
            continue

        thresholds_abn = thresholds.get(name, {})
        best_cls, best_score = None, 0.0
        for cls_name, cls_idx in CLASS_NAME_TO_ID.items():
            threshold = float(thresholds_abn.get(cls_name, 0.5))
            if p[cls_idx] >= threshold and p[cls_idx] > best_score:
                best_cls, best_score = cls_name, float(p[cls_idx])
        if best_cls is None:
            best_cls = CLASS_ID_TO_NAME[int(p.argmax())]
        out[best_cls].append(name)
    return out


def compact_groups(groups: dict[str, list[str]]) -> dict[str, str]:
    return {
        "positive": ", ".join(groups.get("positive", [])) or "None",
        "uncertain": ", ".join(groups.get("uncertain", [])) or "None",
        "negative": "Rest",
    }


def prompt_findings(groups: dict[str, list[str]]) -> str:
    parts = []
    for label in ["positive", "uncertain"]:
        items = groups.get(label, [])
        if items:
            parts.append(f"{label.capitalize()} findings: {', '.join(items)}")
    return ". ".join(parts) if parts else "no common findings"


def build_prompt(groups: dict[str, list[str]]) -> str:
    return PROMPT_TEMPLATE.format(img_block=IMG_TOKEN_BLOCK, findings=prompt_findings(groups))


_LLM_CACHE = {}


def load_vicuna(device: str):
    if "model" in _LLM_CACHE:
        return _LLM_CACHE["model"], _LLM_CACHE["tokenizer"]
    lora_path = PROJECT_DIR / "checkpoints/lora-vicuna-7b-report-20250621"
    print(f"[llm] loading {VICUNA_MODEL_ID}")
    tokenizer = LlamaTokenizer.from_pretrained(
        VICUNA_MODEL_ID,
        use_fast=False,
        truncation_side="left",
        padding_side="left",
    )
    llm_dtype = torch.float16 if device == "cuda" else torch.float32
    llm_kwargs = {"torch_dtype": llm_dtype}
    if device == "cuda":
        llm_kwargs["device_map"] = {"": 0}
    base = LlamaForCausalLM.from_pretrained(VICUNA_MODEL_ID, **llm_kwargs)
    tokenizer.pad_token = tokenizer.unk_token
    base.base_model.img_proj_layer = nn.Linear(768, base.base_model.config.hidden_size).to(
        base.base_model.device
    )
    tokenizer.add_special_tokens({"additional_special_tokens": [IMG_TOKEN]})
    print(f"[llm] attaching LoRA {lora_path}")
    llm = PeftModelForCausalLM.from_pretrained(
        base,
        str(lora_path),
        torch_dtype=llm_dtype,
        use_ram_optimized_load=False,
    )
    if device == "cuda":
        llm = llm.half()
    llm.eval()
    _LLM_CACHE["model"] = llm
    _LLM_CACHE["tokenizer"] = tokenizer
    return llm, tokenizer


@torch.no_grad()
def generate_report(prompt: str, qformer_embs: torch.Tensor, llm, tokenizer, max_new_tokens: int) -> str:
    torch.save(qformer_embs.cpu(), "current_chat_img.pt")
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(llm.device)
    out = llm.generate(
        input_ids=input_ids,
        dicom=None,
        use_img=True,
        return_dict_in_generate=True,
        output_scores=False,
        max_new_tokens=max_new_tokens,
        num_beams=1,
        do_sample=False,
    )
    decoded = tokenizer.batch_decode(out.sequences, skip_special_tokens=True)[0]
    return decoded.split("ASSISTANT:")[-1].strip()


def tensor_to_uint8_image(image_tensor: torch.Tensor) -> np.ndarray:
    arr = image_tensor.detach().cpu().numpy()
    arr = np.transpose(arr, (1, 2, 0))
    if arr.min() < 0 or arr.max() > 1:
        arr = arr - arr.min()
        denom = arr.max() if arr.max() > 0 else 1.0
        arr = arr / denom
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return arr


def stacked_attention_overlay(image_tensor: torch.Tensor, attention_list: list[torch.Tensor]) -> np.ndarray:
    image = tensor_to_uint8_image(image_tensor)
    h, w = image.shape[:2]
    layer_maps = []
    for attn in attention_list:
        # attn: [B, expert_tokens, image_patches]; use first sample.
        a = attn[0].detach().float().cpu().numpy()
        if a.ndim != 2:
            continue
        token_mean = a.mean(axis=0)
        stream_count = max(1, token_mean.shape[0] // 49)
        token_mean = token_mean[: stream_count * 49].reshape(stream_count, 7, 7).mean(axis=0)
        layer_maps.append(token_mean)
    if not layer_maps:
        return image
    heat = np.mean(layer_maps, axis=0)
    heat = cv2.resize(heat, (w, h))
    heat = heat - heat.min()
    denom = heat.max() if heat.max() > 0 else 1.0
    heat = (heat / denom * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image, 0.55, heat, 0.45, 0)
    return overlay


def choose_figure_indices(labels: np.ndarray, count: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    candidates = []
    for i, row in enumerate(labels):
        common_pos = int(sum(row[idx] == 1 for idx in COMMON_5_IDXS))
        common_unc = int(sum(row[idx] == 2 for idx in COMMON_5_IDXS))
        all_pos = int(np.sum(row == 1))
        all_unc = int(np.sum(row == 2))
        score = common_pos * 10 + common_unc * 5 + all_pos * 2 + all_unc
        candidates.append((score, rng.random(), i))
    candidates.sort(reverse=True)
    return [i for _score, _rand, i in candidates[:count]]


def wrap_text(text: str, width: int = 45, max_chars: int = 900) -> str:
    text = " ".join(str(text).split())
    if len(text) > max_chars:
        text = text[: max_chars - 3].rstrip() + "..."
    return "\n".join(textwrap.wrap(text, width=width))


def wrap_block(text: str, width: int, max_chars: int = 1200) -> str:
    text = str(text)
    if len(text) > max_chars:
        text = text[: max_chars - 3].rstrip() + "..."
    wrapped = []
    for line in text.splitlines():
        if not line.strip():
            wrapped.append("")
            continue
        wrapped.extend(textwrap.wrap(line, width=width) or [""])
    return "\n".join(wrapped)


def save_png(path: Path, array: np.ndarray):
    Image.fromarray(array).save(path)


def render_composite_png(examples: list[dict], out_png: Path):
    font = ImageFont.load_default()
    col_widths = [230, 350, 350, 350, 230]
    header_h = 45
    wrapped_rows = []
    line_h = 12
    for ex in examples:
        row_blocks = [
            wrap_block(ex["ground_truth_text"], width=44),
            wrap_block(ex["pn_text"], width=44),
            wrap_block(ex["pnu_text"], width=44),
        ]
        max_lines = max(block.count("\n") + 1 for block in row_blocks)
        row_h = max(430, 38 + max_lines * line_h)
        wrapped_rows.append((row_h, row_blocks))
    width = sum(col_widths)
    height = header_h + sum(row_h for row_h, _blocks in wrapped_rows)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    headers = ["Radiograph", "Ground Truth", "META-CXR (PN)", "META-CXR (PNU)", "Stacked Attention Map"]
    x = 0
    for header, cw in zip(headers, col_widths):
        draw.rectangle([x, 0, x + cw, header_h], outline="gray")
        draw.text((x + 8, 15), header, fill="black", font=font)
        x += cw
    y0 = header_h
    for row_idx, ex in enumerate(examples):
        row_h, text_blocks = wrapped_rows[row_idx]
        x = 0
        for cw in col_widths:
            draw.rectangle([x, y0, x + cw, y0 + row_h], outline="gray")
            x += cw
        for img_key, col_idx in [("radiograph_png", 0), ("attention_png", 4)]:
            img = Image.open(ex[img_key]).convert("RGB")
            img.thumbnail((col_widths[col_idx] - 30, 180))
            ix = sum(col_widths[:col_idx]) + (col_widths[col_idx] - img.width) // 2
            canvas.paste(img, (ix, y0 + 25))
        for offset, block in enumerate(text_blocks, start=1):
            tx = sum(col_widths[:offset]) + 12
            draw.multiline_text((tx, y0 + 18), block, fill="black", font=font, spacing=3)
        y0 += row_h
    canvas.save(out_png)


def generate_figure_examples(
    model: torch.nn.Module,
    cfg: Config,
    labels: np.ndarray,
    meta: list[dict],
    thresholds: dict,
    args: argparse.Namespace,
    device: str,
    out_dir: Path,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    images_dir.mkdir(exist_ok=True)
    indices = choose_figure_indices(labels, args.figure_samples, args.seed)
    dataset = make_dataset(cfg, args.split, truncate=None)
    precomputed = []

    for rank, dataset_idx in enumerate(indices, start=1):
        sample = dataset[dataset_idx]
        image = sample["image"].unsqueeze(0).to(device)
        logits, qformer_embs, attention = forward_with_attention(model, image)

        gt_groups = compact_groups(labels_to_groups(sample["classification_labels"].numpy()))
        pn_groups_raw = classify_from_logits(logits[0], thresholds, "pn")
        pnu_groups_raw = classify_from_logits(logits[0], thresholds, "pnu")
        pn_groups = compact_groups(pn_groups_raw)
        pnu_groups = compact_groups(pnu_groups_raw)

        radiograph = tensor_to_uint8_image(sample["image"])
        attention_overlay = stacked_attention_overlay(sample["image"], attention)
        radiograph_path = images_dir / f"example_{rank}_radiograph.png"
        attention_path = images_dir / f"example_{rank}_stacked_attention.png"
        save_png(radiograph_path, radiograph)
        save_png(attention_path, attention_overlay)

        precomputed.append(
            {
                "rank": rank,
                "dataset_idx": int(dataset_idx),
                "sample": sample,
                "qformer_embs": qformer_embs[0:1].float().cpu(),
                "gt_groups": gt_groups,
                "pn_groups_raw": pn_groups_raw,
                "pnu_groups_raw": pnu_groups_raw,
                "pn_groups": pn_groups,
                "pnu_groups": pnu_groups,
                "radiograph_path": radiograph_path,
                "attention_path": attention_path,
            }
        )

    del dataset
    if device == "cuda":
        model.to("cpu")
        torch.cuda.empty_cache()

    llm, tokenizer = load_vicuna(device)
    examples = []

    for item in precomputed:
        rank = item["rank"]
        sample = item["sample"]
        gt_groups = item["gt_groups"]
        pn_groups_raw = item["pn_groups_raw"]
        pnu_groups_raw = item["pnu_groups_raw"]
        pn_groups = item["pn_groups"]
        pnu_groups = item["pnu_groups"]
        pn_report = generate_report(
            build_prompt(pn_groups_raw),
            item["qformer_embs"],
            llm,
            tokenizer,
            args.max_new_tokens,
        )
        pnu_report = generate_report(
            build_prompt(pnu_groups_raw),
            item["qformer_embs"],
            llm,
            tokenizer,
            args.max_new_tokens,
        )

        gt_text = (
            f"Abnormalities:\n"
            f"Positive: {gt_groups['positive']}\n"
            f"Uncertain: {gt_groups['uncertain']}\n"
            f"Negative: {gt_groups['negative']}\n\n"
            f"Report:\n{wrap_text(sample['text_output'])}"
        )
        pn_text = (
            f"Predicted Abnormalities:\n"
            f"Positive: {pn_groups['positive']}\n"
            f"Uncertain: None\n"
            f"Negative: {pn_groups['negative']}\n\n"
            f"Report:\n{wrap_text(pn_report)}"
        )
        pnu_text = (
            f"Predicted Abnormalities:\n"
            f"Positive: {pnu_groups['positive']}\n"
            f"Uncertain: {pnu_groups['uncertain']}\n"
            f"Negative: {pnu_groups['negative']}\n\n"
            f"Report:\n{wrap_text(pnu_report)}"
        )
        examples.append(
            {
                "rank": rank,
                "dataset_index": int(item["dataset_idx"]),
                "dicom_id": str(sample["dicom_id"]),
                "image_path": str(sample["image_path"]),
                "radiograph_png": str(item["radiograph_path"]),
                "attention_png": str(item["attention_path"]),
                "radiograph_rel": str(Path("images") / item["radiograph_path"].name),
                "attention_rel": str(Path("images") / item["attention_path"].name),
                "ground_truth_groups": gt_groups,
                "pn_groups": pn_groups,
                "pnu_groups": pnu_groups,
                "ground_truth_report": sample["text_output"],
                "pn_report": pn_report,
                "pnu_report": pnu_report,
                "ground_truth_text": gt_text,
                "pn_text": pn_text,
                "pnu_text": pnu_text,
            }
        )

    (out_dir / "figure6_examples.json").write_text(json.dumps(examples, indent=2), encoding="utf-8")
    render_html(examples, out_dir / "figure6_examples.html")
    render_composite_png(examples, out_dir / "figure6_examples.png")
    return examples


def render_html(examples: list[dict], out_html: Path):
    rows = []
    for ex in examples:
        radiograph_src = ex.get("radiograph_rel", ex["radiograph_png"])
        attention_src = ex.get("attention_rel", ex["attention_png"])
        rows.append(
            "<tr>"
            f"<td><img src='{html.escape(radiograph_src)}'></td>"
            f"<td><pre>{html.escape(ex['ground_truth_text'])}</pre></td>"
            f"<td><pre>{html.escape(ex['pn_text'])}</pre></td>"
            f"<td><pre>{html.escape(ex['pnu_text'])}</pre></td>"
            f"<td><img src='{html.escape(attention_src)}'></td>"
            "</tr>"
        )
    doc = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
body {{ font-family: Arial, sans-serif; margin: 20px; }}
table {{ border-collapse: collapse; width: 100%; table-layout: fixed; }}
th, td {{ border: 1px solid #777; vertical-align: top; padding: 10px; }}
th {{ font-weight: 700; text-align: center; }}
img {{ max-width: 100%; height: auto; }}
pre {{ white-space: pre-wrap; font-family: Arial, sans-serif; font-size: 13px; line-height: 1.25; }}
</style>
</head>
<body>
<table>
<thead>
<tr>
<th>Radiograph</th>
<th>Ground Truth</th>
<th>META-CXR (PN)</th>
<th>META-CXR (PNU)</th>
<th>Stacked Attention Map</th>
</tr>
</thead>
<tbody>
{''.join(rows)}
</tbody>
</table>
</body>
</html>
"""
    out_html.write_text(doc, encoding="utf-8")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_label = f"MIMIC-CXR p10 {args.split}"
    out_root = args.outputs / f"{args.run_name}_{args.split}"
    table_dir = out_root / "table4_style"
    figure_dir = out_root / "figure6_style"

    print(f"[device] {device}")
    print(f"[dataset] {dataset_label}")
    print(f"[checkpoint] {args.checkpoint}")

    thresholds = load_thresholds()
    cfg = build_cfg(args.cfg_path)
    model = None
    loader = None
    labels = None
    meta = None
    existing_table_csv = table_dir / "table4_classification_metrics.csv"

    if args.reuse_table4 and existing_table_csv.exists():
        print(f"[table4-style] reusing {existing_table_csv}")
        metrics = metrics_from_table4_csv(existing_table_csv)
        dataset = make_dataset(cfg, args.split, truncate=None)
        labels, meta = collect_labels_and_meta_from_dataset(dataset)
        del dataset
    else:
        model = build_model(cfg, args.checkpoint, device)
        loader = make_loader(cfg, args.split, args.batch_size, args.num_workers)
        logits, labels, meta = collect_predictions(model, loader, device)
        metrics = compute_table4_metrics(logits, labels, thresholds)

    paper_df = save_table4_outputs(metrics, table_dir, dataset_label)
    print("\n[table4-style]")
    print(paper_df.to_string(index=False))

    if not args.skip_figure:
        if model is None:
            model = build_model(cfg, args.checkpoint, device)
        if labels is None or meta is None:
            dataset = make_dataset(cfg, args.split, truncate=None)
            labels, meta = collect_labels_and_meta_from_dataset(dataset)
            del dataset
        print("\n[figure6-style] generating PN/PNU reports and attention maps")
        generate_figure_examples(model, cfg, labels, meta, thresholds, args, device, figure_dir)

    summary = {
        "dataset": dataset_label,
        "run_name": args.run_name,
        "checkpoint": str(args.checkpoint),
        "table4_dir": str(table_dir),
        "figure6_dir": str(figure_dir) if not args.skip_figure else None,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[saved] {out_root}")

    del model, loader
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
