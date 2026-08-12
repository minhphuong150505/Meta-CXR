> Source: `pretraining/configs/blip2_pretrain_stage1_emb.yaml` (97 dòng)
> Status: ✅ ACTIVE — Vicuna/Gradio demo config
> Last verified against source: 2026-08-12

# `pretraining/configs/blip2_pretrain_stage1_emb.yaml`

## Purpose

Config được `inference.sh` truyền vào `inference.py`. Nó dựng BLIP-2 Stage 1 và
trỏ tới LoRA Vicuna cho demo; không phải recipe training production.

## Important keys

| Key | Giá trị / effect |
|---|---|
| `model.encoders` | BioViL + PubMedCLIP; Swin tắt |
| `model.load_finetuned` | `True`; load checkpoint path hardcode dưới `pretraining/outputs/` |
| `model.llm.lora_path` | `checkpoints/lora-vicuna-7b-report-20250621` |
| `model.mhcac.threshold_path` | `threshold.json`, nhưng `inference.py:get_response` hiện không truyền nó vào classifier |
| `run.evaluate` | `True` — chỉ liên quan nếu dùng config qua train entrypoint |

## Failure / portability

Hai path checkpoint/LoRA là local generated artifact và không được Git track;
fresh clone không chạy được nếu chưa cung cấp chúng. Các hyperparameter training
trong file là di sản của run cũ, không nên copy sang production config.

← [`pretraining/configs/`](_index.md) · [HOME](../../../HOME.md)
