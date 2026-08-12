> Source: `requirements-stage2.txt` (9 dòng)
> Status: ✅ ACTIVE — additive Stage-2 lock file
> Last verified against source: 2026-08-12

# `requirements-stage2.txt`

## Purpose

Include nguyên `requirements-stage1.txt`, rồi thêm `accelerate`, `bitsandbytes`,
`peft`, `safetensors`, `sentencepiece` và `bert-score` cho MedGemma QLoRA.

## Important correction

Hai lock file hiện tại **không pin torch/transformers xung đột**:
Stage 2 dùng chính pin Stage 1 qua `-r requirements-stage1.txt`. Hai venv vẫn được
launcher/setup khuyến nghị để cách ly môi trường và tránh cài Stage-2 extras nặng
vào workflow Stage 1, nhưng không phải vì resolver không thể hợp nhất pin hiện tại.

← [HOME](../HOME.md)
