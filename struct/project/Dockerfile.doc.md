> Source: `Dockerfile` (container)
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-12

# `Dockerfile`

## Purpose
Image chạy demo Gradio.

## ★ ENTRYPOINT trỏ vào đường Vicuna
```dockerfile
ENTRYPOINT ["/bin/bash", "inference.sh"]     # :5
```
→ Container build ra chạy [P9](_meta/PIPELINES.md#p9--gradio-demo-vicuna-7b),
**không phải** Stage 1 hay Stage 2. Đây là bằng chứng chính cho
[D-002](_meta/DECISIONS.md#d-002--đường-vicuna-7b-legacy-vẫn-là-demo-active).

## Cách dùng
```bash
./build_container.sh && ./run_container.sh    # GPU + Gradio :7860
```

## Dependencies
Base image `meta-cxr:1.0.0` phải có sẵn Python/model dependencies · Java cho
CheXpert labeler · LoRA adapter trong `checkpoints/` (không track trong Git →
phải mount hoặc copy vào)

## Failure points
Thiếu `configs/env_config.yaml` → `FileNotFoundError` lúc import ·
Thiếu LoRA adapter → không load được Vicuna · Không có GPU → hỏng

## Developer notes
1. Dockerfile không cài requirements; toàn bộ environment đến từ base image
   `meta-cxr:1.0.0`. Kiểm provenance và package version của base trước khi build.
2. Muốn container chạy Stage 1/Stage 2 thay vì demo → phải đổi ENTRYPOINT.
3. **Đừng COPY dữ liệu MIMIC vào image.** Mount lúc chạy.

← [HOME](../HOME.md)
