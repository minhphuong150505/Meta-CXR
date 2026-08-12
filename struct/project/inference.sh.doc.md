> Source: `inference.sh` (shell)
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-12

# `inference.sh`

## Purpose
Launcher một dòng cho `inference.py`.

## Nội dung cốt lõi
```bash
python3 inference.py --cfg-path pretraining/configs/blip2_pretrain_stage1_emb.yaml   # :7
```

Đây là **lý do `blip2_pretrain_stage1_emb.yaml` không phải legacy**, dù nó nằm
cạnh các config legacy khác.

## Called by
`Dockerfile:5` ENTRYPOINT; người dùng chạy trực tiếp.

## Failure points
Thiếu `configs/env_config.yaml` · Thiếu LoRA trong `checkpoints/` · Không GPU

## Developer notes
Đổi config ở đây đổi cả hành vi container.

← [HOME](../HOME.md)
