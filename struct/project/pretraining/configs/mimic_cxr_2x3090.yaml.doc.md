> Source: `pretraining/configs/mimic_cxr_2x3090.yaml` (config)
> Status: ✅ ACTIVE — 2-GPU DDP
> Last verified against source: 2026-08-12

# `pretraining/configs/mimic_cxr_2x3090.yaml`

## Purpose
Recipe Stage 1 cho **2× RTX 3090 qua DDP**.

## Entry point
```bash
python -m torch.distributed.run --standalone --nproc_per_node=2 \
    -m pretraining.train --cfg-path pretraining/configs/mimic_cxr_2x3090.yaml
```

## Quan hệ với config production
Cùng kiến trúc (`vit_model_cls: [pubmedclip]` `:28`, `encoders` `:31`,
`mhcac` `:45`, `lambda_mhcac_contrastive: 0.1` `:61`) nhưng khác tham số chạy cho
hai GPU.

⚠ **Chưa GPU-tested.** README ghi rõ đường DDP "có trong code/config qua torchrun;
chưa GPU-tested".

## ⚠ Điều cần biết về DDP ở đây
| | |
|---|---|
| Effective batch | `batch_size_train × accum_grad_iters × world_size` |
| Seed | `run.seed + get_rank()` — mỗi rank khác nhau |
| W&B | Chỉ rank 0 log thật |
| ITC negative | All-gather qua rank (`_gather_with_local_grad`) + queue |
| `find_unused_parameters` | ⚠ đọc trong file; bật sẽ chậm hơn |

## Consumer
`pretraining/train.py` · `scripts/vm_preflight.py:152` (kiểm tồn tại) ·
`tests/test_pretrained_findings.py:204` (gián tiếp)

## Developer notes
1. Stage 2 **không có DDP** — chỉ Stage 1 dùng config kiểu này.
2. Khi so sánh với run 1-GPU, nhớ effective batch khác nhau nếu không chỉnh
   `accum_grad_iters`.
3. Đừng copy `warmup_steps` từ `mimic_cxr_2gpu.yaml` (32000, không bao giờ xong ramp).

← [`_index.md`](_index.md) · [HOME](../../../HOME.md)
