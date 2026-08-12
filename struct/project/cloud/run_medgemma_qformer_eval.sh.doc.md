> Source: `cloud/run_medgemma_qformer_eval.sh` (7 dòng)
> Status: 🕰 COMPATIBILITY ALIAS — tên gây hiểu nhầm
> Last verified against source: 2026-08-12

# `cloud/run_medgemma_qformer_eval.sh`

Đặt `STAGE2_IMAGE_MODE=qformer` nếu caller chưa đặt, rồi `exec run_stage2.sh`.

Tên có `eval`, nhưng script gọi pipeline Stage 2 đầy đủ; mặc định có thể train
trước khi test. `qformer` cũng là alias cũ được `run_stage2.sh` truyền qua
`--image-mode`, sau đó CLI map thành `meta_cxr_qformer`.

← [`cloud/`](_index.md) · [`run_stage2.sh`](run_stage2.sh.doc.md)
