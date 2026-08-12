> Source: `cloud/run_medgemma_l4_bucket_pipeline.sh` (5 dòng)
> Status: 🕰 COMPATIBILITY ALIAS
> Last verified against source: 2026-08-12

# `cloud/run_medgemma_l4_bucket_pipeline.sh`

Backward-compatible alias của `run_stage2.sh`. File không chọn L4, không cấu hình
bucket riêng và không thay argument; toàn bộ behavior đến từ environment và
`run_stage2.sh "$@"`.

← [`cloud/`](_index.md) · [`run_stage2.sh`](run_stage2.sh.doc.md)
