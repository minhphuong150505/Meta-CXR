# scripts/compare_stage1_predictions.py

## Mục đích
So sánh ghép cặp hai file dự đoán Stage-1 (`.npz`) trên **cùng các study, cùng thứ
tự** — giao thức bài báo (argmax ba lớp, D-023). Viết cho ablation đa góc nhìn
(`docs/handoff/PLAN-2026-10-06-multi-aux-views.md`): hai arm chỉ khác
`model.data.max_aux_views`.

## Đầu vào / đầu ra
- `--baseline`, `--candidate`: `.npz` từ `ClassificationPredictions.save`.
  Từ chối (exit 2) nếu `sample_keys` hoặc `labels` khác nhau.
- `--subgroups-from {candidate,baseline}`: file cung cấp `num_views` để chia nhóm
  (`all`, `views_1`, `views_2`, `views_3plus`, `views_2plus`). Thiếu `num_views` → exit 2.
- `--output`: JSON, mỗi nhóm có `n`, `changed_argmax_fraction` và 9 metric
  (weighted P/R/F1, `mean_weighted_f1_5`, `macro_recall`, AUROC theo lớp và trung bình)
  với delta + CI 95% bootstrap (mặc định 1,000 lần, seed 16).

## Lưu ý
- Nâng `max_aux_views` chỉ đổi input của ~5% study; delta toàn split bị pha loãng
  ~20 lần. Đọc nhóm `views_3plus`.
- Ở thí nghiệm inference-only, nhóm `views_1`/`views_2` phải có
  `changed_argmax_fraction == 0` — input của chúng giống hệt.
- Chỉ dùng numpy + `training.evaluation`; chạy được trên CPU.

## Tests
`tests/test_compare_stage1_predictions.py`
