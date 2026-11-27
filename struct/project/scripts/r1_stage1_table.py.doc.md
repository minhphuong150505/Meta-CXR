# scripts/r1_stage1_table.py

CPU, opt-in 2026-11-27. Bảng Stage 1 theo giao thức bài báo cho OFF + residual lateral R1 so với OFF, chấm lại MỘT lần
đọc test của Bước 3 (ứng viên và cutpoint đã chốt trước; không chọn gì ở đây).

- Đầu vào: npz của [`residual_stack.py --save-arms`](residual_stack.py.doc.md).
- `build_table`: với mỗi arm, `evaluate_classification(decisions=...)` (headline: cutpoint fit trên val — PL theo arm,
  còn lại theo file của OFF; KHÔNG phải giao thức bài báo) và `evaluate_classification` argmax (giao thức bài báo).
  R1 = trung bình các seed chuyên gia (nên wF1 / F1-5 / macro recall trùng `whole_<split>["OFF+L"]`, có test).
- Kiểm: hàng ngoài PL của R1 phải trùng OFF từng bit (xác suất và quyết định), sai thì `ValueError`.
- Khoảng tin cậy: bootstrap cặp theo study (2.000, seed 16), cùng mẫu cho hai arm và mọi seed; nhóm `all` và `pl`.
  `off_reported` = OFF với file cutpoint trên mọi study (như số đã báo cáo của `run_20261024_nomv`), không CI.
- Chỉ ghi số tổng hợp (JSON). Từ chối ghi đè.

Callee: [`classification_metrics.evaluate_classification`](../training/evaluation/classification_metrics.py.methods/evaluate_classification.md),
`schemas.ClassificationPredictions`. Test: `tests/test_residual_lateral.py::test_save_arms_and_paper_table_reproduce_whole_split`.
