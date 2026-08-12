> Source: `training/train_eval_figure9_llm_variants_200.py:1108-1129`
> Status: ✅ ACTIVE

# `VariantLLM.evaluate_loss(records, batch_size, max_length)`

## Located in

[`training/train_eval_figure9_llm_variants_200.py`](../../train_eval_figure9_llm_variants_200.py.doc.md)

## Purpose
Cross-entropy trên validation — **tiêu chí chọn checkpoint của Stage 2**.

## Execution flow
```text
model.eval()
with torch.no_grad():
   FOR batch: _forward_batch(batch) → loss
   tổng loss có trọng số theo số example trong batch
return CE trung bình
```

## ★ Vì sao dùng CE chứ không dùng BLEU
CE tính được **mỗi epoch, rẻ, và không cần generate**. BLEU đòi hỏi generate cho
toàn bộ val split — tốn gấp nhiều lần. CE là proxy hợp lý cho việc chọn checkpoint;
BLEU/ROUGE dành cho đánh giá cuối trên test.

## Parameters
`records` (validation) · `batch_size` · `max_length`

## Returns
`float` — CE trung bình.

## Side effects
Chuyển model và `img_proj` sang `.eval()`. Hàm **không** tự trả về `.train()`;
`train_fine()` bật lại train mode ở đầu epoch kế tiếp. Không cập nhật tham số.

## Called by
`train_fine` sau mỗi epoch.

## Tests
Không có test trực tiếp (cần GPU).

## Modification risk
Nếu gọi hàm này ngoài `train_fine()`, caller phải tự quyết định khi nào bật lại
`.train()`. Đừng đổi trọng số aggregation từ số example sang số batch, vì batch
cuối có thể nhỏ hơn.
