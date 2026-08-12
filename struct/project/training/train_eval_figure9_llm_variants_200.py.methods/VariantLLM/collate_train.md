> Source: `training/train_eval_figure9_llm_variants_200.py:1059-1081`
> Status: ✅ ACTIVE

# `VariantLLM.collate_train(records, max_length=768)`

## Located in

[`training/train_eval_figure9_llm_variants_200.py`](../../train_eval_figure9_llm_variants_200.py.doc.md)

## Purpose
Gộp nhiều record đã encode thành một batch, pad về cùng độ dài.

## Execution flow
```text
FOR record: encode_train_example(record, prompt_style, max_length)
   ↓
pad input_ids / attention_mask / labels về max trong batch
   labels pad bằng -100        ← không tính loss trên padding
   ↓
stack pixel_values (nếu có)
   ↓
(mode Q-Former) stack soft-token embedding + vị trí
```

## ★ Padding label phải là `-100`, không phải pad token id
Pad bằng token id thật → model bị phạt vì không đoán được padding.

## Returns
`dict[str, Tensor]` — batch.

## Called by
`DataLoader(RecordDataset, collate_fn=self.collate_train)` trong `train_fine`.

## Side effects
Không.

## Tests
`tests/test_soft_token_injection.py` (phần shape batch)

## Modification risk
Với mode Q-Former, thứ tự hàng trong batch **phải** khớp thứ tự soft-token
embedding. Lệch = mỗi study mô tả bằng ảnh study khác, hoàn toàn im lặng.
`validate_soft_token_batch` là lưới an toàn duy nhất.
