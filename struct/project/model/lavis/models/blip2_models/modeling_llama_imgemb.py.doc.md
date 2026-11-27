> Source: `model/lavis/models/blip2_models/modeling_llama_imgemb.py` (975 dòng)
> Status: ✅ ACTIVE — nhưng chỉ trên đường Vicuna legacy
> Last verified against source: 2026-08-12

# `modeling_llama_imgemb.py`

## Purpose

Llama (Vicuna-7B) được sửa để **nhận image embedding trực tiếp** thay vì chỉ token
id. Đây là cách đường demo tiêm biểu diễn thị giác vào LLM.

## Why it exists

Llama gốc chỉ nhận `input_ids`. Để đưa embedding của Q-Former vào giữa prompt,
cần một `forward` chấp nhận embedding đã tính sẵn ở vị trí ảnh.

## Role in architecture

⚠ **Từ 2026-11-27 không còn nơi nào dùng file này.** Người dùng duy nhất là demo Vicuna-7B `inference.py` (P9), đã xóa theo D-029. File giữ lại vì nằm trong fork LAVIS.

```text
(đã xóa) inference.py (P9)
   ├─ Blip2Qformer  → image embedding
   └─ LlamaForCausalLM (file này) + LoRA adapter  → text báo cáo
```

⚠ **Chỉ dùng ở đường Vicuna.** Stage 2 MedGemma dùng cơ chế khác
(`SoftTokenEmbeddingWrapper`, thay thế tại vị trí `<qformer_soft_token>`).

## Status

```text
✅ ACTIVE — theo D-002 (Docker vẫn chạy đường này)
🕰 LEGACY về kiến trúc — chưa migrate sang MedGemma
```

## Used in

Inference ✅ (P9 Gradio demo) · Training ❌ · Stage 2 ❌

## Entry point

Không.

## Inputs

`input_ids`, `attention_mask`, `inputs_embeds`, `past_key_values`, `dicom`, và
`use_img`. `dicom` tra embedding theo ID; `use_img=True` nạp
`current_chat_img.pt` ở bước generation đầu tiên.

## Outputs

`CausalLMOutputWithPast` — logits, past_key_values.

## Main classes

Kiến trúc Llama chuẩn đã sửa: `LlamaForCausalLM` là điểm vào chính (trước đây
`inference.py` import, file đó đã xóa 2026-11-27). ⚠ Phần lớn là code upstream HuggingFace/LAVIS.

Các đường chính: [`LlamaModel.forward`](modeling_llama_imgemb.py.methods/LlamaModel/forward.md)
và [`LlamaForCausalLM.forward`](modeling_llama_imgemb.py.methods/LlamaForCausalLM/forward.md).

## Calls

`transformers` Llama primitives.

## Called by

Không còn nơi nào trong repo import file này (`inference.py` đã xóa 2026-11-27).

## Data flow

```text
ảnh ─► Blip2Qformer ─► image embedding
                            │
prompt text ─► token ids ───┤
                            ▼
               LlamaForCausalLM.forward  (embedding chèn tại vị trí ảnh)
                            ▼
                      text báo cáo
```

## Side effects

Cấp phát Vicuna-7B trên GPU khi được gọi (hiện không có caller).

## Error / edge cases

⚠ Cần runtime verification — đường này chưa được chạy trong phiên audit nào.

## Related tests

Không có test nào cover file này.

## Related documentation

[PIPELINES.md → P9](../../../../_meta/PIPELINES.md#p9--gradio-demo-vicuna-7b) ·
`inference.py` (đã xóa 2026-11-27) ·
[D-002](../../../../_meta/DECISIONS.md#d-002--đường-vicuna-7b-legacy-vẫn-là-demo-active)

## Developer notes

1. **Đây không phải đường Stage 2.** Đừng nhầm cơ chế tiêm ảnh ở đây với
   `SoftTokenEmbeddingWrapper` — chúng khác nhau.
2. Demo `inference.py` đã xóa 2026-11-27; file này là code chết trong fork LAVIS.
3. **Không có test.** Sửa file này không có lưới an toàn nào.
4. ⚠ Nhiều nội dung chưa được đọc kỹ trong lần audit này — mô tả ở đây dựa trên
   chữ ký, import và vai trò cũ trong `inference.py`. Kiểm code trước khi dựa vào.

## Source relationships

- **Parent:** [`model/lavis/_index.md`](../../_index.md)
- **Related:** [`blip2_qformer.py`](blip2_qformer.py.doc.md)

← [HOME](../../../../../HOME.md)
