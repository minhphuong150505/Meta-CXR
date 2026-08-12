> Source: `model/lavis/models/blip2_models/modeling_llama_imgemb.py:715-803`
> Status: ✅ ACTIVE — Vicuna demo only

# `LlamaForCausalLM.forward(...)`

## Purpose

Gọi `LlamaModel` có image embedding, chiếu hidden state qua `lm_head`, và tính
shifted causal-language-model loss khi có labels.

## Called by

`inference.py` dùng class qua `PeftModelForCausalLM` và gọi `.generate()`;
generation đi qua `prepare_inputs_for_generation`, giữ `dicom`/`use_img` và cache
giữa các bước.

## Risk

Tham số custom phải được giữ xuyên generation; làm rơi `use_img` sau token đầu sẽ
biến các bước sau thành text-only mà không nhất thiết raise.

← [`modeling_llama_imgemb.py`](../../modeling_llama_imgemb.py.doc.md) · [HOME](../../../../../../../HOME.md)
