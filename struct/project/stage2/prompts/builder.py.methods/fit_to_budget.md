> Source: `stage2/prompts/builder.py:230-259`
> Status: ✅ ACTIVE

# `fit_to_budget(rendered, count_fn, max_tokens)`

## Purpose

Giảm một `RenderedPrompt` theo budget do caller định nghĩa, đồng thời giữ phần
visual/instruction bắt buộc; các phần có `budget_priority` lớn bị bỏ trước.

## Contract

Hàm không tự tokenize: `count_fn(text)` quyết định đơn vị đếm cho phần text, còn
image/soft-token dùng trường `count`. Vì vậy tên `max_tokens` là contract của
caller, không bảo đảm luôn là tokenizer token. Hàm trả một `RenderedPrompt` mới,
giữ nguyên metadata/hash và thứ tự các phần còn lại. Các phần priority `0` không
bị bỏ; nếu riêng chúng đã vượt budget thì kết quả vẫn có thể vượt `max_tokens`.

## Tests / risk

`tests/test_stage2_prompts.py`. Đổi thứ tự cắt thay đổi thông tin model thấy và là
thay đổi thí nghiệm, không phải refactor thẩm mỹ.

← [`builder.py`](../builder.py.doc.md) · [HOME](../../../../HOME.md)
