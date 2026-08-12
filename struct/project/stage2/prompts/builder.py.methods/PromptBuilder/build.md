> Source: `stage2/prompts/builder.py:35-55`
> Status: ✅ ACTIVE

# `PromptBuilder.build(context)`

## Purpose

Validate `PromptContext`, dựng ordered `PromptPart` từ visual/structured/normal/
context/instruction policies, rồi trả `RenderedPrompt` mang config/template hash.

## Important conditions

`qformer_visual_only` không được nhận structured label; guided mode xem label là
auxiliary. Thứ tự part là một phần của prompt contract và train/inference parity.

## Returns

`RenderedPrompt` có parts, version, config hash, template hash; render text xảy ra ở
API của built prompt/caller, không đụng tokenizer hay torch.

## Tests / risk

`tests/test_stage2_prompts.py`. Đổi whitespace/order có thể đổi prompt byte và
artifact hash dù semantic nhìn giống nhau.

← [`builder.py`](../../builder.py.doc.md) · [HOME](../../../../../HOME.md)
