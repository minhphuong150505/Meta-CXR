> Source: `scripts/prompt_length_statistics.py` (109 dòng)
> Status: 🧪 ABLATION
> Last verified against source: 2026-08-12

# `scripts/prompt_length_statistics.py`

## Purpose
Thống kê độ dài prompt và target.

## ⚠ Trung thực về độ chính xác — docstring `:3`
Đếm token chính xác cần tokenizer MedGemma (`--tokenizer google/medgemma-1.5-4b-it`,
cần transformers). **Không có nó, script rơi về đếm theo khoảng trắng và đánh dấu
output `"approximate": true`** — để không ai nhầm proxy với số token thật.

## Entry point
```bash
python scripts/prompt_length_statistics.py --config configs/stage2_prompt_v2.yaml
```
Ghi `outputs/prompt_length_statistics.json`.

## Main functions
`main()` (`:44`) · `_percentile(values, pct)` (`:27`) · `_make_counter(name)` (`:35`)

## Calls / Called by
Gọi: `stage2.prompts` (`:24`), transformers (tùy chọn).

## Side effects
Ghi JSON.

## Developer notes
Số liệu độ dài quyết định `max_length` khi train. Dùng bản có tokenizer thật khi
ra quyết định, không dùng bản approximate.

← [`_index.md`](_index.md) · [HOME](../../HOME.md)
