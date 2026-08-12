> Source: `scripts/export_stage2_prompt_samples.py` (100 dòng)
> Status: 🧪 — ⚠ output chứa findings text
> Last verified against source: 2026-08-12

# `scripts/export_stage2_prompt_samples.py`

## Purpose
Render prompt cho vài record ra JSONL để debug.

## ⚠ Cảnh báo dữ liệu — docstring `:3`
*"Never writes Q-Former embedding tensors. Rendered per-sample prompts DO contain
findings text, so this is a debug tool: point `--output` at a private location."*

**Output chứa report text thật.** Đặt `--output` ở nơi riêng tư, không commit.

## Entry point
```bash
python scripts/export_stage2_prompt_samples.py \
    --config configs/stage2_prompt_v2.yaml --output <private>/samples.jsonl
```

## Main functions
`main()` (`:48`) · `_load_records(path, num)` (`:37`) · `_word_count(text)` (`:44`)

## Calls / Called by
Gọi: `stage2.prompts` (`:28`), `stage2.prompts.schemas.PartKind` (`:34`).

## Side effects
⚠ Ghi file **chứa findings text**.

## Developer notes
`.gitignore` chặn `*.jsonl`, nhưng đừng dựa vào đó — đặt output ngoài repo.

← [`_index.md`](_index.md) · [HOME](../../HOME.md)
