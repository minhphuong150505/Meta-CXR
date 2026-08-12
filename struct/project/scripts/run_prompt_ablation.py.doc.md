> Source: `scripts/run_prompt_ablation.py` (158 dòng)
> Status: 🧪 ABLATION
> Last verified against source: 2026-08-12

# `scripts/run_prompt_ablation.py`

## Purpose
Ablation prompt **không train, không load model**.

## Why it exists
Docstring `:3`: mặc định là **dry run** — render từng prompt variant cho mỗi record
và ghi metadata + aggregate ở mức prompt. *"No model is loaded and NO generation
metrics are produced, so nothing here is a model result."*

## Entry point
```bash
python scripts/run_prompt_ablation.py \
    --prompt-configs configs/prompt_ablation/P5_visual_primary.yaml \
    --max-samples 1000 --output-dir outputs/prompt_ablation
```

## Main functions
`main()` (`:126`) · `_dry_run_variant(config_path, records, out_dir)` (`:56`) ·
`_resolve_configs(patterns)` (`:46`) · `_load_records(path, limit)` (`:38`)

## Calls / Called by
Gọi: `stage2.prompts` (`:30`), `scripts/_stage2_fixtures` (khi không có dữ liệu).

## Side effects
Ghi JSON/JSONL vào `--output-dir`.

## Related tests
`tests/test_stage2_prompts.py`

## Developer notes
Kết quả ở đây là **thống kê prompt**, không phải chất lượng model. Đừng trích dẫn
như metric.

← [`_index.md`](_index.md) · [HOME](../../HOME.md)
