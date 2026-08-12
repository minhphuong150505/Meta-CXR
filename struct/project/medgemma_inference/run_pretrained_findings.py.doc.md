> Source: `medgemma_inference/run_pretrained_findings.py` (241 dòng)
> Status: ✅ ACTIVE — ★ ENTRYPOINT P8
> Last verified against source: 2026-08-12

# `medgemma_inference/run_pretrained_findings.py`

## Purpose
CLI chạy inference FINDINGS trên checkpoint MedGemma bên thứ ba.

## Entry point
```bash
python -m medgemma_inference.run_pretrained_findings \
    --config configs/experiments/pretrained_medgemma_findings_first.yaml \
    --split validation --max-samples 100
```
⚠ **Không** gọi được qua `run_medgemma_qlora.py` — `resolve_pipeline_modes` raise
và chỉ đúng lệnh này.

## Status
```text
✅ ACTIVE — baseline chính thức (D-005)
```

## Main functions
| Hàm | Dòng | Vai trò |
|---|---|---|
| [`main(argv)`](run_pretrained_findings.py.methods/main.md) | 137 | ★ Trả exit code |
| `parse_args(argv)` | 44 | |
| `resolve_split_csv(split, override)` | 72 | |
| `resolve_vis_root(override)` | 90 | |
| `previous_projection(output_dir)` | 98 | Ước tính chi phí lần trước |
| `print_banner(...)` | 109 | ★ In model, split, **trần ngân sách** (`:122`) trước khi chạy |

## Inputs / Outputs
Vào: YAML experiment, `--split`, `--max-samples`, override path.
Ra: JSONL predictions (append-only, fsync), file ước tính chi phí, exit code.

## Calls / Called by
Gọi: `medgemma_inference.config`, `.runner`, `training.dataio.manifest` (`:33`),
`training.stage2_utils.stable_fingerprint` (`:39`), `local_config`.
Được gọi: người dùng.

## Side effects
Ghi JSONL + cost estimate. Cấp phát GPU (lazy, qua runner).

## Error / edge cases
Config có key fine-tuning cũ → `ObsoleteFineTuningConfigError` · Guard Impression
chặn trước mọi thứ · Run identity lệch → `ResumeMismatch`

## Related tests
`tests/test_pretrained_findings.py` (553 dòng)

## Developer notes
`print_banner` in trần ngân sách **trước** khi chạy — cố ý, để người dùng thấy chi
phí tối đa trước khi bấm enter.

← [`_index.md`](_index.md) · [HOME](../../HOME.md)
