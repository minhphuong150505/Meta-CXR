> Source: `training/train_eval_figure9_llm_variants_200.py:1919-1960`
> Status: ✅ ACTIVE (thêm 2026-10-14)

# `VariantLLM.generate_candidates(record, prompt_style, max_new_tokens, *, num_samples, temperature, top_p, top_k)`

## Located in

[`training/train_eval_figure9_llm_variants_200.py`](../../train_eval_figure9_llm_variants_200.py.doc.md)

## Purpose
Sinh `num_samples` báo cáo bằng sampling cho **một** record — các giả tham chiếu mà
[`scripts/mbr_select_stage2.py`](../../../scripts/mbr_select_stage2.py.doc.md) dùng để
chấm từng ứng viên. Cùng prompt, stop ID, `bad_words_ids` và đường soft token với
[`generate`](generate.md); chỉ khác cách giải mã.

## Execution flow
```text
kiểm tra num_samples ≥ 1, temperature > 0, 0 < top_p ≤ 1, top_k ≥ 0  (ValueError)
   ↓
_generate_texts(..., sampling={do_sample: True, temperature, top_p, top_k,
                               num_return_sequences: num_samples})
   ↓ soft/finding token repeat_interleave(num_samples)
model.generate → [num_samples, L] → decode từng chuỗi → clean_text
```

## Parameters
| Tham số | Ghi chú |
|---|---|
| `num_samples` | số chuỗi trả về |
| `temperature` | > 0 |
| `top_p`, `top_k` | **bắt buộc**: `generation_config` của MedGemma không đặt, bỏ trống thì HF dùng `top_k=50`. `top_k=0`, `top_p=1.0` = tắt |

## Returns
`list[str]` dài `num_samples`.

## Side effects
Sampling rút từ RNG toàn cục của torch — người gọi tự seed
(`generate_stage2_reports.py` dùng `study_seed(seed, sample_key)`).

## Called by
`scripts/generate_stage2_reports.py` khi `--num-samples > 0`.

## Tests
`tests/test_beam_generation.py`: `test_candidates_sample_and_decode_every_sequence`,
`test_candidates_refuse_bad_sampling_arguments`, `test_sampling_repeats_soft_tokens_per_sample`.
