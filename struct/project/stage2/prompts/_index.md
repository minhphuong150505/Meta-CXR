> Source: `stage2/prompts/` (7 module)
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-12

# `stage2/prompts/`

## Purpose

`PromptBuilder` — **điểm vào prompt duy nhất** cho cả train và inference Stage 2.
Không đụng model, tokenizer hay torch.

## Role in project

```text
record (native hoặc Q-Former)
        ↓ context_from_record()
    PromptContext
        ↓ PromptBuilder.build()
    PromptPart[] + prompt_version + config_hash + template_hash
        ↓
    MedGemma  (train VÀ inference dùng chung đường này)
```

## Parent

[`stage2/`](../_index.md)

## Children

| File | Doc | Vai trò |
|---|---|---|
| `builder.py` (259) | [📄](builder.py.doc.md) | `PromptBuilder`, `fit_to_budget` |
| `schemas.py` | [📄](schemas.py.doc.md) | 5 visual mode, `PartKind`, `PromptConfig` |
| `policies.py` (205) | [📄](policies.py.doc.md) | Chính sách negative/uncertain/temporal |
| `templates.py` | [📄](templates.py.doc.md) | Template + `template_hash()` |
| `ontology.py` | [📄](ontology.py.doc.md) | 14 tên bệnh lý, mirror từ `fig9` nhưng torch-free |
| `records.py` | [📄](records.py.doc.md) | `context_from_record` |
| `validation.py` (208) | [📄](validation.py.doc.md) | Kiểm tra prompt hợp lệ |
| `__init__.py` | — | Re-export `PromptBuilder`, `load_prompt_config`, `contains_temporal_language`, … |

## Năm visual mode — và vì sao ranh giới quan trọng

`schemas.py`:

| Mode | Thấy Stage-1 labels? |
|---|---|
| `native_anchor_only` | ❌ |
| `native_anchor_guided` | ✅ |
| `native_multiview` | ❌ ⚠ manifest native hiện chỉ luồn anchor, `auxiliary_views` rỗng |
| `qformer_visual_only` | ❌ — **đây là thứ giữ cho ablation không bị nhiễm** |
| `qformer_guided` | ✅ |

> Chỉ guided mode nhìn thấy prediction có cấu trúc, và chúng được diễn đạt là
> **gợi ý phụ trợ có thể sai**, không phải ground truth. `qformer_visual_only`
> **không nhận label nào** — nếu nó nhận, so sánh giữa hai đường không còn nghĩa.

Docstring `schemas.py:16` ghi việc nêu tường minh các mode này đã sửa một
`uses_mhcac_prompt` từng là code chết.

## Main responsibilities

1. Dựng prompt từ record, theo visual mode.
2. Áp chính sách: dùng compact summary khi không có positive/uncertain; giới hạn
   số negative; diễn đạt uncertain bằng ngôn ngữ thận trọng; **cấm so sánh thời
   gian khi không có prior**.
3. Phát ra version + config hash + template hash để truy vết kết quả.
4. Validate prompt.

## Entry points

Không có. Thư viện.

## Dependencies

**Chỉ stdlib** + `yaml` để đọc config. Không torch, không transformers.

## Used by

`training/train_eval_figure9_llm_variants_200.py:146-151, :928` ·
`training/run_medgemma_qlora.py:410` · `scripts/run_prompt_ablation.py` ·
`scripts/export_stage2_prompt_samples.py` · `scripts/prompt_length_statistics.py` ·
`scripts/audit_temporal_targets.py:28` · `tests/test_stage2_prompts.py` (408 dòng)

## Important configurations

| File | Vai trò |
|---|---|
| `configs/stage2_prompt_v2.yaml` | Prompt v2 — **opt-in** qua `--prompt-config`; bỏ flag → prompt legacy |
| `configs/prompt_ablation/P1..P9.yaml` | 9 biến thể cho `scripts/run_prompt_ablation.py` |

## Status

```text
✅ ACTIVE — nhưng OPT-IN. Không có --prompt-config thì dùng prompt legacy.
```

## Notes

- **Prompt prefix bị mask khỏi training label** — model học sinh phần đáp án, không
  học sinh lại prompt.
- **Soft token vào `bad_words_ids`** khi generate.
- ⚠ **Temporal guard trong prompt ≠ dữ liệu đã sạch.** `temporal_target_policy`
  mặc định vẫn là `keep`; split hiện chưa mang prior linkage đầy đủ.
  `scripts/audit_temporal_targets.py` đo mức độ vấn đề này.
- ⚠ `native_multiview` **tồn tại nhưng chưa hoàn chỉnh end-to-end** — manifest
  native chỉ luồn anchor image.
- `ontology.py` mirror tên bệnh lý từ `fig9` để giữ torch-free. **Hai nơi phải
  khớp nhau**; `tests/test_stage2_prompts.py` canh điều đó.

## Related documentation

[ARCHITECTURE.md §4](../../_meta/ARCHITECTURE.md#4-prompt-v2) ·
[PIPELINES.md → P6](../../_meta/PIPELINES.md#p6--prompt-ablation-dry-run) ·
[`configs/_index.md`](../../configs/_index.md)

← [`stage2/`](../_index.md) · [HOME](../../../HOME.md)
