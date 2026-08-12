> Source: `training/train_eval_figure9_llm_variants_200.py` (1746 dòng)
> Status: ✅ ACTIVE — ★ ĐỘNG CƠ Stage 2
> Last verified against source: 2026-08-12

# `training/train_eval_figure9_llm_variants_200.py`

## ⚠ Tên file gây hiểu nhầm nghiêm trọng

Tên gợi ý một script vẽ figure chạy một lần trên 200 mẫu. **Nó là động cơ Stage 2.**
`run_medgemma_qlora.py:49` import nó (`as fig9`), và class `VariantLLM` (~660 dòng,
~25 method) lo toàn bộ: model loading, QLoRA, prompt, collate, soft token, train
loop, generate, NLG metric, checkpoint, upload.

Docs nội bộ (`docs/refactor_hotspots.md:11`, `docs/pending_medgemma_finetuning_teardown.md:31`)
ghi nó đang chờ được tách nhỏ. Phạm vi documentation:
[D-006](../_meta/DECISIONS.md#d-006--độ-sâu-documentation-cho-động-cơ-stage-2) — mức
class + flow, `.methods/` chỉ cho method có ML logic thật.

## Purpose

Hiện thực toàn bộ cơ chế huấn luyện và đánh giá Stage 2.

## Role in architecture

```text
run_medgemma_qlora.py (quyết định kiến trúc)
        ▼
fig9.VariantLLM (cơ chế)
        ├─ MedGemma 4-bit NF4 + LoRA
        ├─ PromptBuilder (stage2/prompts)
        ├─ SoftTokenEmbeddingWrapper (mode Q-Former)
        └─ train / evaluate / generate / compute_nlg
```

## Status

```text
✅ ACTIVE — ⚠ đang chờ refactor
```

## Entry point

Có `__main__` (`:1719 main()`) và `--output-dir` riêng (`:201`), nhưng
⚠ **không nên gọi trực tiếp** — bỏ qua resolve mode, kiểm leakage và ràng buộc
section ở entrypoint chính.

## Important configuration

`MEDGEMMA_MODEL_ID` · `DEFAULT_RUN_NAME = "07_all_three"` (`:159`) ·
`ABNORMALITIES_14` (`:160`) · `CLASS_MAP = {"negative":0,"positive":1,"uncertain":2}` (`:176`) ·
`threshold.json` (`:178` — ⚠ **không bao giờ load ngầm**, phải truyền tường minh)

## Main class — `VariantLLM` (`:595`)

[📄 Chi tiết vòng đời, trạng thái, I/O](train_eval_figure9_llm_variants_200.py.methods/VariantLLM/_index.md)

### Method có ML logic — có trang riêng

| Method | Dòng | Doc |
|---|---|---|
| `__init__` | 596 | [📄](train_eval_figure9_llm_variants_200.py.methods/VariantLLM/__init__.md) |
| `train_fine` | 1130 | [📄](train_eval_figure9_llm_variants_200.py.methods/VariantLLM/train_fine.md) ★ |
| `encode_train_example` | 1017 | [📄](train_eval_figure9_llm_variants_200.py.methods/VariantLLM/encode_train_example.md) |
| `collate_train` | 1059 | [📄](train_eval_figure9_llm_variants_200.py.methods/VariantLLM/collate_train.md) |
| `_forward_batch` | 1082 | [📄](train_eval_figure9_llm_variants_200.py.methods/VariantLLM/_forward_batch.md) |
| `evaluate_loss` | 1108 | [📄](train_eval_figure9_llm_variants_200.py.methods/VariantLLM/evaluate_loss.md) |
| `generate` | 1307 | [📄](train_eval_figure9_llm_variants_200.py.methods/VariantLLM/generate.md) ★ |
| `assert_vision_tower_frozen` | 825 | [📄](train_eval_figure9_llm_variants_200.py.methods/VariantLLM/assert_vision_tower_frozen.md) |

### Method phụ trợ

| Method | Dòng | Vai trò |
|---|---|---|
| `_language_lora_targets` | 760 | Chọn module cho LoRA |
| `_align_output_head_dtype` | 778 | Đồng bộ dtype head |
| `parameter_report` | 789 | Đếm tham số train được |
| `_prompt_metadata` | 839 | version + config hash + template hash |
| `save_adapter` | 864 | Ghi adapter + img_proj |
| `load_img_proj_if_present` | 907 | Nạp lại projection soft token |
| `_render_prompt_text` | 914 | → `PromptBuilder` |
| `_chat_texts` | 930 | Dựng cặp prompt/target |
| `_load_rgb` | 964 | Đọc ảnh |
| `_native_messages`, `_native_chat_inputs` | 971, 999 | Đường native MedGemma |

## Hàm cấp module đáng chú ý

| Hàm | Dòng | Vai trò |
|---|---|---|
| `build_stage1_records` | 467 | ★ Dựng record qua Stage 1; `:316` import lazy `training.stage1.lavis_loader` |
| `classify_with_thresholds` | 352 | Logits → nhóm P/N/U theo threshold |
| `compute_nlg` | 1385 | BLEU/ROUGE/METEOR/CIDEr/BERTScore |
| `compute_sectioned_nlg` | 1455 | Tách FINDINGS/IMPRESSION rồi chấm |
| `evaluate_variant` | 1487 | Vòng eval một variant |
| `assert_private_gcs_destination` | 263 | ★ Chặn upload lên bucket public |
| `stage1_cohort_fingerprint` | 445 | Vân tay cohort để phát hiện lệch |
| `run_family`, `plot_family` | 1651, 1611 | Phần "figure9" thật sự |
| `field_value` | 343 | ⚠ `except Exception: return str(field)` — nuốt lỗi ([S7 trong docs](../_meta/LEGACY_AND_OPTIONAL.md)) |

## Calls / Called by

Gọi: `transformers`, `peft`, `bitsandbytes`, `torch`, `stage2.prompts`,
`training.medgemma.*`, `training.dataio.manifest`, `training.stage2_utils`,
`training.torch_io`, `training.run_context`, và **lazy** `training.stage1.lavis_loader` (`:316`).
Được gọi: `training/run_medgemma_qlora.py:49`.

## Side effects

Cấp phát MedGemma 4-bit trên GPU · Ghi adapter/metric/plot · Upload GCS ·
Stage-1 identity được truyền bằng immutable `Stage1Context`; file này không còn
đọc `RUN_NAME`/`THRESHOLDS` global. Đây là migration đã hoàn tất, không phải việc
đang dở.

## Error / edge cases

`assert_private_gcs_destination` raise nếu đích không riêng tư ·
`assert_vision_tower_frozen` raise nếu vision tower không đóng băng ·
⚠ `field_value:366` nuốt mọi exception → batch Stage-1 sai cấu trúc bị **âm thầm
stringify** vào prompt thay vì fail.

## Related tests

`tests/test_native_independence.py:140,166,197,207` · `tests/test_multimodal_capability.py:277` ·
`tests/test_soft_token_injection.py` · `tests/test_inference_only_invariants.py:36,191`

## Developer notes

1. **Đây là file lớn nhất repo (gấp ~2,6× file thứ hai).** Mọi thay đổi có bán kính
   ảnh hưởng rộng.
2. `Stage1Context` phải được truyền xuyên suốt; đừng đưa identity quay lại global.
3. `SoftTokenEmbeddingWrapper` và `build_cfg`/`build_stage1_model`/`make_stage1_loader`
   **đã được tách ra**; tên cũ còn re-export (`docs/migration_guide.md`). Dùng đường mới.
4. `train_fine` ghi `checkpoints/last` sau **mỗi epoch** và promote checkpoint tốt
   nhất vào output root; resume khôi phục optimizer, scheduler, epoch và RNG state.
5. `evaluate_variant` gọi `compute_sectioned_nlg` tại `:1570`; với
   `findings_and_impression`, metric full-report và metric từng section đều được ghi.

## Source relationships

- **Parent:** [`training/_index.md`](_index.md)
- **Methods:** [`train_eval_figure9_llm_variants_200.py.methods/`](train_eval_figure9_llm_variants_200.py.methods/)
- **Related:** [`run_medgemma_qlora.py`](run_medgemma_qlora.py.doc.md) · [`medgemma/_index.md`](medgemma/_index.md) · [`stage2/prompts/_index.md`](../stage2/prompts/_index.md)

← [HOME](../../HOME.md)
