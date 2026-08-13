> Source: `mhcac/`
> Status: ✅ ACTIVE (chỉ 3/16 file)
> Last verified against source: 2026-08-12

# `mhcac/`

## Purpose

**M**ulti-**H**ead **C**ross-**A**ttention **C**lassification — khối phân loại 14
bệnh lý × 3 lớp, cộng toàn bộ hàm loss của Stage 1 và module hợp nhất đa view.

## ⚠ Đọc trước: chỉ 3 trong 16 file đang được dùng

| File | Status | Bằng chứng |
|---|---|---|
| `mhcac_12.py` | ✅ | `blip2_qformer.py:23` |
| `loss.py` | ✅ | `blip2_qformer.py:35` |
| `view_fusion.py` | 🟡 | `blip2_qformer.py:41`, chỉ khi `multi_view: true` |
| `mhcac.py`, `mhcac_2.py` … `mhcac_11.py` (11 file) | 🕰 | **zero reference** |
| `utils.py` (624 dòng) | 🕰 | ⚠ **không còn caller nào** — notebook 03 đã bị xóa 2026-08-13 |
| `aggregator.py` | 🕰 | không có `import`; chỉ còn chuỗi `"aggregator"` trong freeze-list `runner_base.py:189` |

11 variant là **lịch sử phát triển kiến trúc**, không phải ablation đang chạy.
Chúng cũng bị loại khỏi ruff trong `pyproject.toml` — dấu hiệu đã được coi là
legacy từ trước. Quyết định: [D-003](../_meta/DECISIONS.md#d-003--mhcac-variants-và-encoder-trùng-lặp-là-legacy).

## Role in project

Nằm giữa biểu diễn thị giác và output phân loại của Stage 1:

```text
SharedVisualTokens ──► MHCAC ──► logits [B,14,3] ──► ClassificationLoss
                         ▲
                         │ (chỉ lúc train)
                    text embeddings
```

Đây cũng là nơi **teacher/student separation** được hiện thực — ranh giới quan
trọng nhất của Stage 1.

## Parent

[`struct/project/`](../../HOME.md#source-code-tree)

## Children

| File | Doc | Status | Vai trò |
|---|---|---|---|
| `mhcac_12.py` (448) | [📄](mhcac_12.py.doc.md) | ✅ | `AbnormalityClassificationModel` + 4 module phụ |
| `loss.py` (628) | [📄](loss.py.doc.md) | ✅ | `ClassificationLoss`, `soft_target_kl_loss`, `MultiPositiveContrastiveLoss`, `view_consistency_loss`, `AbnormalitySpecificLoss`, `AttentionLoss` |
| `view_fusion.py` | [📄](view_fusion.py.doc.md) | 🟡 | `ViewFusionModule`, `ViewFusionBlock` |
| `__init__.py` | — | ✅ | **Rỗng** — mọi import phải nêu đủ tên module |
| 13 file legacy | — | 🕰 | [LEGACY_AND_OPTIONAL.md](../_meta/LEGACY_AND_OPTIONAL.md#l1--mhcac-variants) |

## Main responsibilities

1. **Phân loại** — 14 expert token cross-attend vào token thị giác qua 6 lớp.
2. **Teacher/student** — cùng một module, hai chế độ, phân biệt bằng
   `text_embeddings=None` hay không.
3. **Toàn bộ loss Stage 1** — `loss.py` là nơi duy nhất định nghĩa chúng.
4. **Hợp nhất đa view** — `view_fusion.py`, một module cho mỗi encoder.

## Entry points

Không có. Đây là thư viện, chỉ được import.

## Dependencies

`torch`, `torch.nn`, `torch.nn.functional`. **Không** phụ thuộc LAVIS, không phụ
thuộc `vision_encoders/` (nhận `SharedVisualTokens` như một duck-typed object).

## Used by

| Ai | Import gì |
|---|---|
| `model/lavis/models/blip2_models/blip2_qformer.py` | `:23` `mhcac_12.AbnormalityClassificationModel` · `:35` `loss.*` · `:41` `view_fusion.ViewFusionModule` |
| `tests/test_stage1_objectives.py` | `mhcac_12`, `loss` |
| `tests/test_view_fusion.py` | `view_fusion` |
| `tests/test_multiview_losses.py` | `loss.MultiPositiveContrastiveLoss`, `view_consistency_loss` |

## Execution flow

```text
Blip2Qformer.forward()
   │
   ├─ mhcac(shared_visual, text_embeddings=None, labels, sample_mask)   ← STUDENT
   │     ↓
   │  embedding_alignment → slice theo spans → pos_enc → 6 lớp attention
   │     ↓
   │  expert_loss → orth/contrastive/sparsity     14 × classifier → logits
   │
   ├─ mhcac(shared_visual, text_embeddings=<text>)                      ← TEACHER
   │
   └─ mhcac(anchor_shared, ...)                          ← ANCHOR-ONLY (view consistency)
```

⚠ **MHCAC được gọi tối đa 3 lần trong một forward.** Đây là chi phí tính toán
đáng kể và là điều cần biết trước khi tối ưu.

## Important configurations

Trong khối `model:` của run YAML:

| Key | Mặc định | Ảnh hưởng |
|---|---|---|
| `mhcac.uncertain_policy` | `three_class` (prod: `ignore_uncertain`) | Cách xử lý lớp Uncertain |
| `mhcac.distill_temperature` | `2.0` | Nhiệt độ trong `soft_target_kl_loss` |
| `mhcac.text_dropout` | `0.2` | Dropout trên đường text của teacher |
| `mhcac.label_smoothing` | `0.05` | |
| `mhcac.class_weights` | 14×3 sqrt inverse-frequency | `[]` → tắt weighting (ablation) |
| `multi_view` | `false` (prod: `true`) | Có dựng `ViewFusionModule` không |
| `view_fusion.*` | heads 8, ffn_ratio 4, blocks 1, dropout 0.1, p_view_drop 0.15 | |
| `loss.lambda_*` | xem [ARCHITECTURE.md §2.6](../_meta/ARCHITECTURE.md#26-tổng-hợp-loss) | Trọng số 11 loss |

## Status

```text
✅ ACTIVE — nhưng chỉ 3/16 file
```

## Notes

- ⚠ **Typo trong API công khai:** tham số tên `num_commmon_tokens` (ba chữ `m`),
  `mhcac_12.py:208` và `blip2_qformer.py:347`. Sửa nó là breaking change.

- ⚠ **Freeze-list mồ côi:** `runner_base.py:189` vẫn tìm attribute `aggregator`,
  nhưng `Blip2Qformer` không còn tạo nó. Nếu ai thêm lại `self.aggregator`, nó sẽ
  **tự động bị đóng băng** mà không có thay đổi code nào khác.

- **`__init__.py` rỗng** — không re-export gì. Mọi import phải là
  `from mhcac.mhcac_12 import …`, không phải `from mhcac import …`.

- **Đừng sửa `mhcac_8..11.py`** để "thống nhất style" — chúng bị loại khỏi ruff
  có chủ đích.

## Related documentation

- [ARCHITECTURE.md §2.4](../_meta/ARCHITECTURE.md#24-mhcac--phân-loại-bất-thường)
- [GLOSSARY: MHCAC](../_meta/GLOSSARY.md#mhcac) · [P/N/U](../_meta/GLOSSARY.md#pnu) · [Teacher/student](../_meta/GLOSSARY.md#teacherstudent)
- [`blip2_qformer.py`](../model/lavis/models/blip2_models/blip2_qformer.py.doc.md)
- [`tests/_index.md`](../tests/_index.md)

← [Về HOME](../../HOME.md)
