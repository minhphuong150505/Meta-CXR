> Source: `training/medgemma/finding_tokens.py` (~300 dòng)
> Status: 🧪 EXPERIMENTAL — mặc định TẮT (`--finding-tokens off`)
> Last verified against source: 2026-09-10

# `training/medgemma/finding_tokens.py`

## Purpose

Đưa **thông tin mention của Stage 1** vào Stage 2 qua **13 finding token học
được**, thay vì qua ngưỡng cứng + một câu tiếng Anh như `--cue-rule`.

Stage 1 cho hai đại lượng mỗi finding:

| | |
|---|---|
| `m = sigmoid(mention_logits)` | báo cáo có **nhắc tới** finding này không |
| `q = softmax(classification_logits)` | phân bố {negative, positive, uncertain}, **có điều kiện đã nhắc tới** |

79,5% ma trận CheXpert để trống và bị mask khỏi classification loss, nên `q`
**chưa từng thấy** "vắng mặt khỏi báo cáo". Đó là lý do `m` không thể bỏ đi.

## ⚠ Đây là nhánh thử nghiệm, không phải mặc định

Mọi đường vào module này chỉ mở khi `--finding-tokens` khác `off`. Với cờ tắt,
đường Stage 2 mặc định **giống hệt từng byte** — đó là điều kiện để các arm
không có finding token vẫn so sánh được với mọi số đã ghi.

## ⚠ "Không được nhắc đến" KHÔNG phải "âm tính"

`m` thấp làm **cả ba** số polarity co về 0, chứ không lật thành một khẳng định
âm tính. `q` không xác định khi finding chưa từng được nhắc; mã hoá nó thành
một negative tự tin là điều tệ nhất kênh này có thể làm.
Pinned bởi `test_low_mention_never_becomes_a_negative_assertion`.

## Main items

| Tên | Vai trò |
|---|---|
| `FINDING_TOKEN` | `"<finding_token>"` — placeholder, khác `<qformer_soft_token>` |
| `NUM_FINDING_TOKENS = 13` | `ABNORMALITIES_14` bỏ `No Finding` |
| `finding_features(class_logits, mention_logits, mode)` | `[14,3]` + `[14]` → `[13, k]` |
| `FindingTokenEncoder(nn.Module)` | `[B,13,k]` → `[B,13,hidden]` |
| `FindingTokenEmbeddingWrapper(nn.Module)` | thay thế embedding tại 13 vị trí |
| `apply_finding_feature_ablation(...)` | can thiệp lúc inference (`zero`, `shuffle_within`) |

## Feature vector

| mode | số học | k |
|---|---|---|
| `q_only` | `[q_neg, q_pos, q_unc]` | 3 |
| `full` | `[m, m*q_pos, m*q_neg, m*q_unc]` | 4 |

`m*q_pos + m*q_neg + m*q_unc == m` chính xác, nên `m` **thừa về mặt tuyến
tính** trong `full`. Vẫn giữ vì nó cho một projection tuyến tính đường đi thẳng
tới "có được nhắc không".

`q_only` là **đối chứng cô lập đóng góp của mention**: cùng identity, cùng
projection, cùng 13 vị trí — chỉ khác ở chỗ `m` có nhân vào hay không.

## Ba lựa chọn thiết kế, mỗi cái là một cách thí nghiệm có thể im lặng đo sai

1. **Identity từ embedding học được, không từ vị trí.** Thứ tự cố định và được
   test ghim, nhưng model không phải suy ra token nói về finding nào từ chỗ nó
   nằm. Nhờ vậy `W` **dùng chung** cho 13 finding, nên các số có **một** ý
   nghĩa nhất quán thay vì 13 ý nghĩa học riêng.
2. **LayerNorm trước projection.** `E[i]` tự do lớn lên khi train, còn `f_i`
   bị chặn trong [0,1]. Không chuẩn hoá thì các con số thành phần vô cùng nhỏ
   của norm đầu vào và projection có thể học cách **bỏ qua** chúng — rồi kết
   quả sẽ bị báo cáo nhầm thành "thông tin mention không giúp gì".
3. **Rescale RMS theo chính bảng embedding.** Việc thay thế xảy ra **bên
   trong** `get_input_embeddings()`, tức là **sau** khi Gemma nhân hệ số
   `Gemma3TextScaledWordEmbedding` cho token thật nhưng **không** cho vector
   thay thế. `gamma` khởi tạo bằng RMS **đầu ra** của bảng embedding (không
   phải RMS của `weight` — lệch đúng bằng hệ số đó).

## Attention mask

**Không đổi gì, và cố ý không đổi.** 13 finding token là vị trí bình thường
trong `input_ids` với `attention_mask = 1`; decoder của MedGemma là causal, nên
đặt chúng **sau** soft token và **trước** instruction là đủ để instruction và
mọi token sinh ra đều nhìn thấy được. Thêm mask riêng ở đây sẽ tạo phân kỳ im
lặng giữa đường train và đường generate mà chẳng được gì.

## Composition, không phải sửa `soft_tokens.py`

```text
FindingTokenEmbeddingWrapper( SoftTokenEmbeddingWrapper( base_embedding ) )
        ↑ 13 <finding_token>            ↑ 32 <qformer_soft_token>
```

Wrapper trong thay 32 soft token, wrapper ngoài thay 13 finding token.
`soft_tokens.py` **không bị sửa một dòng nào**, nên các arm không dùng finding
token chạy đúng code đã tạo ra mọi kết quả đã ghi.

## Fail-closed

Cùng lý do như `soft_tokens.py`: sai index theo hàng thì loss vẫn giảm, mỗi
study vẫn được mô tả — bằng dự đoán Stage-1 của study khác, không có lỗi nào.
Wrapper raise khi batch không khớp, số vị trí khác 13, hoặc hidden width sai.

Ở tầng trên: `adapter_is_complete` / `resumable_adapter` đòi `finding_tokens.pt`
khi cờ bật, và `load_finding_encoder_if_present` **raise** khi thiếu file (khác
`load_img_proj_if_present`, vốn trả về im lặng).

## Cache identity

Records cũ không có `class_logits`. `stage1_cohort_fingerprint` thêm khoá
`record_features = "with_class_logits"` **chỉ khi** cờ bật — đúng khuôn mẫu
`cue_rule` đã dùng — nên mọi cache 10 GiB hiện có vẫn hit khi cờ tắt.
`assert_class_logits_present` fail-closed nếu records thiếu.

## Ablation lúc inference

`zero`, `shuffle_within` (đổi chỗ 13 hàng: token nói "Edema" nhưng mang số của
Pneumothorax), `permute_across` (áp dụng ở tầng caller trong
`scripts/generate_stage2_reports.py` vì cần study thứ hai).

> ⚠ Đây là can thiệp **ngoài phân phối**, là probe cơ chế, **không thay thế**
> arm `q_only` được train riêng: model có thể xấu đi khi đầu vào bị phá mà vẫn
> không hưởng lợi gì từ đầu vào đúng.

## Dependencies

`torch` — và không gì khác. Đó là điều cho phép test shape/thứ tự/mask/gradient
chạy trên máy CPU không có MedGemma.

## Used by

`training/train_eval_figure9_llm_variants_200.py` (dual-import shim) ·
`training/run_medgemma_qlora.py` (`--finding-tokens`) ·
`scripts/generate_stage2_reports.py` (`--finding-tokens`,
`--finding-feature-ablation`) · `tests/test_finding_tokens.py`

## Related documentation

[`soft_tokens.py.doc.md`](soft_tokens.py.doc.md) ·
[`docs/handoff/PLAN-2026-09-10-mention-finding-tokens.md`](../../../../docs/handoff/PLAN-2026-09-10-mention-finding-tokens.md)

← [`training/medgemma/`](_index.md) · [HOME](../../../HOME.md)
