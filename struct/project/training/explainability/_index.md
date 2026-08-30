> Source: `training/explainability/` (5 file Python, ~1.700 dòng)
> Status: ✅ ACTIVE — Chặng 1 xong và đã chạy trên GPU thật 2026-08-30
> Last verified against source: 2026-08-30

# `training/explainability/`

## Purpose

Lớp giải thích **hậu nghiệm** cho tầng sinh ngôn ngữ (Stage 2): câu nào trong
báo cáo sinh ra dựa vào phần nào của đầu vào thị giác.

## ⚠ Ràng buộc số một: package này là NGƯỜI QUAN SÁT

Không một module nào của Stage 1 hay Stage 2 import package này. Xoá cả thư mục
đi thì hai pipeline chạy y nguyên, không đổi một byte.

```text
pretraining/train.py, training/run_medgemma_qlora.py
        │  KHÔNG import training/explainability
        ▼  (một chiều, chỉ đọc)
training/explainability/     ← đọc attention, không sửa gì
```

Lý do không phải khẩu hiệu: một lớp giải thích có thể thay đổi thứ nó đang giải
thích thì không còn là lớp giải thích. Cùng lý do đó, `attention_capture.py`
lắp/tháo mọi hook trong `finally`, không mutate model vĩnh viễn.

## ⚠ Ràng buộc số hai: đâu là đường CÓ nghĩa không gian

| Đường | Chặng 1 một mình cho ra gì |
|---|---|
| `medgemma_direct` | ✅ **heatmap hợp lệ** — vision tower của MedGemma có patch grid thật |
| `meta_cxr_qformer*` | ❌ chỉ biết câu nào dùng soft token nào, **không phải vùng ảnh nào** |

32 soft token là `query_output.last_hidden_state` của Q-Former. Chúng **không
mang vị trí**: mỗi cái đã cross-attend qua toàn bộ 246 visual token, và trọng số
cross-attention đó **không nằm trong record Stage-2 đã cache** (record chỉ có
`[32, 768]`). Muốn thành heatmap phải chạy lại Stage 1 với hook — đó là Chặng 2,
chưa implement.

`projection.assert_spatial_projection_supported("qformer_soft_token")` **ném
`SpatialProjectionUnsupported`** thay vì vẽ ra một bức ảnh trông có nghĩa mà
không có nghĩa.

## Parent

[`training/`](../_index.md)

## Children

| File | LOC | Doc | Status |
|---|---|---|---|
| `rollout.py` | 325 | [📄](rollout.py.doc.md) | ✅ Chefer rollout, **chỉ import `torch`** |
| `projection.py` | 297 | [📄](projection.py.doc.md) | ✅ lưới đặc trưng → không gian ảnh, **chỉ import `torch`** |
| `sentence_attribution.py` | 326 | [📄](sentence_attribution.py.doc.md) | ✅ tách câu, nhãn `lexicon_v1`, parse coverage |
| `__init__.py` | 34 | — | ✅ chỉ docstring, không import eager |
| `attention_capture.py` | ~560 | [📄](attention_capture.py.doc.md) | ✅ module DUY NHẤT chạm model sống |

## Vì sao chia module đúng theo lằn ranh này

Ba module đầu **chạy được trên máy CPU không có transformers/torchvision** —
đúng môi trường của phiên lập kế hoạch (`/home/phuong/venv`: torch CPU + numpy,
không có gì khác, và CLAUDE.md cấm cài thêm). `attention_capture.py` là module
DUY NHẤT chạm model sống. Mọi thứ nằm trên nó testable không cần GPU chính là để
nó giữ được kích thước nhỏ.

`rollout.py` bị ràng buộc mạnh nhất: **không import model, không import
`transformers`, không import chéo trong repo.** Vào là tensor attention, ra là
vector quy kết. Đó là điều khiến toán học của nó kiểm được bằng ma trận dựng tay.

## Bảo mật dữ liệu

Kế thừa nguyên quy tắc của `scripts/evaluate_explanation.py`:

- Không định danh bệnh nhân trong tên file.
- Từ chối ghi vào path trong repo mà `git check-ignore` không xác nhận là
  ignored (`_assert_private_output_location`).
- Attribution map lưu `.npz` ở **độ phân giải lưới gốc** (14×14 / 7×7), không
  lưu PNG đã upsample.
- Đầu ra dạng `.jsonl` (đã nằm trong `.gitignore`), một dòng một study.

## Tests

[`tests/explainability/`](../../tests/_index.md#nhóm-8--explainability-stage-2)
— **149 test**, toàn bộ chạy CPU, `rc=0`. Phần GPU (4 góc hình học, triệt tiêu
trên study thật) chưa tự động hoá — số liệu trong
[`attention_capture.py.doc.md`](attention_capture.py.doc.md).

## Related documentation

- [`training/`](../_index.md) — parent
- [`training/medgemma/soft_tokens.py`](../medgemma/soft_tokens.py.doc.md) — nơi
  32 soft token được thay vào embedding
- [`scripts/evaluate_explanation.py`](../../scripts/_index.md) — XAI của Stage 1,
  nguồn của `_assert_private_output_location` và `_normalize_cam`
- [`safety/claims.py`](../../safety/_index.md) — `split_sentences`,
  `LexiconClaimParser`
