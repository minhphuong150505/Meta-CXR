# `vision_encoders/qformer_inputs.py`

> Thư mục: [`vision_encoders/`](_index.md) · Thêm 2026-10-15 · 🧪 opt-in (mọi cờ mặc định tắt)
> Last verified against source: 2026-10-15

## Purpose
Hai công cụ chống **soft token META-Former sụp** (32 token của một study gần như là một vector,
cosine 0,9989 sau 1a — [`diagnose_qformer_collapse.py`](../scripts/diagnose_qformer_collapse.py.doc.md)).
Dùng bởi pha `phase1q` (train lại chỉ META-Former trên checkpoint đã xong, đóng băng). Kế hoạch và
cổng: `docs/handoff/PLAN-2026-10-15-qformer-refit.md`. Chỉ phụ thuộc torch → test được trên CPU.

## Thành phần
| Tên | Vai trò |
|---|---|
| `resolve_qformer_input_norm(value)` | `None` (khóa vắng = đường cũ, giống từng bit) hoặc `"per_stream"`; giá trị khác → lỗi |
| `QformerInputNorm(stream_names, visual_dim)` | một `LayerNorm(1408)` mỗi luồng, áp lên span của luồng trong `SharedVisualTokens`; trả **bản sao** cho Q-Former. Từ chối tập luồng khác hoặc span không liền nhau |
| `batch_centred_participation_ratio(states)` | `[B, Q, D]` → `[B]`: số hướng hiệu dụng của phần thay đổi theo ảnh (mỗi query trừ trung bình của nó trong batch), tính từ Gram: `(tr G)² / ‖G‖²_F` |
| `query_diversity_loss(states)` | `mean_b log(Q/PR_b)/log(Q)` ∈ [0, 1]: 1 = một hướng thông tin ảnh, 0 = Q hướng độc lập bằng nhau. Batch 1 study → 0 có graph (chunk lẻ cuối của GradCache) |
| `assert_input_norm_keys_match(model_keys, ckpt_keys)` | báo lỗi khi model và checkpoint không cùng có/không có `qformer_input_norm.*` |

## Vì sao như vậy
- **Chỉ Q-Former đọc bản đã chuẩn hóa.** Q-Former của BLIP-2 được pretrain trên token ViT đã
  LayerNorm (`ln_vision`); ở đây sau StreamAdapter chỉ BioViL có LayerNorm, PubMedCLIP/Swin vào
  thẳng (pha 1b đẩy norm 19,6 → 1.713 và 12 → 413). MHCAC vẫn đọc token gốc, nên MHCAC của checkpoint
  không đổi. Đây là phép per-span duy nhất được phép chỉ một nhánh thấy; nó chuẩn hóa, không chiếu lại
  (quy tắc của [`shared_visual_tokens.py`](shared_visual_tokens.py.doc.md)).
- **Participation ratio sau khi trừ trung bình theo batch, không phải trung bình cosine.** Trừ trung
  bình từng query trong batch để một hằng số khác nhau cho mỗi query không được tính là đa dạng;
  dùng PR (không phải cosine² từng cặp) để "một query mang thông tin + 31 query gần hằng số" (hướng
  phần dư chỉ là nhiễu dropout, cosine ≈ 0) vẫn bị tính là sụp. Cả hai ghim bằng test.
- ⚠ **Đo 2026-10-15: PR vẫn bị lách được bằng nhiễu dropout nếu loss tính lúc train.** Nhiễu dropout
  gần đẳng hướng giữa 32 query; trong refit 1q mô hình học cách khuếch đại nó lên 3,9 lần phần phụ
  thuộc ảnh, nên lúc train PR ~30 (loss 0,014) trong khi chế độ eval chỉ 2,32 (loss 0,77). Chỉ dùng
  loss này khi Q-Former không có dropout, và so giá trị train với val
  ([báo cáo thực thi](../../../docs/handoff/PLAN-2026-10-15-qformer-refit.md)).
- **Báo lỗi khi khóa norm không khớp.** Thiếu norm trong model → tensor norm của checkpoint bị bỏ âm
  thầm; thiếu trong checkpoint → norm chạy ở khởi tạo. Cả hai cho báo cáo trôi chảy từ soft token sai.

## Calls / Called by
- Gọi: `torch`.
- Được gọi bởi: [`Blip2Qformer`](../model/lavis/models/blip2_models/blip2_qformer.py.doc.md)
  (`qformer_input_norm`, `_qformer_tokens`, `forward`, `forward_gradcache`),
  [`training/stage1/lavis_loader.py`](../training/stage1/lavis_loader.py.doc.md) (`build_stage1_model`),
  [`scripts/diagnose_qformer_collapse.py`](../scripts/diagnose_qformer_collapse.py.doc.md).

## Tests
`tests/test_qformer_inputs.py` (CPU): norm = LayerNorm từng span, không sửa bản MHCAC, ở khởi tạo
không luồng nào trội về độ lớn; sụp → 1; hằng số theo query → 1; một query + 31 hằng → > 0,99; query
độc lập → < 0,25, bất biến theo thang; batch 1 → 0 có backward; một bước gradient phá sụp một phần;
khóa norm phải khớp. Mức model (máy train): `tests/test_gradcache_siglip.py`.
