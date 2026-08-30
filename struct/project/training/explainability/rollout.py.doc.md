> Source: `training/explainability/rollout.py` (325 dòng)
> Status: ✅ ACTIVE — ⚠ công thức chưa đối chiếu tài liệu thiết kế
> Last verified against source: 2026-08-30

# `training/explainability/rollout.py`

## Purpose

Gradient-weighted attention rollout, viết dưới dạng đại số tensor thuần.

## ⚠ Đọc trước: module này CỐ TÌNH không có dependency

Import duy nhất: `torch`, `dataclasses`, `collections.abc`.
**Không model, không `transformers`, không import chéo trong repo.**

Vào là ma trận attention, ra là vector quy kết. Đó là lý do toán học của nó kiểm
được bằng đáp án tính tay trên máy CPU — máy duy nhất phiên lập kế hoạch có.
Thêm một import "cho tiện" vào đây là phá mất tính chất đó.

## Công thức đã implement

`METHOD_CHEFER` (mặc định) — Chefer et al. 2021, *Generic Attention-model
Explainability*, quy tắc self-attention:

```text
Abar[l] = mean_h( relu( grad(A[l]) ⊙ A[l] ) )     # fuse_heads
R[0]    = I
R[l]    = R[l-1] + Abar[l] @ R[l-1]               # rollout
```

`METHOD_ABNAR` — Abnar & Zuidema 2020, không dùng gradient:

```text
Ahat[l] = rownorm( 0.5*A[l] + 0.5*I )
R       = Ahat[L] @ ... @ Ahat[1]
```

⚠ **Ba phương trình trên viết ra từ cách repo này đọc Chefer et al.** Tài liệu
thiết kế của dự án (Mục 4) KHÔNG đọc được ở phiên viết module. Đối chiếu với
mục đó trước khi trích bất kỳ con số nào. Nếu công thức khác, chỗ phải sửa gọn
trong `fuse_heads` + `_rollout_chefer`.

## ⚠ Clamp TRƯỚC rồi mới trung bình head

`fuse_heads` làm `.clamp(min=0).mean(dim=1)`, đúng thứ tự của implementation gốc.
Đảo lại thành mean-rồi-clamp cho phép một head âm triệt tiêu một head dương:

```text
head 0: A*G = +4 ,  head 1: A*G = -4
  clamp-rồi-mean = mean(4, 0) = 2.0   ← đúng
  mean-rồi-clamp = clamp(0)   = 0.0   ← sai, cùng shape, không báo lỗi
```

Pin bằng `test_fuse_heads_clamps_before_averaging_not_after`.

## Quy ước hướng

`R[i, j]` = ảnh hưởng quy kết của **key** `j` lên **query** `i`. Hàng attention
là query. Trong decoder nhân quả `A` là tam giác dưới; rollout không cần xử lý
riêng — soft token đứng trước mọi token sinh ra nên tất cả đều với tới được.

## Main items

| Item | Vai trò |
|---|---|
| `stack_layers(per_layer, batch_index)` | Tuple `[B,H,S,S]` của `output_attentions=True` → `[L,H,S,S]`. Đường nối tới model sống, nhưng là tensor thuần nên nằm ở phía test được. |
| `fuse_heads(attention, gradient=None, row_normalize=False)` | `[L,H,S,S]` → `[L,S,S]`. `gradient=None` **chính là fallback không trọng số gradient** |
| `rollout(fused, method, subtract_identity, ...)` | → `([S,S], RolloutTrace)` |
| `span_attribution(R, query_positions, key_positions, reduce, normalize)` | → một giá trị cho mỗi key, đúng thứ tự truyền vào |
| `RolloutTrace` | `method`, `num_layers`, `sequence_length`, `gradient_weighted`, `row_normalized` |

## ⚠ Fallback không bao giờ im lặng

`RolloutTrace.gradient_weighted` mặc định `False` khi caller không khai báo — thà
báo thiếu còn hơn báo thừa. Một map chạy bằng fallback không được phép ghi lại
như một map gradient-weighted đầy đủ.

## ⚠ Span không nhận khối lượng → zeros, KHÔNG phải phân phối đều

"Câu này không dùng ảnh" là một kết quả thật. Biến nó thành `1/N` là bịa ra bằng
chứng không có. Pin bằng
`test_a_span_with_no_mass_returns_zeros_not_a_uniform_distribution`.

## `subtract_identity` — vì sao mặc định tắt

Nó **không liên quan** tới câu hỏi của dự án: token sinh ra và dải soft token rời
nhau, nên `I` đóng góp đúng 0 vào `R[generated, soft]`. Chỉ bật khi đọc khối
đường chéo.

## Error / edge cases

- Không vuông ở hai chiều cuối, rỗng, hoặc chứa giá trị không hữu hạn → `ValueError`.
- `gradient` khác shape `attention` → `ValueError` nêu cả hai shape.
- `query_positions`/`key_positions` ngoài phạm vi → `IndexError`; trùng lặp hoặc
  rỗng → `ValueError`.
- Hàng toàn 0 khi `row_normalize=True` → giữ 0, không thành `NaN`.

## Calls / Called by

- Calls: `torch` only.
- Called by: `attention_capture.py` (chưa tồn tại).

## Related tests

`tests/explainability/test_rollout.py` — 30 test. Mọi giá trị kỳ vọng đều có
phép tính tay trong comment ngay trên nó. Đáng chú ý:
`test_chefer_layers_compose_in_order_and_are_not_commuted` (bắt lỗi đảo thứ tự
lớp) và `test_soft_token_span_is_attributed_through_two_layers_by_hand`.
