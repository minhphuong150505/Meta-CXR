> Source: `mhcac/late_view_pool.py`
> Status: 🧪 OPT-IN — chỉ khi `model.view_fusion.mode: late` (vắng key = `early`, mọi run đã ghi)
> Last verified against source: 2026-10-26

# `mhcac/late_view_pool.py`

## Purpose

Late fusion đa view: MHCAC phân loại **từng view riêng** (cùng trọng số), rồi
`LateViewPooling` gộp logit của các view thật của study, riêng cho từng finding.

## Why it exists

Early fusion (`view_fusion.py`) sửa token của anchor trước MHCAC, và mọi study (kể
cả study một view) đều đi qua StreamAdapter và MPC. Đo được: fusion ON kém OFF trên
study một view (step 0b/0c, pilot aux-drop, Swin-only). Late fusion giữ đường của
anchor đúng như OFF (không adapter, không khối fusion, không MPC).

**Đảm bảo theo cấu trúc:** study chỉ có một view thật nhận đúng logit của anchor.
**Không đảm bảo:** MHCAC còn được train trên ảnh phụ (phần lớn là lateral), nên độ
chính xác của nhóm một view vẫn có thể đổi. Plan và quy tắc đọc:
`docs/handoff/PLAN-2026-10-26-late-view-fusion.md`.

## Role in architecture

```text
anchor ─► encoder ─► shared projector ─► MHCAC ─► z_anchor [B,14,3] ─┐
aux (chỉ hàng thật) ─► encoder (no_grad) ─► ln_vision / projector ─► │
                         MHCAC (cùng trọng số) ─► z_aux [B,N,14,3] ──┴► LateViewPooling ─► [B,14,3]
```

## Contents

| Tên | Vai trò |
|---|---|
| `VIEW_FUSION_MODES` | `("early", "late")` |
| `resolve_view_fusion_mode(mode, *, multi_view, lambda_mpc, lambda_view_consistency, lambda_explanation, lambda_explanation_strong)` | `None` → `early`. `late` cần `multi_view: true` (đó là thứ làm dataset nạp ảnh phụ). Từ chối MPC, view consistency (đọc stream trước fusion của early) và explanation loss (Grad-CAM một chuỗi token) |
| `LateViewPooling(num_findings=14, num_classes=3, num_view_types=4)` | `s[v,f] = w_f · [log_softmax(z[v,f]), onehot(loại view)] + b_f`; softmax trên các view THẬT; `out[f] = Σ_v a[v,f] z[v,f]`. `w = b = 0` lúc khởi tạo → trung bình các view thật. 14 × 8 = 112 tham số |
| `LateViewPooling.weights(...)` | Trọng số gộp `[B, 1+N, 14]`, anchor đứng đầu, bằng 0 ở slot đệm |
| `late_classification_loss(cls_loss_fn, pooled, anchor, labels, sample_mask, anchor_weight=0.5)` | `(1-w) CE(pooled) + w CE(anchor)`. Study một view: pooled == anchor → đúng loss của OFF |

## Invariants (pinned by `tests/test_late_view_fusion.py`)

- Study không có ảnh phụ thật → trả `anchor_logits` nguyên bit (`torch.where`, không nhân 1.0).
- Giá trị rác (NaN) hay id loại view rác ở slot đệm không đổi output.
- Trọng số cộng lại bằng 1 trên các view thật, bằng 0 ở slot đệm.
- Gradient tới `weight`, tới anchor của study một view, không tới slot đệm.
- Shape sai hoặc id loại view ngoài 0..3 → `ValueError`.

## Used by

`model/lavis/models/blip2_models/blip2_qformer.py` (`Blip2Qformer.__init__`,
`_late_view_logits`, `forward`); `tests/test_late_view_fusion.py`.

## Dependencies

Chỉ `torch`. Không LAVIS, nên kiểm được trên máy CPU.

## Status of evidence

Đã chạy trên GPU 2026-10-26 (pilot val 2 seed, phase 1b, 60.000 study, 2 epoch). Study một
view nhận đúng logit của anchor, có hay không có ảnh phụ (kiểm trên cả hai model pilot, 988 study).
**Kết quả: DỪNG, không thắng OFF** (wF1 trung bình hai seed +0,0000, F1-5 -0,0005). Trọng số gộp
gần như đứng yên ở 0,5 cho ảnh phụ (0,48-0,50 mỗi finding), tức là trung bình hai view. Chi tiết
trong plan trên.

Biến thể tiếp theo, `model.view_fusion.late_detach_aux: true` (2026-10-26): lượt ảnh phụ không
truyền gradient vào MHCAC, chỉ pool học từ nó. Module này không đổi; công tắc nằm trong
`Blip2Qformer._late_view_logits`. Plan: `docs/handoff/PLAN-2026-10-26-late-fusion-detach-aux.md`.
