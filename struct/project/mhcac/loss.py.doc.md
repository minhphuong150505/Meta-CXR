> Source: `mhcac/loss.py` (628 dòng)
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-12

# `mhcac/loss.py`

## Purpose

**Nơi duy nhất** định nghĩa hàm loss của Stage 1. Sáu thành phần, ba trong số đó
được `Blip2Qformer.forward` gọi trực tiếp.

## Why it exists

Gom loss vào một file cho phép chúng chia sẻ ngữ nghĩa mask — đặc biệt là
`sample_mask`, thứ khiến một dòng không có nhãn đóng góp **đúng 0** thay vì một
giá trị nhỏ.

## Role in architecture

```text
Blip2Qformer.forward
   ├─ ClassificationLoss           → L_cls, L_teacher_cls
   ├─ soft_target_kl_loss          → L_distill
   ├─ MultiPositiveContrastiveLoss → L_mpc
   └─ view_consistency_loss        → L_vc
MHCAC nội bộ
   └─ AbnormalitySpecificLoss      → orth, contrastive, sparsity
```

## Status

```text
✅ ACTIVE — ClassificationLoss, soft_target_kl_loss, MultiPositiveContrastiveLoss,
            view_consistency_loss, AbnormalitySpecificLoss
⚠ AttentionLoss (:439) — không thấy caller
```

## Used in

Training ✅ · Validation ✅ (tính loss) · Inference ❌

## Entry point

Không.

## Main classes / functions

| Tên | Dòng | Doc | Được gọi bởi |
|---|---|---|---|
| `ClassificationLoss` | 35 | [📄](loss.py.methods/ClassificationLoss/_index.md) | `blip2_qformer:913` (student), `:933` (teacher) |
| `soft_target_kl_loss` | 123 | [📄](loss.py.methods/soft_target_kl_loss.md) | `blip2_qformer:936` |
| `MultiPositiveContrastiveLoss` | 523 | [📄](loss.py.methods/MultiPositiveContrastiveLoss.md) | `blip2_qformer:949` |
| `view_consistency_loss` | 590 | [📄](loss.py.methods/view_consistency_loss.md) | `blip2_qformer:970` |
| `AbnormalitySpecificLoss` | 241 | [📄](loss.py.methods/AbnormalitySpecificLoss.md) | Trong `mhcac_12` (`expert_loss`) |
| `AttentionPooling` | 215 | — | Trong `AbnormalitySpecificLoss` |
| `AttentionLoss` | 439 | — | ⚠ Không thấy caller |

## Từng loss làm gì

### `ClassificationLoss(logits, true_labels, sample_mask=None)`
Cross-entropy 3 lớp cho 14 bệnh lý, có `class_weights` (14×3), `label_smoothing`,
và `uncertain_policy`.

**`sample_mask` là điểm mấu chốt:** dòng không có nhãn CheXpert đóng góp **0**.
Không phải "một giá trị nhỏ", không phải "được điền mặc định".

`class_weights` mặc định (`blip2_qformer.py:357`) là `sqrt(prevalence_negative /
prevalence_class)`, cap ở 10, tính từ **cohort train theo study**. `[]` trong YAML
tắt weighting cho ablation.

### `soft_target_kl_loss(student_logits, teacher_logits, sample_mask, temperature)`
KL divergence student ← teacher. **Teacher được `detach()`** — gradient không chảy
ngược vào teacher. `temperature` từ `mhcac.distill_temperature` (prod: 2.0).

### `MultiPositiveContrastiveLoss(anchor, aux, aux_mask)`
Kéo các view **của cùng một study** lại gần nhau. `temperature=0.07`. Chỉ chạy khi
`multi_view` bật và batch có aux thật.

### `view_consistency_loss(fused_logits, anchor_logits, has_aux, margin=0.0, confidence_gate=False, gate_tolerance=0.0)`

Kéo dự đoán đã fuse về gần dự đoán chỉ-anchor, **có điều kiện**. `has_aux` giới
hạn về đúng những study có aux view.

⚠ **Tiền đề cũ đã bị bác bỏ 2026-08-16.** Docstring gốc biện minh rằng "thêm view
không được làm đổi *những* bất thường nào được dự đoán". Điều đó sai với bộ dữ
liệu này: view lateral tồn tại chính là để cho thấy thứ view frontal không thấy,
và 55.3% study train (121,738 / 220,216) có aux view. Dạng symmetric KL vô điều
kiện phạt mô hình vì đã *dùng* view thứ hai.

Hai knob nới nó ra, **cả hai mặc định tắt** để tái lập được hành vi cũ cho ablation:

| Knob | Ý nghĩa |
|---|---|
| `margin` | Hinge: phân kỳ dưới ngưỡng này không tốn gì. Trôi nhẹ là tái cân bằng bình thường, chỉ lật thật mới bị tính. |
| `confidence_gate` | Miễn phạt ở ô mà bản fuse **tự tin hơn** (entropy thấp hơn) so với anchor quá `gate_tolerance` nat. Sắc nét lại là dấu hiệu của bằng chứng mới; nhoè đi là dấu hiệu của nhiễu. |

★ **Gate được `.detach()`.** Nó chỉ chọn *nơi* loss được áp, không được mang
gradient — nếu không, mô hình có thể tối thiểu hoá term bằng cách thao túng gate
thay vì sửa dự đoán.

Với `margin=0.0, confidence_gate=False` hàm trả về **đúng** giá trị cũ; pinned bởi
`tests/test_multiview_losses.py::test_view_consistency_defaults_reproduce_legacy_value`.

Config: `model.view_consistency.{margin, confidence_gate, gate_tolerance}`.
Prod dùng `margin: 0.05`, `confidence_gate: true` — **chưa chạy trên GPU**.

### `MentionConditionedClassificationLoss` + `mention_marginal_log_probs`

Thêm 2026-08-16. Thay **cả** `ClassificationLoss` lẫn `MentionGateLoss` bằng một
likelihood phân cấp duy nhất, bật bằng `loss.lambda_mention_conditioned_cls > 0`
(constructor **raise** nếu `lambda_cls` hoặc `lambda_gate` cũng > 0).

⚠ **Vấn đề nó sửa.** Gate và classifier là hai head **không bao giờ gặp nhau**:
gate có thể trả lời "không hề được nhắc" trong khi classifier trả lời "Positive",
và không có gì hoà giải, vì dự đoán của gate chỉ nuôi BCE của chính nó. Gate được
thêm vào chính là để chặn việc đoán dương tràn lan, và **nó đã không làm được** —
đo trên test split ngày 2026-08-16, ngưỡng đã hiệu chỉnh, `ignore_uncertain`:

```text
macro_specificity 0.2637      recall 0.9021 vs precision 0.6835
specificity ~0: Support Devices 0.000, Fracture 0.000,
                Lung Opacity 0.017, Atelectasis 0.100
```

```text
không được nhắc     ->  -log(1 - m)
được nhắc, lớp y    ->  -log(m) - log(q[y])

P(Negative)  = (1 - m) + m * q_negative
P(Positive)  =           m * q_positive
P(Uncertain) =           m * q_uncertain
```

`m = sigmoid(mention_logits)`, `q = softmax(conditional_logits)`. Nhờ nhân vào
nhau, **im lặng dìm được dự đoán dương** thay vì nằm cạnh nó.

★ **Không có trọng số inverse-frequency hay kappa lâm sàng.** Điểm vận hành thuộc
về ngưỡng hiệu chỉnh sau train (project này đã làm sẵn), không thuộc về hàm hợp
lý. Điều này cũng cho nghỉ luôn bảng kappa vốn chưa có bác sĩ ký duyệt.

★ **Ô có lớp bị mask vẫn train term mention.** "Có được viết ra không" là thứ biết
được kể cả khi không biết cực tính; chỉ term lớp có điều kiện bị bỏ.

`mention_marginal_log_probs` tính toàn bộ trong log space (`logsigmoid`,
`log_softmax`, `logaddexp`). Trả về **log của xác suất biên**, nên `softmax` của
nó khôi phục đúng phân bố — evaluator, `argmax` và file `.npz` không cần đổi gì.
`Blip2Qformer` gán nó vào `classification_logits` và giữ head thô ở
`conditional_classification_logits` + `mention_logits` để chẩn đoán.

Pinned bởi `tests/test_mention_gate.py` — đặc biệt
`test_silence_suppresses_a_confident_positive` (conditional hét Positive ở logit
+6, mention −8 → P(Positive) < 0.001, dự đoán Negative) và
`test_unmentioned_target_trains_only_the_mention_head` (gradient của conditional
đúng bằng 0).

⚠ **Chưa chạy trên GPU.** Prod để `lambda_mention_conditioned_cls: 0.0`.

### `AbnormalitySpecificLoss`
Ba loss phụ giữ expert token lành mạnh:
- `orthogonality_loss` (`:272`) — giữ 14 expert token trực giao nhau
- `compute_weighted_sparsity_loss` (`:298`) — ép attention tập trung
- contrastive (trong `forward` `:338`)

## Important configuration

| Key | Ảnh hưởng |
|---|---|
| `mhcac.class_weights` | 14×3; `[]` tắt |
| `mhcac.label_smoothing` | prod 0.05 |
| `mhcac.uncertain_policy` | prod `ignore_uncertain` |
| `mhcac.distill_temperature` | prod 2.0 |
| `loss.lambda_*` | Trọng số khi tổng hợp (ở `blip2_qformer`, không ở đây) |

## Calls

`torch.nn.functional` (`cross_entropy`, `kl_div`, `log_softmax`, `normalize`).

## Called by

`model/lavis/models/blip2_models/blip2_qformer.py:35` (import) và 4 điểm gọi ·
`mhcac/mhcac_12.py` (`AbnormalitySpecificLoss` qua `self.expert_loss`) ·
`tests/test_stage1_objectives.py`, `tests/test_multiview_losses.py`

## Data flow

```text
logits [B,14,3] + labels [B,14] + sample_mask [B]  ──► scalar loss
student/teacher logits [B,14,3] + mask             ──► scalar KL
anchor [B,P,D] + aux [B,N,P,D] + aux_mask [B,N]    ──► scalar contrastive
fused/anchor logits [B,14,3] + has_aux [B]         ──► scalar consistency
```

## Side effects

Không. Hàm thuần (trừ dropout khi training).

## Error / edge cases

- `sample_mask` không có mẫu nào → trả 0 **nối vào đồ thị autograd** (không crash,
  không treo DDP).
- `uncertain_policy` không hợp lệ → ⚠ cần runtime verification.

## Related tests

`tests/test_stage1_objectives.py` — `ClassificationLoss` với mask, `soft_target_kl_loss`
`tests/test_multiview_losses.py` — `MultiPositiveContrastiveLoss`, `view_consistency_loss`

## Related documentation

[ARCHITECTURE.md §2.6](../_meta/ARCHITECTURE.md#26-tổng-hợp-loss) · [`mhcac_12.py`](mhcac_12.py.doc.md)

## Developer notes

1. **`sample_mask` là hợp đồng.** Mọi loss mới ở đây nên nhận nó, nếu không dòng
   thiếu dữ liệu sẽ âm thầm đóng góp nhiễu.
2. **Teacher phải `detach()`** trong `soft_target_kl_loss`. Bỏ đi = teacher học
   ngược từ student, phá vỡ ý nghĩa distillation.
3. ⚠ `AttentionLoss` (`:439`) không có caller. Ghi nhận, không xóa.
4. Trọng số `lambda_*` **không** nằm ở file này — chúng ở `blip2_qformer.forward:974`.

## Source relationships

- **Parent:** [`mhcac/_index.md`](_index.md)
- **Methods:** [`loss.py.methods/`](loss.py.methods/)
- **Related:** [`mhcac_12.py`](mhcac_12.py.doc.md) · [`blip2_qformer.py`](../model/lavis/models/blip2_models/blip2_qformer.py.doc.md)

← [HOME](../../HOME.md)
