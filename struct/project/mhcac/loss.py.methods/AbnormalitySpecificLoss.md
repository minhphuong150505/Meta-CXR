> Source: `mhcac/loss.py:241-438`
> Status: ✅ ACTIVE

# `class AbnormalitySpecificLoss(nn.Module)`

## Located in

[`loss.py`](../loss.py.doc.md)

## Purpose
Ba loss phụ giữ **expert token** của MHCAC lành mạnh, cộng pooling.

## Vì sao expert token cần được "giữ lành mạnh"
14 expert token học tự do sẽ có xu hướng **hội tụ về nhau** (cùng nhìn vào vùng
ảnh nổi bật nhất) hoặc **trải attention khắp ảnh** (không chuyên biệt hóa). Cả hai
đều làm mất ý nghĩa "mỗi token một bệnh lý".

## Constructor (`:242`)
Nhận trọng số cho từng thành phần.

## Methods
| Method | Dòng | Vai trò |
|---|---|---|
| `forward(expert_tokens, attention_weights_list, labels, sample_mask)` | 338 | ★ Trả `(pooled, orth, contrastive, sparsity)` |
| `orthogonality_loss(common_representations)` | 272 | ★ Ép 14 token **trực giao** nhau |
| `compute_weighted_sparsity_loss(attention_weights_list)` | 298 | ★ Ép attention **tập trung** |

`AttentionPooling` (`:215`) gộp attention thành `pooled_representations [B,14,D]`.

## Ba loss
| Loss | Chống lại | Prod λ |
|---|---|---|
| `orth_loss` | Token hội tụ về nhau | 0.05 |
| `sparsity_loss` | Attention trải đều khắp ảnh | 0.01 |
| `contrastive_loss` | Token không phân biệt được mẫu | 0.1 |

`contrastive_loss` cần `labels`; `Blip2Qformer.forward:931` truyền `labels=None`
cho teacher — comment `:929` ghi rõ: *"Teacher supervision is applied by
ClassificationLoss below; do not duplicate the O(B²) token contrastive calculation."*

## Called by
`AbnormalityClassificationModel.forward` qua `self.expert_loss(...)` (`mhcac_12.py:~430`)

## Returns
`(pooled_representations [B,14,D], orth_loss, contrastive_loss, sparsity_loss)` —
ba loss sau được `mhcac_12.forward` trả lên cho `Blip2Qformer`.

## Tests
`tests/test_stage1_objectives.py` (gián tiếp)

## Modification risk
Bỏ `orth_loss` → 14 token có thể hội tụ, attention map mất tính giải thích, và
phân loại per-bệnh-lý xuống cấp mà accuracy tổng vẫn có thể ổn.
