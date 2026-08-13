> Source: `mhcac/mhcac_12.py:208-319`
> Status: ✅ ACTIVE

# `AbnormalityClassificationModel.__init__(...)`

## Located in

[`mhcac_12.py`](../../mhcac_12.py.doc.md)

## Purpose
Dựng expert token, alignment, pos-enc, 6 lớp attention, 14 classifier, và loss phụ.

## Signature (như gọi ở `blip2_qformer.py:342`)
```python
AbnormalityClassificationModel(
    embed_dim=768, num_abnormalities=14, num_classes=3, num_layers=6,
    num_commmon_tokens=14,          # ⚠ typo: BA chữ m
    initial_expert_tokens=None, visual_dim=1408,
    text_dropout_rate=<mhcac.text_dropout>, use_cnn=<use_biovil>,
    uncertain_policy=<mhcac.uncertain_policy>)
```

## Parameters
| Tham số | Ý nghĩa |
|---|---|
| `embed_dim=768` | Chiều làm việc nội bộ |
| `visual_dim=1408` | Chiều token vào — **phải khớp `VISUAL_DIM`** |
| `num_abnormalities=14` / `num_classes=3` | 14 bệnh lý × P/N/U |
| `num_layers=6` | Số lớp cross-attention |
| `num_commmon_tokens=14` | ⚠ typo trong API công khai |
| `use_cnn` | Bật `cnn_downsampler` cho luồng biovil |
| `text_dropout_rate` | Dropout đường text (teacher) |
| `uncertain_policy` | Truyền xuống loss |

## Sub-module dựng
| Attribute | Class |
|---|---|
| `expert_tokens` | `nn.Parameter [14, embed_dim]` |
| `embedding_alignment` | `CrossModalEmbeddingAlignment` |
| `pos_enc` | `TrainablePositionalEncoding` — ★ `target_patch_count` đọc từ đây |
| `cnn_downsampler` | `DownsamplePatches` (chỉ khi `use_cnn`) |
| `attention_layers` | 6 × `ExpertTokenCrossAttention` |
| `expert_token_norm` | LayerNorm cho residual lớp áp chót |
| `classifiers` | 14 × `nn.Linear(embed_dim, 3)` |
| `expert_loss` | `AbnormalitySpecificLoss` |
| `capture_streams` | `False` mặc định — cờ opt-in cho Grad-CAM |
| `_last_cam_streams` | `None` mặc định — state tạm, không phải buffer |

## ★ `expert_to_text_attention` là teacher-gate
Chỉ một số lớp có nhánh này (gated bởi `num_text_teacher_layers` ⚠ đọc code để
biết chính xác lớp nào). `forward` kiểm `layer.expert_to_text_attention is not None`
để quyết định có truyền text không.

## Side effects
Cấp phát tham số và khởi tạo state capture rỗng. Hai attribute capture không đi
vào `state_dict` và không đổi checkpoint compatibility.

## Modification risk
| Sửa | Ảnh hưởng |
|---|---|
| `embed_dim` / `visual_dim` | State dict đổi → checkpoint cũ không load |
| `num_layers` | Đổi lớp nào cộng residual (`num_layers-2`) |
| Sửa typo `num_commmon_tokens` | **Breaking change** cho caller |
