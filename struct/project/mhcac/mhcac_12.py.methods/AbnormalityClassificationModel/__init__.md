> Source: `mhcac/mhcac_12.py:234-417`
> Status: ✅ ACTIVE

# `AbnormalityClassificationModel.__init__(...)`

## Located in

[`mhcac_12.py`](../../mhcac_12.py.doc.md)

## Purpose
Dựng expert token, alignment, pos-enc, 6 lớp attention, 14 classifier, và loss phụ.

## Signature (như gọi ở `blip2_qformer.py:388`)
```python
AbnormalityClassificationModel(
    embed_dim=768, num_abnormalities=14, num_classes=3, num_layers=6,
    num_commmon_tokens=14,          # ⚠ typo: BA chữ m
    initial_expert_tokens=None, visual_dim=1408,
    text_dropout_rate=<mhcac.text_dropout>, use_cnn=<use_biovil>,
    uncertain_policy=<mhcac.uncertain_policy>,
    stream_layouts=<Blip2Qformer._native_stream_layouts(img_size)>)
```

## Parameters
| Tham số | Ý nghĩa |
|---|---|
| `embed_dim=768` | Chiều làm việc nội bộ |
| `visual_dim=1408` | Chiều token vào — **phải khớp `VISUAL_DIM`** |
| `num_abnormalities=14` / `num_classes=3` | 14 bệnh lý × P/N/U |
| `num_layers=6` | Số lớp cross-attention |
| `num_commmon_tokens=14` | ⚠ typo trong API công khai |
| `use_cnn` | Cho phép dựng `cnn_downsampler` — nhưng **bị `stream_layouts` phủ quyết** |
| `stream_layouts` | `dict[str, StreamLayout]` hoặc `None`. Có → mỗi encoder giữ chuỗi token gốc và có pos-enc riêng; `None` → hành vi legacy |
| `text_dropout_rate` | Dropout đường text (teacher) |
| `uncertain_policy` | Truyền xuống loss |

## Sub-module dựng
| Attribute | Class |
|---|---|
| `expert_tokens` | `nn.Parameter [14, embed_dim]` |
| `embedding_alignment` | `CrossModalEmbeddingAlignment` |
| `pos_enc` | `nn.ModuleDict[name → TrainablePositionalEncoding]` khi có layout; một `TrainablePositionalEncoding` dùng chung khi không |
| `cnn_downsampler` | `DownsamplePatches` — chỉ khi `use_cnn` **và** `"biovil" not in stream_layouts` |
| `stream_layouts` | `dict` đã copy; rỗng nghĩa là nhánh legacy |
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
| `stream_layouts` | Đổi số pos-enc và sự tồn tại của `cnn_downsampler` → **state dict đổi, checkpoint cũ không load**. Truyền layout thiếu một encoder đang bật sẽ khiến stream đó không có pos-enc → caller phải trả `None` thay vì layout một phần |
| `TrainablePositionalEncoding(std=)` | Chỉ đổi khởi tạo, không đổi shape. `0.02` giữ pos-enc cùng bậc với token đã L2-normalize |
