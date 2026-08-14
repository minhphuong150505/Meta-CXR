> Source: `model/lavis/models/blip2_models/blip2_qformer.py:542-575`
> Status: ✅ ACTIVE — thêm 2026-08-14

# `Blip2Qformer._native_stream_layouts(img_size)`

## Located in

[`blip2_qformer.py`](../../blip2_qformer.py.doc.md)

## Purpose

Trả `dict[str, StreamLayout]` mô tả chuỗi token **gốc** của từng encoder đang
bật, hoặc `None` để MHCAC dùng nhánh legacy. Gọi một lần trong `__init__`, kết
quả truyền thẳng vào `AbnormalityClassificationModel(stream_layouts=...)`.

## Signature

```python
def _native_stream_layouts(self, img_size) -> dict[str, StreamLayout] | None
```

## Giá trị trả về ở cấu hình production

| Stream | `num_tokens` | `num_global_tokens` | Nguồn |
|---|---|---|---|
| `biovil` | 196 | 0 | `(img_size // 32) ** 2` — ResNet50 stride tổng 32 |
| `pubmedclip` | 50 | 1 | `(vision_config.image_size // patch_size) ** 2 + 1` |

## ★ Vì sao có thể trả `None`

Swin và RadDINO không phơi số token ở đây. Một layout **thiếu** encoder đang bật
sẽ để stream đó không có positional encoding, nên hàm trả `None` cho cả cụm và
mọi thứ quay về hành vi cũ (resize hết về `target_patch_count`). Không bao giờ
dựng model với layout một phần.

## ★ Bẫy `img_size`

`model.image_size` **phải** bằng `datasets.mimic_cxr.vis_processor.*.image_size`.
Điều này không hiển nhiên: `init_vision_encoder` bỏ qua `img_size` với biovil,
nên suốt thời gian dài không ai đọc giá trị này và mặc định **224** của YAML
BLIP-2 gốc nằm đó trong khi vis_processor sinh **448**. Lỗi lộ ra ngay ở smoke
đầu tiên sau khi hàm này được thêm:

```text
ValueError: stream 'biovil' carries 196 tokens but its layout declares 49
```

Hai lớp bảo vệ, cả hai đều fail loud:

1. `img_size is None` → `ValueError` nêu rõ phải đặt `model.image_size`.
2. MHCAC `forward` đối chiếu `num_tokens` với span thật → `ValueError`.

`pretraining/configs/mimic_cxr_full.yaml` nay khai báo `model.image_size: 448`
tường minh.

## Calls / Called by

Gọi: `StreamLayout` (`mhcac/mhcac_12.py`), `self.pubmedclip.model.config`.
Được gọi bởi: `Blip2Qformer.__init__` (`:399`), đúng một lần.

## Modification risk

| Sửa | Ảnh hưởng |
|---|---|
| Trả `None` thay vì layout | MHCAC mất `pos_enc` ModuleDict và dựng lại `cnn_downsampler` → **state dict đổi, checkpoint không load** |
| Đổi trunk BioViL (stride ≠ 32) | Layout sai; MHCAC raise ở forward đầu tiên chứ không train sai âm thầm |
| Bật swin/raddino | Tự động về legacy — kiến trúc khác hẳn, đừng so kết quả trực tiếp |

## Source relationships

- **Parent:** [`Blip2Qformer/_index.md`](_index.md)
- **Related:** [`mhcac_12.py`](../../../../../../mhcac/mhcac_12.py.doc.md)

← [HOME](../../../../../../../HOME.md)
