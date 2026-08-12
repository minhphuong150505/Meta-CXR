> Source: `vision_encoders/rad_dino/rad_dino_encoder.py` (92 dòng)
> Status: 🟡 CONDITIONAL — tắt ở mọi config
> Last verified against source: 2026-08-12

# `vision_encoders/rad_dino/rad_dino_encoder.py`

## Purpose

Bọc `microsoft/rad-dino` (DINOv2 y khoa) thành encoder đóng băng.

## ⚠ Không phải dead code

Nó có **đủ đường dữ liệu** trong `blip2_qformer.py`:

| Vị trí | Việc |
|---|---|
| `:28` | import |
| `:141-144` | tham số `use_raddino`, `raddino_model_name`, `raddino_frozen`, `raddino_normalize` |
| `:274` | khởi tạo |
| `:284` | `raddino_dim` |
| `:313`, `:336` | vào `stream_dims` / `shared_stream_dims` |
| `:450` | encode aux |
| `:530-536` | encode anchor + fuse + `_last_raddino_patches` |
| `:1353` | `from_config` |

Cộng có mặt trong `CANONICAL_STREAM_ORDER`. **Bật một dòng config là nó chạy.**

Nhưng mọi YAML hiện tại đặt `raddino: false`.

## Status

```text
🟡 CONDITIONAL — encoders.raddino: false ở mimic_cxr_full_l4, 2x3090, 2gpu, blip2_*, 07_all_three
```

## Main class

`RadDinoEncoder(nn.Module)` (`:15`)

| Method | Dòng |
|---|---|
| `__init__(model_name, pretrained, frozen, normalize)` | 21 |
| `train(mode=True)` | 73 |
| `forward(x)` | 79 |

`self.embed_dim` ⚠ runtime verification.

## Important configuration

```yaml
model:
  encoders: {raddino: true}       # ← mặc định false
  raddino:
    model_name: "microsoft/rad-dino"
    frozen: true
    normalize: true
```

## Calls / Called by

Gọi: `transformers` — `:34` raise `ImportError` có thông điệp rõ nếu thiếu.
Được gọi: `blip2_qformer.py` (xem bảng trên); `tests/test_shared_visual_tokens.py:192`
dùng stream `raddino` giả lập.

## Error / edge cases

Thiếu `transformers` → `ImportError` nêu rõ package cần cài (`:34`).

## Related tests

`tests/test_shared_visual_tokens.py:111,113,192` — `stream("raddino")` và
`without("raddino")` raise khi stream không có mặt.

## Developer notes

1. **Bật RadDINO đổi `ΣP` và `stream_dims`** → state dict của
   `SharedVisualTokenProjector` và `ViewFusionModule` đổi theo. Checkpoint cũ
   **không load được**.
2. `_last_raddino_patches` (`blip2_qformer.py:324`, `:535`) là instance state được
   giữ lại — ⚠ chưa tìm thấy consumer trong đường chính. Có thể là móc cho debug.

## Source relationships

- **Parent:** [`vision_encoders/_index.md`](_index.md)
- **Related:** [`shared_visual_tokens.py`](shared_visual_tokens.py.doc.md)

← [HOME](../../HOME.md)
