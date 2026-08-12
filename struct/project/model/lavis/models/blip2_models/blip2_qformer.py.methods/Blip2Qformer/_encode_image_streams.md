> Source: `model/lavis/models/blip2_models/blip2_qformer.py:498-572`
> Status: ✅ ACTIVE

# `Blip2Qformer._encode_image_streams(...)`

## Located in

[`blip2_qformer.py`](../../blip2_qformer.py.doc.md)

## Purpose
Encoder → view fusion → shared projector. Nơi duy nhất ảnh biến thành
`SharedVisualTokens`.

## Signature
```python
def _encode_image_streams(self, image, apply_aug=False, cached=None,
                          aux_image=None, aux_cached=None, aux_mask=None,
                          anchor_view_id=None, aux_view_ids=None) -> SharedVisualTokens
```

## Parameters
| Tham số | Type | Ý nghĩa |
|---|---|---|
| `image` | `[B,3,448,448]` | Anchor |
| `apply_aug` | bool | ⚠ Luôn `False` ở cả hai caller — augmentation làm trong dataset |
| `cached` | dict | Raw feature đã precompute; thay forward encoder |
| `aux_image` | `[B,N,3,448,448]` | Auxiliary view |
| `aux_cached` | dict | Cache cho aux |
| `aux_mask` | `[B,N]` bool | |
| `anchor_view_id` / `aux_view_ids` | long | Cho view embedding |

## Returns
`SharedVisualTokens(tokens=[B,ΣP,1408], spans={...})`

## Local variables
| Biến | Ý nghĩa |
|---|---|
| `raw_streams` | dict tên→tensor **trước** merge |
| `aux_streams` | Chỉ encode khi `fuse_on and has_aux_input` |
| `has_aux_input` | ★ `N_max == 0` → **bỏ qua encode aux hoàn toàn** |
| `self._last_prefusion_streams` | Reset mỗi lần gọi; chỉ điền khi `_keep_prefusion` |

## Execution flow
```text
reset _last_prefusion_streams, _last_raddino_patches
   ↓
has_aux_input? → _encode_aux_streams(aux_image, aux_cached)     [no_grad]
   ↓
FOR mỗi encoder bật:
   raw = cached[name]  HOẶC  encoder(image)
   _stash_prefusion(name, raw, aux_streams)      ← chỉ khi λ_mpc>0 hoặc λ_vc>0
   raw = _fuse(name, raw, aux_streams, ...)      ← ViewFusionModule[name]
   biovil: raw_streams["biovil"] = ln_vision(raw)   ← chuẩn hóa, KHÔNG phải chiếu
   khác  : raw_streams[name] = raw
   ↓
raw_streams rỗng → ValueError
   ↓
return shared_visual_projector(raw_streams)
```

## Detailed logic
**`ln_vision` chỉ cho biovil và nằm phía encoder của merge** (comment `:507`): nó
là chuẩn hóa chứ không phải phép chiếu chiều, nên thuộc về encoder chứ không phải
projector.

**PubMedCLIP bỏ head `mlp` riêng** (comment `:519`): projector chung sở hữu phép
chiếu cho mọi luồng như nhau.

**`_stash_prefusion` chỉ giữ tensor khi có loss phụ dùng** (`_keep_prefusion`,
`:300`) — nếu không, giữ tham chiếu chỉ tốn bộ nhớ.

## Side effects
Ghi `self._last_prefusion_streams`, `self._last_raddino_patches`.

## Error handling
`raw_streams` rỗng → `ValueError("No image encoder stream is enabled.")` (`:539`)

## Config dependencies
`model.encoders.*`, `model.multi_view`, `run.feature_cache_dir`

## Tests
`tests/test_shared_visual_tokens.py`, `tests/test_view_fusion.py`

## Modification risk
Đổi thứ tự fuse ↔ project sẽ đổi ngữ nghĩa **và** làm checkpoint không tương thích.
Fusion **phải** chạy trên output thô.
