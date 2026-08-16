> Source: `vision_encoders/`
> Status: ✅ ACTIVE (một phần)
> Last verified against source: 2026-08-16

# `vision_encoders/`

## Purpose

Các vision encoder bổ sung ngoài BioViL-T, cộng **điểm chiếu duy nhất** hợp nhất
đầu ra của mọi encoder thành một biểu diễn thị giác chung.

## ⚠ Bẫy lớn nhất: hai bản BioViL-T

```text
biovil_t/                    ← ✅ BẢN ĐƯỢC IMPORT (mọi `from biovil_t.* import`)
vision_encoders/biovil_t/    ← 🕰 BẢN SAO, KHÔNG AI DÙNG
```

Sửa file trong `vision_encoders/biovil_t/` **không có tác dụng gì** và rất khó
debug. Kiểm chứng:

```bash
grep -rn "vision_encoders.biovil_t" --include='*.py' . | grep -v '^./vision_encoders/biovil_t/'
# → rỗng
```

## Role in project

```text
image → [encoder 1..n] → SharedVisualTokenProjector → SharedVisualTokens
                                                        ├─► MHCAC
                                                        └─► Q-Former
```

`shared_visual_tokens.py` là **thứ quan trọng nhất trong thư mục này**, dù nó
không phải encoder. Trước khi có nó, MHCAC và Q-Former mỗi bên tự chiếu đặc trưng
và có thể trôi dạt sang hai biểu diễn khác nhau.

## Parent

[`struct/project/`](../../HOME.md#source-code-tree)

## Children

| Đường dẫn | Doc | Status | Chiều ra | Ghi chú |
|---|---|---|---|---|
| `shared_visual_tokens.py` | [📄](shared_visual_tokens.py.doc.md) | ✅ ★ | → 1408 | `SharedVisualTokens`, `SharedVisualTokenProjector` |
| `stream_adapter.py` | [📄](stream_adapter.py.doc.md) | 🟡 | giữ `D` | `StreamAdapter` (đường chính, identity ở init), `ContrastiveProjectionHead`, `pool_stream`. Lý do tồn tại: MPC từng có **gradient bằng 0** |
| `pubmedclip/pubmed_clip.py` | [📄](pubmedclip.py.doc.md) | ✅ | 768 | Dựng với `project=False` — projector sở hữu phép chiếu |
| `swin/swin_encoder.py` | [📄](swin_encoder.py.doc.md) | ✅ | `embed_dim` ⚠ runtime | `ChayanM/SwinV2-GPT2_Mimic` |
| `rad_dino/rad_dino_encoder.py` | [📄](rad_dino_encoder.py.doc.md) | 🟡 | `embed_dim` ⚠ runtime | `microsoft/rad-dino`; `raddino: false` ở **mọi** config |
| `biovil_t/` (8 file) | — | 🕰 | — | **Bản sao.** Xem cảnh báo trên |
| `medclip/medclip.py` | — | 🕰 | — | Import bị comment `blip2_qformer.py:30`, `:286` |
| `__init__.py` | — | ✅ | — | |

## Main responsibilities

1. **Trích đặc trưng** từ ảnh bằng backbone đóng băng.
2. **Hợp nhất** mọi luồng về `VISUAL_DIM = 1408` và nối theo trục token, kèm
   `spans` để hạ nguồn biết token nào thuộc encoder nào.

## Entry points

Không có. Thư viện.

## Dependencies

`torch`, `transformers` (Swin, RadDINO), `open_clip`/`transformers` (PubMedCLIP)
⚠ cần runtime verification cho chính xác package.

`medclip/medclip.py:3` import package `medclip` bên ngoài — **không có trong
requirements nào**. Đó là một lý do nữa để coi nó là legacy.

## Used by

| Ai | Import gì |
|---|---|
| `model/lavis/models/blip2_models/blip2_qformer.py` | `:26` Pubmedclip · `:27` SwinEncoder · `:28` RadDinoEncoder · `:29` SharedVisualTokenProjector |
| `tests/test_shared_visual_tokens.py` | `shared_visual_tokens` |
| `pretraining/precompute_features.py` | gián tiếp, qua model |

## Execution flow

```text
Blip2Qformer._encode_image_streams()
   │
   ├─ biovil     → visual_encoder(image).projected_patch_embeddings → ln_vision
   ├─ pubmedclip → self.pubmedclip(image, apply_aug=False)[0]
   ├─ swin       → self.swin(image)
   └─ raddino    → self.raddino(image)          (nếu bật)
        │
        ▼  (mỗi luồng đi qua ViewFusionModule của nó trước, nếu multi_view)
   SharedVisualTokenProjector(raw_streams)
        ▼
   SharedVisualTokens(tokens=[B, ΣP, 1408], spans={...})
```

**Thứ tự stream cố định:** `CANONICAL_STREAM_ORDER = ("biovil", "pubmedclip", "swin", "raddino")`
(`shared_visual_tokens.py:29`) — để `spans` ổn định giữa các lần chạy, kể cả khi
dict đầu vào có thứ tự khác.

## Important configurations

```yaml
model:
  encoders:
    biovil: true
    pubmedclip: true
    swin: true
    raddino: false        # ← tắt ở MỌI config
  swin:
    backend: hf
    model_name: "ChayanM/SwinV2-GPT2_Mimic"
    pretrained: true
    frozen: true
    normalize: true
  raddino:
    model_name: "microsoft/rad-dino"
    frozen: true
    normalize: true
```

⚠ `blip2_qformer.py:173` raise nếu **không** encoder nào bật.

## Status

```text
✅ ACTIVE — pubmedclip, swin, shared_visual_tokens
🟡 CONDITIONAL — rad_dino (có wire đầy đủ, tắt theo config)
🕰 LEGACY — biovil_t/ (bản sao), medclip/
```

> **RadDINO không phải dead code.** Nó có đủ đường dữ liệu: `blip2_qformer.py:274`
> (khởi tạo), `:530` (encode), `:1353` (from_config), và có mặt trong
> `CANONICAL_STREAM_ORDER`. Bật một dòng config là nó chạy.

## Notes

- **PubMedCLIP được dựng với `project=False`** (`blip2_qformer.py:258`). Head
  `mlp` riêng của nó bị bỏ qua có chủ đích — `SharedVisualTokenProjector` sở hữu
  phép chiếu cho mọi luồng như nhau.

- **`ln_vision` chỉ áp cho BioViL-T** và nằm ở *phía encoder* của phép merge, vì
  nó là chuẩn hóa chứ không phải chiếu chiều.

- ⚠ **`.pyc` bị git track** trong `vision_encoders/biovil_t/`? Không — nhưng có
  trong `biovil_t/__pycache__/`. Xem [I1](../_meta/LEGACY_AND_OPTIONAL.md#-potential-issues--ghi-nhận-không-sửa).

- `SharedVisualTokens.without(name)` cho phép **ablation encoder tại thời điểm
  chạy** — zero-out một luồng mà không đổi shape. Hiện chỉ test dùng.

## Related documentation

- [ARCHITECTURE.md §2.1–2.3](../_meta/ARCHITECTURE.md#21-vision-encoders--đóng-băng)
- [`biovil_t/_index.md`](../biovil_t/_index.md) — bản BioViL-T được dùng thật
- [LEGACY_AND_OPTIONAL.md §L4, §L5](../_meta/LEGACY_AND_OPTIONAL.md#l4--vision_encodersbiovil_t--bản-sao)

← [Về HOME](../../HOME.md)
