> Source: `model/lavis/models/blip2_models/blip2_qformer.py:1330-1489`
> Status: ✅ ACTIVE

# `Blip2Qformer.from_config(cfg)`

## Located in

[`blip2_qformer.py`](../blip2_qformer.py.doc.md)

## Purpose
YAML → tham số constructor. Đây là **cầu duy nhất** giữa config và model.

## ⚠ Nó âm thầm bỏ qua key lạ
Comment `:1367`: *"from_config silently drops unknown config blocks, so these keys
must be read explicitly to take effect."*

**Thêm key vào YAML mà quên thêm dòng đọc ở đây = key không làm gì cả, không cảnh
báo.** Đây là bẫy lớn nhất khi thêm tính năng.

## Signature
```python
@classmethod
def from_config(cls, cfg) -> Blip2Qformer
```

## Local variables — `cfg_bool` (`:1315`)
Helper chấp nhận `None` (→ default), chuỗi (`"1"/"true"/"yes"/"y"/"on"`), hoặc bool.
Nhờ vậy `--options model.encoders.swin=true` (chuỗi) và YAML `swin: true` (bool)
đều hoạt động.

## Cấu trúc đọc — fallback nhiều tầng
```python
use_biovil = cfg_bool(encoders.get("biovil", cfg.get("use_biovil", vit_model == "biovil")))
```
Ba tầng: `encoders.biovil` → `use_biovil` phẳng → suy từ `vit_model`. Cho phép
config cũ vẫn chạy.

Tương tự cho swin (`:1332-1352`) và raddino (`:1353-1364`): mỗi tham số đọc từ
`model.swin.*` **hoặc** key phẳng `swin_*`.

## Các khối được đọc
| Khối YAML | Dòng | Ghi chú |
|---|---|---|
| `encoders.*` | 1327-1353 | |
| `swin.*` / `raddino.*` | 1332-1364 | |
| `multi_view`, `view_fusion.*` | 1369-1377 | ★ 6 key đọc **tường minh** |
| `loss.lambda_*`, `loss.itc_queue_size` | 1379-1394 | |
| `mhcac.*` | 1398-1405 | |
| `num_query_token`, `cross_attention_freq`, `max_txt_len`, `freeze_vit`, `image_size`, `vit_precision`, `drop_path_rate` | 1299-1313 | |

## Execution flow
```text
đọc mọi key (có default)
   ↓
model = cls(**tất cả tham số)      ← :1408-1440
   ↓
model.load_checkpoint_from_config(cfg)   ← :1442, nạp BLIP-2 pretrained
   ↓
return model
```

## Side effects
Dựng toàn bộ model; tải pretrained weights (mạng).

## Config dependencies
Toàn bộ khối `model:` của run YAML.

## Called by
`task.build_model(cfg)` qua registry — từ `pretraining/train.py:161` và
`training/stage1/lavis_loader.build_stage1_model`.

## Tests
Không có test trực tiếp (cần torch + weights).

## Modification risk
| Sửa | Ảnh hưởng |
|---|---|
| Thêm tham số constructor mà quên đọc ở đây | Tham số luôn dùng default, YAML bị bỏ qua |
| Đổi default | Ảnh hưởng mọi config không nêu key đó |
| Bỏ `load_checkpoint_from_config` | Model train từ đầu, không có BLIP-2 pretrained |
