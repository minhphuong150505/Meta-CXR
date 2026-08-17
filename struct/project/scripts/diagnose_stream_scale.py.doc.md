> Source: `scripts/diagnose_stream_scale.py` (231 dòng)
> Status: 🔬 DIAGNOSTIC — read-only, không sửa logic, không train, không ghi checkpoint
> Last verified against source: 2026-08-13

# `scripts/diagnose_stream_scale.py`

## Purpose

Đo **độ mất cân bằng biên độ giữa các encoder** trong chuỗi token dùng chung, và
**tỉ lệ attention** mà MHCAC dành cho từng encoder.

Sinh ra để trả lời một câu hỏi cụ thể: Table 5 cho thấy all-three chỉ hơn
BioViL-only 0.00002 mean F1 — tại sao PubMedCLIP và SwinV2 không đóng góp gì?
`~/agent_run/span_diag.py` (chỉ đọc checkpoint) đã loại trừ giả thuyết "projection
sụp về 0": gain 0.64, weight std ≈ 0.02, hoàn toàn khỏe mạnh. Phần còn lại phụ
thuộc **activation**, nên phải chạy forward pass — đó là việc của script này.

## Đo cái gì

| Chỉ số | Lấy ở đâu | Ý nghĩa |
|---|---|---|
| Token RMS mỗi span | `SharedVisualTokens.stream(name)`, đầu ra `SharedVisualTokenProjector` | Biên độ tại **điểm nối** — tensor mà cả MHCAC lẫn Q-Former cùng đọc |
| `max\|x\|` mỗi span | như trên | Bắt outlier mà RMS che mất |
| Attention mass mỗi span | `attention_weights_list` do `mhcac(...)` trả về (giá trị thứ 2) | Tỉ lệ xác suất `expert_to_image_attention` rơi vào từng encoder, theo từng layer |

RMS gộp sum-of-squares trên **toàn bộ token của toàn bộ study**, không phải trung
bình của các trung bình theo batch.

## Kết quả đã đo (64 study, split val, checkpoint_best epoch 9)

```
stream         span tokens         RMS      max|x|   RMS ratio
swin                   144     37.7190    233.8515      1.0000
pubmedclip              50     24.2024    103.7959      0.6417
biovil                 196      1.1619     14.5355      0.0308
```

**Chênh 32 lần.** BioViL-T là stream **yếu nhất**, không phải mạnh nhất — vì nó là
stream duy nhất được LayerNorm, và LayerNorm ở *upstream* của merge
(`blip2_qformer.py:538`: `raw_streams["biovil"] = self.ln_vision(cnn_raw)`).
PubMedCLIP và SwinV2 vào thẳng chuỗi chung với biên độ thô của encoder gốc.

Attention mass thì **không** lệch theo hướng đó — dao động 0.19–0.69 quanh mức
đồng đều 0.333, không stream nào áp đảo có hệ thống. Lý do ở
`mhcac/mhcac_12.py:97`: `F.normalize(self.visual_proj(visual_tokens), dim=-1)`
đưa mọi token về unit-norm **trước** attention, nên MHCAC miễn nhiễm với chênh
lệch biên độ.

## Hệ quả — hai nhánh chịu ảnh hưởng khác nhau

| Nhánh | Đọc gì | Có chuẩn hóa? | Kết luận |
|---|---|---|---|
| MHCAC | `SharedVisualTokens` | ✅ `F.normalize` per-token | Miễn nhiễm scale. Việc nó bỏ qua PubMedCLIP/Swin **không phải** do mất cân bằng biên độ |
| Q-Former | `shared_visual.tokens` thô (`blip2_qformer.py:873`) | ❌ **không** | Nhận chuỗi lệch 32 lần |

⚠ Đây là **giả thuyết chưa kiểm chứng**, không phải kết luận: mất cân bằng ở
nhánh Q-Former có thể là lý do ITC và ITM đứng đúng mức ngẫu nhiên trong run
10-epoch (`val loss_itc` 1.7918 = ln(6) với batch eval 6; `train loss_itc` 6.9375
≈ ln(1024) với queue 1024; `loss_itm` 0.6365 so với ln(2) = 0.693). Cần thí
nghiệm riêng để xác nhận.

Còn **chưa giải thích được**: MHCAC attend đáng kể vào cả ba stream nhưng zero hai
stream kia lại không đổi kết quả. Nghĩa là nội dung của chúng không mang thông tin
phân biệt nhãn mà classifier dùng — nguyên nhân khác với scale.

## Entry points

```bash
# Chạy TRÊN phuong@phuong-b760m-pro-rs-d4-wifi, không phải checkout dev
CUDA_VISIBLE_DEVICES=0 python scripts/diagnose_stream_scale.py --num-studies 64
```

| Flag | Mặc định | Ghi chú |
|---|---|---|
| `--cfg-path` | `pretraining/configs/ablation/all_three.yaml` | Config đặt `load_finetuned` và bật cả ba encoder |
| `--split` | `val` | **test giữ nguyên ngoài vòng lặp** |
| `--num-studies` | 32 | Vài chục là đủ; RMS ổn định, attention mass thì nhiễu |
| `--batch-size` | 4 | |
| `--device` | tự dò | cuda nếu có |
| `--json-out` | — | Ghi số đo ra JSON |
| `--options` | — | Override config, cú pháp như `pretraining/train.py` |

## Dependencies

| Phụ thuộc | Dùng để làm gì |
|---|---|
| [`model/lavis/`](../model/lavis/_index.md) | `Config`, `tasks.setup_task`, `MIMIC_CXR_Dataset` — cùng đường load với `pretraining/train.py` |
| [`local_config.py`](../local_config.py.doc.md) | `VIS_ROOT` |
| `torch` | forward pass, `DataLoader` |

Import `torch`/LAVIS được **hoãn vào trong hàm**, nên `--help` chạy được trên box
CPU không có torch.

## Used by

Không có caller tự động. Chạy tay khi cần chẩn đoán.

## Notes

- **Read-only, có chủ ý.** Không sửa module nào, không ghi checkpoint, không train.
  Chỉ đọc tensor trên đường forward.
- Gọi `model._encode_image_streams(...)` — method private. Cố ý: đó đúng là tensor
  hai nhánh downstream tiêu thụ; dựng lại encode ở đây sẽ có nguy cơ đo một thứ mà
  model không bao giờ thấy. Chạy trong `eval()` + `no_grad()`.
- Gọi `model.mhcac(shared, text_embeddings=None, labels=None)` — không text, không
  label, nên đi đúng nhánh **student** (nhánh chạy lúc inference) và không tính loss.
- Attention mass nhiễu theo mẫu: 8 study và 64 study cho phân bố khác nhau rõ rệt.
  Đừng kết luận từ mẫu nhỏ. RMS thì ổn định.
- Script **không diễn giải**. Nó in số; kết luận nằm ở người đọc.

## Tests

Chưa có test riêng. Là script chẩn đoán một lần, không nằm trên đường production.

## Related documentation

- [`vision_encoders/shared_visual_tokens.py`](../vision_encoders/shared_visual_tokens.py.doc.md) — chỗ merge, nơi thiếu chuẩn hóa
- [`mhcac/mhcac_12.py`](../mhcac/mhcac_12.py.doc.md) — `F.normalize` per-token
- [`results/`](../results/_index.md) — Table 5 encoder ablation
- [ARCHITECTURE.md](../_meta/ARCHITECTURE.md) · [LEGACY_AND_OPTIONAL.md](../_meta/LEGACY_AND_OPTIONAL.md)

← [📁 scripts](_index.md) · [HOME](../../HOME.md)
