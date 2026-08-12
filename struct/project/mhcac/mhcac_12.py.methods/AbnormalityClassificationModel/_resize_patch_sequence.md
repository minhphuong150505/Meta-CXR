> Source: `mhcac/mhcac_12.py:317-353`
> Status: ✅ ACTIVE

# `AbnormalityClassificationModel._resize_patch_sequence(patches)`

## Located in

[`mhcac_12.py`](../../mhcac_12.py.doc.md)

## Purpose
Đưa số patch của một encoder về `target_patch_count` chung.

## ★ Vì sao cần
Ba encoder cho số patch khác nhau (BioViL-T, PubMedCLIP, SwinV2 đều khác). MHCAC
nối chúng lại và cấp positional encoding — pos-enc có kích thước cố định, nên mọi
stream phải cùng độ dài.

Đây là chỗ **Swin từng gây shape mismatch**; notebook legacy `03` từng vá
`mhcac_12.py` bằng string replacement lúc runtime
(`ensure_swin_mhcac_shape_patch`, dòng 843–874).

## Signature
```python
def _resize_patch_sequence(self, patches) -> Tensor
```
`[B, P, D]` → `[B, target_patch_count, D]`

## Execution flow
```text
P == target_patch_count → trả nguyên
   ↓
grid_size = int(P ** 0.5) ; target_size = int(target_patch_count ** 0.5)
   ↓
CẢ HAI là số chính phương?
   ✓ → view(B, grid, grid, D).permute → adaptive_avg_pool2d(target_size, target_size)
        → permute lại → reshape(B, target_patch_count, D)      ← giữ cấu trúc 2-D
   ✗ → F.interpolate(transpose(1,2), size=target, mode="linear") → transpose lại
```

## Detailed logic
**Hai đường có ý nghĩa khác nhau:**
- **Adaptive avg-pool 2-D** hiểu patch là lưới không gian — gộp vùng lân cận, giữ
  quan hệ giải phẫu.
- **Nội suy tuyến tính 1-D** coi patch là chuỗi — dùng khi số patch không phải số
  chính phương, chấp nhận mất cấu trúc 2-D.

Đường 2 là fallback, không phải lựa chọn ưu tiên.

## Local variables
`target_patch_count` — ⚠ đọc từ `self.pos_enc.positional_encoding.size(1)`, **giá
trị cụ thể cần runtime verification**.

## Data / Tensor flow
```text
[B, P, D] ─ chính phương ─► [B,grid,grid,D] ─► pool ─► [B,target,D]
          └ khác ─────────► [B,D,P] ─► interpolate ─► [B,target,D]
```

## Side effects
Không.

## Called by
`AbnormalityClassificationModel.forward` — cho **mọi stream trừ biovil** (biovil
dùng `cnn_downsampler` khi `use_cnn`).

## Tests
Gián tiếp qua `tests/test_stage1_objectives.py`.

## Modification risk
Đổi hàm này đổi cách thông tin không gian được nén → ảnh hưởng chất lượng phân
loại, và **không tương thích checkpoint cũ** nếu `target_patch_count` đổi.
