> Source: `vision_encoders/stream_adapter.py`
> Status: 🟡 CONDITIONAL — chỉ được dựng khi `multi_view: true`; head chỉ khi `lambda_mpc > 0`
> Last verified against source: 2026-08-16

# `vision_encoders/stream_adapter.py`

## Purpose

Cấp **năng lực học được** đặt lên trên đặc trưng của encoder đã đóng băng, để
`MultiPositiveContrastiveLoss` có thứ để huấn luyện.

## Why it exists — loss chết suốt một run thật

`MultiPositiveContrastiveLoss` mang `lambda_mpc = 0.1` (≈22% giá trị loss được
log) và **dạy được số không**. Nó tính trên tensor stash **trước** module trainable
duy nhất, mà tensor đó là đầu ra thô của encoder đóng băng, còn nhánh aux thì nằm
trong `torch.no_grad()`. Không vế nào có tham số ở thượng nguồn.

Đo trên run thật, bốn epoch liên tiếp:

```text
3.9929   3.9949   3.9941   3.9943      ← hằng số, dao động ±0.001
```

Sau khi thêm adapter + head, cùng công thức loss, smoke 4 epoch:

```text
4.0875 → 3.7786 → 3.3919 → 3.0873      ← giảm đơn điệu
```

## Main classes / functions

| Tên | Vai trò |
|---|---|
| `StreamAdapter` | Khối dư nhỏ trên **đường chính**, mỗi encoder một cái |
| `ContrastiveProjectionHead` | `g(.)` kiểu SimCLR, **chỉ dùng lúc train** |
| `pool_stream` | Gộp chuỗi token của một encoder thành một vector |

## `StreamAdapter`

```text
h = x + up(GELU(down(x)))        down: Linear(D, D/4)   up: Linear(D/4, D)
```

`up` được **zero-init** cả weight lẫn bias → khối là **identity chính xác ở step 0**,
đúng thủ thuật `ViewFusionBlock` đang dùng và cùng một lý do: bật nó lên không được
làm nhiễu một model vừa dựng.

★ **Vị trí là điều quan trọng nhất.** Ngay sau đầu ra encoder đóng băng, **trước cả**
điểm stash `_last_prefusion_streams` **và** `ViewFusionModule`. Áp cùng trọng số cho
anchor và **mọi** aux view.

⚠ Gắn MPC **sau** fusion sẽ tự phá mục đích: anchor đã fuse thì đã chứa sẵn aux view,
nên "hai view giống nhau" trở thành hiển nhiên.

⚠ **Đây là tham số trên đường INFERENCE.** Checkpoint train trước thay đổi này
**không resume được** vào model có adapter.

## `ContrastiveProjectionHead`

```text
LayerNorm(D) → Linear(D, 512) → GELU → Linear(512, 256) → L2 normalize
```

Một head cho mỗi stream, dùng chung giữa anchor và aux. Tách khỏi biểu diễn MHCAC
đọc, để mục tiêu contrastive không kéo trực tiếp lên đặc trưng phân loại. Không có
vai trò lúc inference.

## `pool_stream`

| Stream | Cách gộp | Lý do |
|---|---|---|
| PubMedCLIP | token CLS (`[..., 0, :]`) | Có token toàn cục thật |
| BioViL | mean các patch | Không có CLS; đầu ra toàn cục của chính nó **là** mean patch (`biovil_t/model.py:84`) |

⚠ **Không pool trên 246 token đã nối.** Làm vậy cân BioViL 196/246 so với
PubMedCLIP 50/246 thuần theo **số token**, và trộn hai không gian đặc trưng khác nhau.

## Gradient của nhánh aux

Forward của encoder vẫn trong `torch.no_grad()`; adapter và head chạy **ngoài** khối
đó và **không** detach. Đây là thiết lập SimCLR chia sẻ trọng số — không có momentum
encoder hay queue nào để một target detach kiểu MoCo có nghĩa, nên cả hai view cùng
cập nhật adapter và head.

## Called by

`Blip2Qformer.__init__` dựng `self.stream_adapters` / `self.mpc_heads`;
`Blip2Qformer._adapt` áp adapter; nhánh MPC trong `forward` dùng
`pool_stream` + head trước khi gọi loss.

## Related tests

`tests/test_multiview_losses.py` — adapter là identity ở init, head trả vector đơn
vị, `pool_stream` chọn đúng CLS/mean, và
`test_mpc_gradient_reaches_the_adapter_through_frozen_features`: gradient phải tới
được adapter **xuất phát từ tensor không `requires_grad`**. Đó chính là test lẽ ra
đã bắt được lỗi gốc.

## Modification risk

Bỏ zero-init của `up` sẽ làm model bị nhiễu ngay ở step 0 và một checkpoint
single-view không còn nạp sạch. Dời adapter xuống sau `_stash_prefusion` sẽ đưa MPC
về đúng trạng thái chết cũ, **không có lỗi nào báo** — chỉ có giá trị loss phẳng.
