> Source: `pretraining/itc_gate.py`
> Status: ✅ ACTIVE — `scripts/check_itc_gate.py` và gate mỗi epoch trong runner
> Last verified against source: 2026-09-24

# `pretraining/itc_gate.py`

← [pretraining](_index.md) · [D-020](../_meta/DECISIONS.md#d-020--meta-former-3-pha-theo-bài-báo-medclip-swin-mhcac-một-nhánh)

InfoNCE hai chiều trên một tập cố định `pairs` cặp hợp lệ (bỏ study không có
FINDINGS), eval mode, không queue, không gradient. Báo `delta_nats = ln N −
L_itc`, hạng trung bình của cặp đúng, R@1/R@5 hai chiều. `meets_threshold` cần
**cả** `delta_nats >= min_delta` **và** R@5 vượt ngẫu nhiên hai chiều: số lần
trúng >= phân vị một phía 1−alpha của Binomial(n, 5/n) (alpha 0.01; n=256 → ≥ 12
lần). R@1 chỉ báo, không bắt buộc (kỳ vọng 1 lần trúng ở n=256).

`itc_features` đi qua `model.encode_samples`, cùng lệnh `forward()` dùng — kể
cả `swin_image` của MedCLIP và đặc trưng từ cache.

Test: `tests/test_paper_mode.py`.
