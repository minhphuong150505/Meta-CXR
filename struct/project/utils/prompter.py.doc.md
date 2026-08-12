> Source: `utils/prompter.py` (51 dòng)
> Status: ⚠ POTENTIALLY_UNUSED
> Last verified against source: 2026-08-12

# `utils/prompter.py`

## Purpose (theo thiết kế)
`Prompter` — dựng prompt từ template JSON cho đường Vicuna.

## ⚠ Không có caller
**Zero import trong toàn repo.** `docs/stage2_prompt_audit.md:20` ghi nó dùng bởi
`inference.py` — grep **không xác nhận**. Code thắng documentation.

## ⚠ Đường dẫn template không khớp
`:19` tìm `data/templates/{name}.json`. File template thật nằm ở
`model/lavis/data/templates/vicuna.json`. Hai đường không khớp — nếu ai gọi
`Prompter("vicuna")` từ repo root, nó sẽ `ValueError("Cant read …")`.

## Main class
`Prompter` (`:10`) — `__slots__ = ("template", "_verbose")`; mặc định template
`"alpaca"` khi truyền chuỗi rỗng.

## Calls / Called by
Gọi: `json`, `os.path`.
Được gọi: **không ai**.

## Side effects
Đọc file JSON lúc `__init__`.

## Error / edge cases
Template không tồn tại → `ValueError` (`:21`).

## Developer notes
Muốn dùng: sửa đường dẫn template trước. Xem
[D-002](../_meta/DECISIONS.md#d-002--đường-vicuna-7b-legacy-vẫn-là-demo-active).

← [`_index.md`](_index.md) · [HOME](../../HOME.md)
