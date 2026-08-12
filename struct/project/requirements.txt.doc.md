> Source: `requirements.txt` (3 dòng)
> Status: ✅ ACTIVE — default Stage-1 alias
> Last verified against source: 2026-08-12

# `requirements.txt`

File chỉ có `-r requirements-stage1.txt`. Vì vậy `pip install -r requirements.txt`
cài environment Stage 1 mặc định; nó **không phải pin gộp legacy** và không cài
QLoRA Stage 2.

Muốn Stage 2 dùng `requirements-stage2.txt` trong environment riêng được khuyến
nghị bởi cloud setup.

← [HOME](../HOME.md)
