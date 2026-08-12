> Source: `cloud/run_paper_assets.sh` (37 dòng)
> Status: ⚠ BROKEN / POTENTIALLY_UNUSED
> Last verified against source: 2026-08-12

# `cloud/run_paper_assets.sh`

## Intended purpose

Chạy `paper_assets.py` trên checkpoint Stage 1/test split rồi upload asset vào
private GCS.

## Current evidence

- Script kiểm bucket private và checkpoint trước khi chạy.
- Nó gọi `"$PYTHON_BIN" paper_assets.py ...` tại dòng 24.
- **Repository hiện không có `paper_assets.py`** và không có module thay thế cùng
  tên; chạy từ checkout này sẽ fail tại bước đó.

Vì chưa có user confirmation về caller/tool nằm ngoài repo, status chỉ là
POTENTIALLY_UNUSED/BROKEN, chưa kết luận legacy.

## Data handling

Comment nói asset có thể chứa identifier hoặc generated clinical text và không
publishable. Dù upload đích đã được kiểm private, `OUT_DIR` local vẫn là dữ liệu
nhạy cảm và không được commit.

← [`cloud/`](_index.md) · [Legacy & Optional](../_meta/LEGACY_AND_OPTIONAL.md)
