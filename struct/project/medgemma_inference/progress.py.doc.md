> Source: `medgemma_inference/progress.py` (102 dòng)
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-12

# `medgemma_inference/progress.py`

## Purpose
Đảm bảo một run được resume **là cùng một run**.

## ★ Vì sao đây không phải chuyện nhỏ
Docstring `:3` giải thích: nếu model, revision, generation settings, split hay
dataset fingerprint đã đổi, việc append vào file predictions cũ sẽ **âm thầm trộn
output của hai cấu hình khác nhau vào một tập kết quả** — và **không metric hạ
nguồn nào phân biệt được**.

Nên module này **từ chối** thay vì đoán.

## Status
```text
✅ ACTIVE
```

## Main items
| Tên | Dòng |
|---|---|
| `RunIdentity` | 25 |
| [`ProgressFile`](progress.py.methods/ProgressFile/_index.md) | 55 |
| `ResumeMismatch` | 20 |

## Calls / Called by
Gọi: `json`, `hashlib`.
Được gọi: `runner.py` (`build_run_identity`); `tests/test_pretrained_findings.py`.

## Side effects
Ghi file progress.

## Error / edge cases
Identity lệch → `ResumeMismatch` raise, nêu trường nào khác.

## Related tests
`tests/test_pretrained_findings.py`

## Developer notes
Nếu bạn **cố ý** muốn đổi cấu hình, hãy dùng thư mục output mới — đừng nới lỏng
kiểm tra này.

← [`_index.md`](_index.md) · [HOME](../../HOME.md)
