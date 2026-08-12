> Source: `utils/` (4 file, 253 LOC)
> Status: ⚠ POTENTIALLY_UNUSED
> Last verified against source: 2026-08-12

# `utils/`

## ⚠ Đọc trước: không file nào có caller

**Zero import trong toàn repository** — kể cả từ `inference.py`.

```bash
grep -rn "Prompter\|MyDataCollatorForSeq2Seq\|split_emb" --include='*.py' --include='*.ipynb' . \
  | grep -v '^./utils/'
# → rỗng
```

⚠ **Mâu thuẫn với documentation:** `docs/stage2_prompt_audit.md:20` ghi
*"Legacy Vicuna JSON prompter (inference.py only)"*. Grep không xác nhận điều đó.

Theo nguyên tắc của repo: **code thắng documentation**. Trang này ghi trung thực
"không có caller", không suy diễn rằng `inference.py` dùng chúng.

Lưu ý bối cảnh: [D-002](../_meta/DECISIONS.md#d-002--đường-vicuna-7b-legacy-vẫn-là-demo-active)
xác nhận **đường Vicuna là ACTIVE** (Docker vẫn chạy nó) — nhưng `utils/` vẫn
không được đường đó import.

## Purpose (theo thiết kế)

Helper cho đường Vicuna-7B: dựng prompt từ template JSON, streaming generation
callback, và data collator seq2seq.

## Parent

[`struct/project/`](../../HOME.md#source-code-tree)

## Children

| File | LOC | Doc | Status | Ghi chú |
|---|---|---|---|---|
| `prompter.py` | 50 | [📄](prompter.py.doc.md) | ⚠ | `Prompter` đọc `data/templates/{name}.json`. ⚠ Template thật nằm ở `model/lavis/data/templates/vicuna.json` — **đường dẫn không khớp** |
| `callbacks.py` | 75 | [📄](callbacks.py.doc.md) | ⚠ | `Stream(transformers.StoppingCriteria)`; mượn từ text-generation-webui |
| `datacollator.py` | 106 | [📄](datacollator.py.doc.md) | ⚠ | `MyDataCollatorForSeq2Seq` |
| `split_emb.py` | 22 | [📄](split_emb.py.doc.md) | ⚠⚠ | **Chạy ngay khi import** |
| `__init__.py` | 0 | — | | Rỗng |

## ⚠⚠ `split_emb.py` — nguy hiểm khi import

Không có `if __name__ == "__main__"` guard. Toàn bộ nội dung chạy ở module scope:

```python
input_path = "pretraining/embs/2025_05_20_blip2_pretrain_stage1_emb_embeddings_iu_xray_test.pkl"
...
with open(input_path, "rb") as f:      # ← chạy ngay khi import
```

`pretraining/embs/` bị `.gitignore` chặn và **không tồn tại** trong working tree.

**Hệ quả:** bất kỳ `from utils import *` hay `import utils.split_emb` nào cũng sẽ
thực thi và ném `FileNotFoundError`. Đây là [I3](../_meta/LEGACY_AND_OPTIONAL.md#-potential-issues--ghi-nhận-không-sửa).

## Main responsibilities

Theo thiết kế — nhưng hiện không được thực thi bởi ai.

## Entry points

Không có (`split_emb.py` là script nhưng không có guard).

## Dependencies

`transformers` (callbacks, datacollator) · `torch` · `numpy` · stdlib

## Used by

**Không ai.**

## Status

```text
⚠ POTENTIALLY_UNUSED
```

Không gắn LEGACY vì D-002 xác nhận đường Vicuna còn sống — có thể chúng được dự
định dùng nhưng chưa nối, hoặc đã bị thay thế mà chưa dọn.

## Notes

- Nếu bạn xác nhận `utils/` không cần nữa, thêm decision và chuyển sang
  [LEGACY_AND_OPTIONAL.md](../_meta/LEGACY_AND_OPTIONAL.md).
- Nếu bạn định dùng `Prompter`, phải sửa đường dẫn template trước — nó tìm
  `data/templates/`, còn file thật ở `model/lavis/data/templates/`.
- ⚠ Đừng thêm re-export vào `utils/__init__.py`: sẽ kéo `split_emb.py` vào và
  làm mọi `import utils` nổ.

## Related documentation

[LEGACY_AND_OPTIONAL.md](../_meta/LEGACY_AND_OPTIONAL.md#-potentially_unused--utils) ·
[D-002](../_meta/DECISIONS.md#d-002--đường-vicuna-7b-legacy-vẫn-là-demo-active) ·
[`inference.py.doc.md`](../inference.py.doc.md)

← [Về HOME](../../HOME.md)
