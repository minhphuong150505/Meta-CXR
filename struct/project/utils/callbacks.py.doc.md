> Source: `utils/callbacks.py` (75 dòng)
> Status: ⚠ POTENTIALLY_UNUSED
> Last verified against source: 2026-08-12

# `utils/callbacks.py`

## Purpose (theo thiết kế)
Hỗ trợ streaming output khi generate.

## ⚠ Không có caller
Zero import trong repo. `inference.py:624` có comment `# --------- Callbacks ----------`
nhưng **không import module này**.

## Nguồn gốc
Docstring `:3`: mượn từ `oobabooga/text-generation-webui`.

## Main classes
| Class | Dòng |
|---|---|
| `Stream(transformers.StoppingCriteria)` | 15 |
| `Iteratorize` | 25 |

`Iteratorize` chạy generate trong thread riêng và đẩy token qua `Queue` — biến API
callback thành iterator.

## Calls / Called by
Gọi: `transformers`, `torch`, `queue`, `threading`, `gc`.
Được gọi: **không ai**.

## Side effects
Tạo thread; `gc.collect()`.

## Developer notes
Code bên thứ ba. Nếu bật streaming cho Gradio, đây là thứ có sẵn.

← [`_index.md`](_index.md) · [HOME](../../HOME.md)
