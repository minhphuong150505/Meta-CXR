> Source: `utils/datacollator.py` (107 dòng)
> Status: ⚠ POTENTIALLY_UNUSED
> Last verified against source: 2026-08-12

# `utils/datacollator.py`

## Purpose (theo thiết kế)
`MyDataCollatorForSeq2Seq` — pad động input và label.

## ⚠ Không có caller
Zero import. Stage 2 dùng `VariantLLM.collate_train` (`fig9:1059`) thay vì cái này.

## Main class
`MyDataCollatorForSeq2Seq` (`:15`) — dataclass, nhận `PreTrainedTokenizerBase`,
`PaddingStrategy`.

## Calls / Called by
Gọi: `transformers`, `numpy`, `random`.
Được gọi: **không ai**.

## Side effects
Không.

## Developer notes
Nếu cần collator, `fig9.collate_train` là bản đang được dùng thật.

← [`_index.md`](_index.md) · [HOME](../../HOME.md)
