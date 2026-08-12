> Source: `stage2/`
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-12

# `stage2/`

## Purpose

Chứa **Prompt v2** — hệ thống prompt có version cho Stage 2. Cố ý tách khỏi
`training/` để nó hoàn toàn **stdlib-only**: không torch, không transformers,
không model.

## Role in project

```text
records ──► stage2/prompts/PromptBuilder ──► PromptPart[] ──► MedGemma
                    ▲
        cùng một builder cho CẢ train VÀ inference
```

Nhờ không đụng model/tokenizer, parity giữa train và inference kiểm tra được
**byte-for-byte**.

## Parent

[`struct/project/`](../../HOME.md#source-code-tree)

## Children

| Thư mục | Doc |
|---|---|
| `prompts/` | [📁](prompts/_index.md) |
| `__init__.py` | — |

## Status

```text
✅ ACTIVE
```

## Related documentation

[`prompts/_index.md`](prompts/_index.md) · [ARCHITECTURE.md §4](../_meta/ARCHITECTURE.md#4-prompt-v2)

← [Về HOME](../../HOME.md)
