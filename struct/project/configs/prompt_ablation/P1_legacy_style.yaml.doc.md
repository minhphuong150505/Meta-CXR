> Source: `configs/prompt_ablation/P1_legacy_style.yaml` (25 dòng)
> Status: 🧪 ABLATION
> Last verified against source: 2026-08-12

# `P1_legacy_style.yaml`

Policy-equivalent gần nhất với prompt shipped cũ: `qformer_guided`, dump toàn bộ
negative, giữ P/N/U, không context view/prior và không nhấn visual-primary.
**Không byte-identical** với legacy `build_instruction`; comment source nói rõ đây
chỉ là baseline gần nhất trong framework v2.

Consumer: `scripts/run_prompt_ablation.py` / `load_prompt_config`.

← [`prompt_ablation/`](_index.md) · [HOME](../../../HOME.md)
