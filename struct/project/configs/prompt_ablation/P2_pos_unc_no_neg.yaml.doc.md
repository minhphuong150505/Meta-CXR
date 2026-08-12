> Source: `configs/prompt_ablation/P2_pos_unc_no_neg.yaml` (22 dòng)
> Status: 🧪 ABLATION
> Last verified against source: 2026-08-12

# `P2_pos_unc_no_neg.yaml`

Chỉ đưa positive + uncertain vào guided prompt (`negative_policy: none`), không
statement normal có cấu trúc, vẫn mang prior flag. Dùng để đo tác động của việc
bỏ hoàn toàn negative findings.

Consumer: `scripts/run_prompt_ablation.py` / `load_prompt_config`.

← [`prompt_ablation/`](_index.md) · [HOME](../../../HOME.md)
