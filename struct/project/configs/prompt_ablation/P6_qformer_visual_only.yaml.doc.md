> Source: `configs/prompt_ablation/P6_qformer_visual_only.yaml` (17 dòng)
> Status: 🧪 ABLATION — clean visual control
> Last verified against source: 2026-08-12

# `P6_qformer_visual_only.yaml`

`visual_mode: qformer_visual_only`: prompt nhận Q-Former soft token nhưng **không
nhận structured Stage-1 labels**. Đây là control sạch để đo labels có giúp hay
không. Nếu builder đưa P/N/U vào mode này, test prompt phải fail.

← [`prompt_ablation/`](_index.md) · [HOME](../../../HOME.md)
