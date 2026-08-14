> Source: `scripts/evaluate_explanation.py::_build_runtime`

# `_build_runtime(args)`

Import torch, OmegaConf, `local_config`, LAVIS và `ReportDataset` **bên trong hàm**.
Config được override trước `task.build_model`: load checkpoint CLI như finetuned,
đặt mask cache CLI, tắt feature cache để luôn có ảnh cho figure. Dựng DataLoader
batch 1, shuffle false, đúng collater study-level.

Model được đặt `eval()` nhưng parameter không bị `requires_grad_(False)`: graph
từ projection/head phải còn sống để Grad-CAM đạo hàm theo activation. Không có
optimizer nên không có cập nhật tham số.

← [`methods`](./_index.md)
