> Source: `inference.py` (670 dòng)
> Status: ✅ ACTIVE (demo) — kiến trúc legacy
> Last verified against source: 2026-08-12

# `inference.py`

## Purpose
Gradio UI demo: ảnh X-quang → phân loại bất thường → báo cáo bằng **Vicuna-7B + LoRA**.

## ⚠ Đây là đường legacy về kiến trúc, nhưng ACTIVE về vận hành
README và `CLAUDE.md` gọi nó "legacy" — đúng, nó **chưa migrate sang MedGemma**.
Nhưng `Dockerfile:5` vẫn đặt `ENTRYPOINT ["/bin/bash", "inference.sh"]`, nên
container build ra vẫn chạy đường này.
[D-002](_meta/DECISIONS.md#d-002--đường-vicuna-7b-legacy-vẫn-là-demo-active) xác nhận
nó còn được dùng để demo.

## Entry point
```bash
./build_container.sh && ./run_container.sh    # Gradio :7860
bash inference.sh                             # trực tiếp
```

## Main functions
| Hàm | Dòng | Vai trò |
|---|---|---|
| `build_gradio_interface()` | 456 | ★ UI |
| `init_blip(cfg)` | 156 | Dựng `Blip2Qformer` từ config |
| `init_vicuna()` | 307 | ★ Vicuna-7B + LoRA — ⚠ `:312` `device_map={"": 0}` |
| `classify_abnormalities(...)` | 211 | MHCAC → P/N/U |
| `format_findings_dict(findings_dict)` | 289 | |
| `get_response(input_text, dicom)` | 333 | ★ Sinh báo cáo |
| `bot(history)` | 434 | Callback chat |
| `add_text` / `add_file` / `set_dicom` / `clear_history` | 415,420,410,426 | UI |
| `load_image(path)` / `remap_to_uint8(array)` | 197,164 | Tiền xử lý |
| `Conversation` / `SeparatorStyle` | 77,71 | Định dạng hội thoại |
| `parse_args()` | 54 | |

## Configuration
`--cfg-path pretraining/configs/blip2_pretrain_stage1_emb.yaml` ·
`SEED = 16` (đặt cứng `:32`) · `JAVA_HOME`/`JAVA_PATH` từ `local_config` (CheXpert
labeler cần Java) · `GRADIO_TEMP_DIR` đặt vào CWD

## Calls / Called by
Gọi: `model.lavis.tasks`, `Config`, `ReportDataset.create_chest_xray_transform_for_inference`,
`ExpandChannels`, `modeling_llama_imgemb.LlamaForCausalLM`, `peft.PeftModelForCausalLM`,
`gradio`, `skimage`.
Được gọi: `inference.sh:7`, `Dockerfile` ENTRYPOINT.

## Side effects
Cấp phát Vicuna-7B + Blip2Qformer trên GPU · `demo.launch(share=True)` tạo Gradio
share URL qua hạ tầng bên ngoài · Đặt env var Java và `GRADIO_TEMP_DIR` · Seed
toàn cục, `cudnn.deterministic = True`

## Error / edge cases
⚠ `:312` `device_map={"": 0}` ghim cứng GPU 0 — chỗ duy nhất còn lại trong repo
([I4](_meta/LEGACY_AND_OPTIONAL.md#-potential-issues--ghi-nhận-không-sửa)) ·
Thiếu Java → CheXpert labeler hỏng · Thiếu LoRA trong `checkpoints/` → không load
được · ⚠ `share=True` có thể đưa UI xử lý ảnh credentialed ra public URL (I11)

## Related tests
**Không có test nào.** Sửa file này không có lưới an toàn.

## Developer notes
1. ⚠ **Không dùng `stage2/prompts/PromptBuilder`** — đường prompt riêng. Đổi
   Prompt v2 không ảnh hưởng demo này.
2. ⚠ **Không import `utils/prompter.py`** dù docs cũ nói vậy.
3. `checkpoints/` chứa LoRA Vicuna (~29 MB), **không track trong Git**.
4. Migrate sang MedGemma sẽ ảnh hưởng: file này, `modeling_llama_imgemb.py`,
   `Dockerfile`, `inference.sh`, và làm `checkpoints/` hiện tại vô dụng.
5. **Không dùng ảnh MIMIC thật khi `share=True`.** Đây là potential DUA/privacy
   issue; task documentation chỉ ghi nhận, chưa sửa source.

← [HOME](../HOME.md)
