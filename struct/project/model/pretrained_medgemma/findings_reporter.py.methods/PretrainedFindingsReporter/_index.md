> Source: `model/pretrained_medgemma/findings_reporter.py:53-116`
> Status: ✅ ACTIVE

# `PretrainedFindingsReporter`

## Responsibility

Giữ model bundle đã load và sinh FINDINGS cho một ảnh; class không tự load model
và không có reference tới Impression checkpoint.

## `generate(image)` flow

```text
build_messages(image): system + image + FINDINGS instruction
  → processor.apply_chat_template(..., tokenize=True)
  → inputs.to(model.device)
  → torch.inference_mode(): model.generate(settings)
  → slice outputs sau prompt_length
  → processor.decode(new_tokens)
  → postprocess_findings → FindingsGeneration(text, warnings, elapsed)
```

Slicing token theo `prompt_length` chính xác hơn xóa prompt bằng string matching.
`GenerationSettings.to_kwargs` là nơi duy nhất map generation parameter.

## Tests / risk

`tests/test_pretrained_findings.py`. Đừng decode toàn sequence rồi tìm/xóa prompt;
text trùng ngẫu nhiên có thể làm hỏng output.

← [`findings_reporter.py`](../../findings_reporter.py.doc.md) · [HOME](../../../../../HOME.md)
