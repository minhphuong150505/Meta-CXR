> Source: `model/pretrained_medgemma/findings_loader.py:81-197`
> Status: ✅ ACTIVE

# `PretrainedFindingsLoader`

## Responsibility

Resolve device/dtype/4-bit plan, load đúng **multimodal** auto-class + processor,
verify image processor/vision tower, đặt eval và trả `FindingsModelBundle` có
provenance revision.

## `load()` flow

```text
plan_device(device, dtype, load_in_4bit)
  → import torch/AutoProcessor (thiếu → FindingsModelLoadError)
  → _model_class(): ImageTextToText hoặc Vision2Seq, không CausalLM
  → optional NF4 BitsAndBytesConfig
  → processor.from_pretrained + model.from_pretrained
  → _assert_has_image_processor + _assert_has_vision_tower
  → model.eval()
  → FindingsModelBundle(resolved_revision=_commit_hash hoặc requested revision)
```

## Fail-closed behavior

Không có text-only fallback. Nếu transformers không có multimodal auto-class,
raise và yêu cầu upgrade. Không biết Impression model và không thể load model thứ hai.

## Tests

`tests/test_pretrained_findings.py`, `tests/test_inference_only_invariants.py`.

← [`findings_loader.py`](../../findings_loader.py.doc.md) · [HOME](../../../../../HOME.md)
