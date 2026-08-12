> Source: `medgemma_inference/prediction_writer.py:71-109`
> Status: ✅ ACTIVE

# `PredictionWriter`

## Responsibility

Append JSONL bền theo từng record và theo dõi `sample_key` đã hoàn tất để resume.

## Lifecycle

```text
__init__ → mkdir + read_completed_keys (cắt partial trailing line)
  → __enter__ mở append handle
  → write(record): assert_publishable → JSON line → flush + fsync → add key
  → __exit__/close: flush + fsync + close
```

## Privacy contract

`assert_publishable` từ chối ID, image path và reference report. `write` raise nếu
dùng ngoài context manager. `fsync` mỗi record tốn throughput nhưng bảo toàn mọi
sample đã trả tiền GPU khi run bị kill/budget stop.

## Called by / tests

`runner.run_findings_inference`; `tests/test_pretrained_findings.py`.

← [`prediction_writer.py`](../../prediction_writer.py.doc.md) · [HOME](../../../../HOME.md)
