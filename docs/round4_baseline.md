# Round 4 baseline and runtime blocker

## Source state

| Field | Value |
|---|---|
| Branch | `refactor/stage2-real-medgemma-runtime` |
| Base SHA | `8411647` |
| Backup tag | `stage2-runtime-before-real-env-8411647` (pushed) |
| Prior branches pushed | `refactor/clean-medgemma-xai-pipeline`, `refactor/stage2-runtime-integration` |
| Working tree at baseline | clean |

## Measured baseline

Python 3.12.3, Linux 6.17.0-40-generic, `/home/phuong/venv/bin/python`.

| Command | Result |
|---|---|
| `python -m compileall -q .` | exit 0 |
| `python -m pytest` | **250 passed, 0 failed, 0 skipped, 2.8 s** |
| `ruff check .` | 426 errors (372 auto-fixable) |
| `ruff format --check .` | 68 would reformat, 26 already formatted |

250 tests matches the round-3 exit state. No investigation needed.

## Runtime blocker: real MedGemma validation is not possible on this machine

Round 4's P1–P3 (build `.venv-stage2`, load real MedGemma, validate the
capability checker against the real object) were **not attempted**, because the
hardware cannot support them. This is a measured blocker, not an estimate.

```
$ nvidia-smi
bash: nvidia-smi: command not found

$ free -g
               total        used        free      shared  buff/cache   available
Mem:              15          11           0           1           3           3

$ df -h /home/phuong
/dev/nvme0n1p6  296G  187G   94G  67% /home

$ curl -s -o /dev/null -w "%{http_code}" https://huggingface.co/api/models/google/medgemma-1.5-4b-it
200

$ [ -n "$HF_TOKEN" ] && echo yes || echo no
no
```

| Requirement | Available | Verdict |
|---|---|---|
| CUDA device | none (`nvidia-smi` absent) | **blocked** |
| RAM for a 4B model in bf16 (~8 GB) | ~3 GB available of 15 GB total | **blocked** |
| 4-bit NF4 load as a workaround | needs `bitsandbytes`, which needs CUDA | **blocked** |
| Hugging Face reachability | HTTP 200 | ok |
| `HF_TOKEN` for the gated repo | not set | **blocked** |
| Disk | 94 GB free | ok |

Three independent blockers, any one of which is sufficient. Consequently this
round makes **no claim** about:

- MedGemma config, processor or model loading;
- `MultimodalCapabilityValidator` behaviour against the real object;
- any forward or generate pass;
- GPU execution of any kind.

`docs/medgemma_real_runtime_smoke.md` is deliberately **not** created: an empty
or speculative smoke-test report would be worse than its absence.

### What this means for the capability validator

`MultimodalCapabilityValidator` remains validated against fakes only. The
specific residual risk, stated so it is not forgotten:

`_forward_accepts_pixels` inspects the `forward` signature for a pixel key, and
falls back to accepting any `**kwargs` forward. MedGemma's real `forward` may
well be `**kwargs`-based, in which case that check passes vacuously and the
verdict rests entirely on the config and module evidence. That is probably
still correct, but it is **unverified**. First action once a machine with a GPU
and model access is available: run the validator against the real object and
confirm `forward_accepts_pixel_values` is doing real work rather than passing
by fallback.

## Scope actually executed in round 4

CPU-verifiable work only:

- extract the MedGemma loader and LoRA configuration from the god script;
- notebook privacy guard;
- device/distributed audit (documentation only, no DDP);
- trainer wiring where it can be executed.

No `.venv-stage2` was created. No multi-GB download was performed.
