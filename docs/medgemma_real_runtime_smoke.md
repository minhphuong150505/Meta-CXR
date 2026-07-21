# MedGemma real-runtime smoke test — BLOCKED, not run

**This is not a smoke-test result. Every runtime stage below is `NOT RUN`.**
The document exists to record *why*, with measured evidence, so the next round
does not re-derive it.

Round: 5 (`refactor/medgemma-real-runtime-validation`, branched from `05728bc`)
Date: 2026-07-21
Measured on: the local development machine, not a training host.

## Verdict

| Stage | Status | Evidence |
|---|---|---|
| CONFIG_LOAD | **NOT RUN** | blocked at hardware gate, before any download |
| PROCESSOR_LOAD | **NOT RUN** | blocked at hardware gate |
| MODEL_LOAD | **NOT RUN** | blocked at hardware gate |
| CAPABILITY_VALIDATION | **NOT RUN** | requires a real loaded model object |
| FORWARD | **NOT RUN** | requires a loaded model |
| GENERATION | **NOT RUN** | requires a loaded model |

No stage is reported as passed. No aggregate "passed" state is claimed.

## Blockers (measured, each independently sufficient)

### 1. No CUDA device

```
nvidia-smi                  -> command not found
lspci | grep -Ei 'vga|3d'   -> 00:02.0 VGA compatible controller:
                               Intel Corporation Meteor Lake-P [Intel Arc Graphics] (rev 08)
ls /dev/nvidia*             -> No such file or directory
lsmod | grep -i nvidia      -> no nvidia kernel module loaded
nvcc --version              -> command not found
/proc/driver/nvidia/version -> No such file or directory
```

GPU model: Intel Arc integrated (Meteor Lake-P). NVIDIA driver: none.
CUDA runtime: none.

This alone blocks the Stage-2 path. `requirements-stage2.txt` pins
`bitsandbytes==0.46.0`, and the 4-bit NF4 QLoRA load that `medgemma_direct`
depends on is CUDA-only. `requirements-stage1.txt` (pulled in transitively)
pins the CUDA 12.4 torch build.

Substituting a CPU load, an XPU/Arc backend, or a smaller stand-in model would
not validate the shipped path. Doing so and reporting MedGemma as verified is
the specific failure mode this round was meant to avoid.

### 2. Insufficient RAM for a CPU fallback

```
free -h  ->  total 15Gi | used 11Gi | free 996Mi | available 4.0Gi
             swap: 8.0Gi total, 5.3Gi already used, 2.7Gi free
```

MedGemma 1.5 4B-it in bf16 is ~8 GB of weights before activations, optimizer
state, or the vision tower's image tensors. Available RAM is 4.0 GiB with swap
already two-thirds consumed. A CPU-only load would OOM or thrash.

### 3. Model access not established through the sanctioned channel

```
HF_TOKEN                 -> NOT SET
HUGGING_FACE_HUB_TOKEN   -> NOT SET
~/.cache/huggingface/token -> file present (contents not read, not printed)
```

The brief specifies reading the token from `HF_TOKEN` only. That variable is
unset. A cached token file exists on disk and `huggingface_hub` would pick it
up implicitly, but relying on ambient on-disk credentials to reach a gated
repository is not the sanctioned path and was not exercised. Its contents were
never read, logged, or copied into config.

## Disk (not a blocker, recorded for completeness)

```
/      174G total, 31G available (82% used)
/home  296G total, 94G available (67% used)
```

Sufficient. Disk is not the constraint.

## What was deliberately NOT done

- `.venv-stage2` was **not** built. Installing the CUDA torch + bitsandbytes
  stack (~10 GB) for a machine with no CUDA device would validate nothing.
  The brief directs stopping *before* model download when the hardware gate
  fails; the environment build is downstream of that gate.
- No model, processor, or config was downloaded.
- No loader / reporter / collator / LoRA extraction was performed. Those steps
  are explicitly gated on a passing smoke test, and refactoring
  runtime-sensitive code against an unverifiable runtime is what the brief
  forbids.

## Residual risk carried forward (unchanged from round 4)

`MultimodalCapabilityValidator._forward_accepts_pixels` treats a `**kwargs`
forward signature as acceptance. Against the real MedGemma object this check
may pass *vacuously*, leaving the multimodal verdict resting on config and
module evidence alone. This cannot be confirmed or fixed without a real model
object, so it remains open.

## To unblock

A host with an NVIDIA GPU (the project's stated target is a single L4),
`HF_TOKEN` exported in the environment, and accepted MedGemma model terms.
Resume at section 3 of the round-5 brief.
