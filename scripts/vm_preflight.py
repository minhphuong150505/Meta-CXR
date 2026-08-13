#!/usr/bin/env python3
"""VM preflight check for META-CXR training (Stage 1 + Stage 2).

Verifies a freshly-cloned checkout can train on the target host (default target:
2x RTX 3090, 64 GB RAM) BEFORE any long run is started. It deliberately loads no
model weights and downloads nothing, so it is safe and fast to run repeatedly.

Usage:
    python scripts/vm_preflight.py                 # human-readable report
    python scripts/vm_preflight.py --stage 1       # only Stage-1 relevant checks
    python scripts/vm_preflight.py --json          # machine-readable

Exit code is 0 only when there are no FAIL rows (WARN rows do not fail the run).
Never prints secret values: environment variables are reported as set/unset only.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
_rows: list[tuple[str, str, str]] = []


def record(status: str, name: str, detail: str = "") -> None:
    _rows.append((status, name, detail))


def _spec(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except (ModuleNotFoundError, ValueError):
        return False


def check_python() -> None:
    v = sys.version_info
    detail = f"{v.major}.{v.minor}.{v.micro}"
    record(PASS if (v.major, v.minor) >= (3, 10) else FAIL, "Python >= 3.10", detail)


def check_torch_cuda(stage: str) -> None:
    if not _spec("torch"):
        # Stage 1 and Stage 2 use different envs; torch may be absent in the one
        # you happen to have activated. Report, do not crash.
        record(WARN, "PyTorch importable", "torch not found in active env")
        return
    import torch

    record(PASS, "PyTorch importable", f"torch {torch.__version__}")
    if not torch.cuda.is_available():
        record(FAIL, "CUDA available", "torch.cuda.is_available() is False")
        return
    n = torch.cuda.device_count()
    record(PASS if n >= 1 else FAIL, "GPU count", str(n))
    if stage in ("1", "both") and n < 2:
        record(WARN, "Stage-1 DDP (2 GPUs)", f"{n} GPU visible; DDP wants 2 (single-GPU still works)")
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        gb = p.total_memory / 1024**3
        row = FAIL if gb < 20 else PASS
        record(row, f"GPU{i} VRAM", f"{p.name}, {gb:.1f} GB")


def check_ram() -> None:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        size = os.sysconf("SC_PAGE_SIZE")
        gb = pages * size / 1024**3
    except (ValueError, OSError):
        record(WARN, "System RAM", "could not read")
        return
    # Stage-2 builds a ~10-11 GB feature cache; 64 GB target, warn under ~48 GB.
    record(PASS if gb >= 48 else WARN, "System RAM", f"{gb:.0f} GB")


def check_disk() -> None:
    free = shutil.disk_usage(REPO_ROOT).free / 1024**3
    # Weights + checkpoints + Stage-2 cache. Warn if under ~100 GB free.
    record(PASS if free >= 100 else WARN, "Free disk at repo", f"{free:.0f} GB free")


def check_shm() -> None:
    shm = Path("/dev/shm")
    if not shm.exists():
        record(WARN, "/dev/shm", "absent (DataLoader workers may fail with shared-memory errors)")
        return
    free = shutil.disk_usage(shm).free / 1024**3
    # DataLoader workers pass tensors via /dev/shm; small default shm OOMs workers.
    record(PASS if free >= 4 else WARN, "/dev/shm free", f"{free:.1f} GB (raise with --shm-size if in Docker)")


def check_write_perms() -> None:
    for rel in ("pretraining/outputs", "training/outputs", "checkpoints"):
        target = REPO_ROOT / rel
        try:
            target.mkdir(parents=True, exist_ok=True)
            probe = target / ".preflight_write_probe"
            probe.write_text("ok")
            probe.unlink()
            record(PASS, f"writable: {rel}", "")
        except OSError as exc:
            record(FAIL, f"writable: {rel}", f"{type(exc).__name__}")


def check_imports(stage: str) -> None:
    core = ["numpy", "pandas", "yaml", "omegaconf"]
    for m in core:
        record(PASS if _spec(m) else FAIL, f"import {m}", "")
    stage1 = ["torch", "torchvision", "timm", "transformers"]
    stage2 = ["torch", "transformers", "peft", "bitsandbytes", "accelerate"]
    optional_eval = ["nltk", "bert_score", "pycocoevalcap"]
    if stage in ("1", "both"):
        for m in stage1:
            record(PASS if _spec(m) else WARN, f"[stage1] import {m}", "install requirements-stage1.txt" if not _spec(m) else "")
    if stage in ("2", "both"):
        for m in stage2:
            record(PASS if _spec(m) else WARN, f"[stage2] import {m}", "install requirements-stage2.txt" if not _spec(m) else "")
    for m in optional_eval:
        if not _spec(m):
            record(WARN, f"[eval] optional {m}", "generation metrics reported as unavailable without it")


def check_paths() -> None:
    cfg = REPO_ROOT / "configs" / "env_config.yaml"
    if not cfg.exists():
        record(FAIL, "configs/env_config.yaml", "missing -- copy from env_config.yaml.example and fill paths")
        return
    record(PASS, "configs/env_config.yaml", "present")
    try:
        from omegaconf import OmegaConf

        env = OmegaConf.load(cfg)
        vis_root = OmegaConf.select(env, "paths.mimic_cxr_jpg_root")
        if vis_root and (Path(vis_root) / "files").is_dir():
            record(PASS, "MIMIC-CXR jpg root", "contains files/")
        elif vis_root:
            record(WARN, "MIMIC-CXR jpg root", f"{vis_root} has no files/ (mount before training)")
        else:
            record(WARN, "MIMIC-CXR jpg root", "paths.mimic_cxr_jpg_root not set")
    except Exception as exc:  # noqa: BLE001 - report, never crash preflight
        record(WARN, "env_config.yaml parse", f"{type(exc).__name__}")


def check_train_configs() -> None:
    for rel in ("pretraining/configs/mimic_cxr_full.yaml",):
        p = REPO_ROOT / rel
        record(PASS if p.exists() else WARN, f"config: {rel}", "present" if p.exists() else "absent")


def check_env_vars() -> None:
    # Presence only -- never the value.
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(var):
            record(PASS, f"env {var}", "set (value hidden)")
            return
    record(WARN, "HF auth token", "neither HF_TOKEN nor HUGGING_FACE_HUB_TOKEN set (needed for gated MedGemma)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["1", "2", "both"], default="both")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)

    check_python()
    check_torch_cuda(args.stage)
    check_ram()
    check_disk()
    check_shm()
    check_write_perms()
    check_imports(args.stage)
    check_paths()
    check_train_configs()
    check_env_vars()

    fails = sum(1 for s, _, _ in _rows if s == FAIL)
    warns = sum(1 for s, _, _ in _rows if s == WARN)

    if args.json:
        import json

        print(json.dumps({"rows": [{"status": s, "check": n, "detail": d} for s, n, d in _rows],
                          "fail": fails, "warn": warns}, indent=2))
    else:
        width = max(len(n) for _, n, _ in _rows)
        for s, n, d in _rows:
            mark = {PASS: "  OK", WARN: "WARN", FAIL: "FAIL"}[s]
            print(f"  [{mark}] {n:<{width}}  {d}")
        print(f"\n  {len(_rows)} checks: {fails} FAIL, {warns} WARN")
        print("  Preflight " + ("FAILED -- resolve FAIL rows before training." if fails else "OK -- WARN rows are advisory."))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
