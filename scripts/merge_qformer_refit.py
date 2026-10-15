#!/usr/bin/env python3
"""Write a finished Stage-1 checkpoint with its META-Former side replaced by a phase1q refit.

phase1q (2026-10-15, ``docs/handoff/PLAN-2026-10-15-qformer-refit.md``) trains only the
Q-Former side on a frozen checkpoint, and its own checkpoints do NOT hold the encoder blocks
phase 1c unfroze (no phase lists them in ``trainable``, so ``checkpoint_keep`` drops them).
Loaded alone, the encoders would silently fall back to their pretrained weights and MHCAC
would see different features. So Stage 2, the probe and the diagnostics read the file this
script writes: the BASE checkpoint's state with every tensor under the refit's prefixes
(``trainable`` plus ``init_from.skip`` of the refit's phase, read from its saved config)
taken from the REFIT checkpoint.

It refuses, rather than writes, when the refit moved anything else: every tensor the two
files share outside those prefixes (MHCAC, StreamAdapters, view fusion, projector, BatchNorm
buffers) must be bit-identical, which is the proof that MHCAC, the cutpoints and every Stage-1
number of the base still hold. It also refuses a refit that lacks a tensor of the base under
its prefixes, or that carries a tensor the base does not have outside them.

    python scripts/merge_qformer_refit.py --base <run>/phase1c/.../checkpoint_best.pth \\
        --refit <refit>/.../checkpoint_best.pth --output <private dir>/stage1_refit_merged.pth

Prints counts only. The output is a model checkpoint: never inside the repository.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pretraining.phases import matches_any  # noqa: E402


class MergeError(ValueError):
    """The refit checkpoint cannot be merged without changing the base model."""


def refit_prefixes(config: dict) -> tuple[str, ...]:
    """``trainable`` + ``init_from.skip`` of the phase the refit checkpoint was trained in."""
    run = (config or {}).get("run", {}) or {}
    phase = run.get("phase")
    block = ((run.get("phases") or {}).get(phase) or {}) if phase else {}
    if not block:
        raise MergeError(f"the refit checkpoint's config names no phase block (run.phase={phase!r})")
    init_from = block.get("init_from") or {}
    prefixes = tuple(dict.fromkeys([*(block.get("trainable") or []), *(init_from.get("skip") or [])]))
    if not prefixes:
        raise MergeError(f"phase {phase!r} has no trainable or skipped prefixes")
    return prefixes


def merge_states(base: dict, refit: dict, prefixes) -> tuple[dict, dict]:
    """Return ``(merged_state, summary)``; raise :class:`MergeError` on any violation."""
    import torch

    prefixes = tuple(prefixes)
    under = {k for k in refit if matches_any(k, prefixes)}
    base_under = {k for k in base if matches_any(k, prefixes)}
    missing = sorted(base_under - under)
    if missing:
        raise MergeError(f"the refit lacks {len(missing)} tensors of the base under its prefixes, "
                         f"e.g. {missing[:5]}")
    empty = [p for p in prefixes if not any(matches_any(k, (p,)) for k in under)]
    stray_new = sorted(k for k in refit if k not in under and k not in base)
    if stray_new:
        raise MergeError(f"the refit carries {len(stray_new)} tensors the base does not have outside "
                         f"its prefixes, e.g. {stray_new[:5]}")
    shared = [k for k in refit if k not in under and k in base]
    changed = []
    for key in shared:
        a, b = base[key], refit[key]
        if not (torch.is_tensor(a) and torch.is_tensor(b)):
            continue
        if a.shape != b.shape or a.dtype != b.dtype or not torch.equal(a, b):
            diff = (float((a.double() - b.double()).abs().max()) if a.shape == b.shape else float("nan"))
            changed.append((key, diff))
    if changed:
        raise MergeError(
            f"the refit changed {len(changed)} tensors outside its prefixes (the base model would "
            f"not be the base any more), e.g. {[(k, f'{d:.3g}') for k, d in changed[:5]]}"
        )
    merged = dict(base)
    for key in under:
        merged[key] = refit[key]
    summary = {
        "prefixes": list(prefixes),
        "prefixes_without_tensors": empty,
        "replaced": len(under & base_under),
        "added": len(under - base_under),
        "verified_identical": len(shared),
        "base_only_kept": len(set(base) - set(refit)),
        "total": len(merged),
    }
    return merged, summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--refit", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args(argv)

    out = args.output.expanduser().resolve()
    if _REPO_ROOT in out.parents:
        raise SystemExit(f"{out} is inside the repository; checkpoints never go there")
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")

    import torch

    base_obj = torch.load(args.base, map_location="cpu", weights_only=False)
    refit_obj = torch.load(args.refit, map_location="cpu", weights_only=False)
    prefixes = refit_prefixes(refit_obj.get("config"))
    merged, summary = merge_states(base_obj["model"], refit_obj["model"], prefixes)
    summary.update({"base": str(args.base), "refit": str(args.refit),
                    "refit_epoch": refit_obj.get("epoch")})
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": merged, "merged_from": {"base": str(args.base), "refit": str(args.refit)},
                "refit_prefixes": list(prefixes), "config": refit_obj.get("config")}, out)
    print("[merge] " + json.dumps({k: v for k, v in summary.items() if k not in ("base", "refit")}),
          flush=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
