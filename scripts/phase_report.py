"""Summarise a three-phase Stage-1 run (section H of the 3-phase plan).

Reads what the runner writes under ``<root>/<phase>/<run_name>/``:

* ``phase_metrics.jsonl`` -- per-epoch val stats, including the ``sp_*``
  study_presence macros (12-label primary, ``_13labels``, ``_14labels``);
* ``itc_gate_epoch<N>.json`` -- the per-epoch ITC gate;
* ``grad_interference.jsonl`` -- phase-1c cosine / norm ratio of classification
  vs alignment gradients on shared parameters.

Prints markdown: the gate per epoch, each phase-1c epoch's classification
against the LAST phase-1b epoch (``checkpoint_phase1b``), and the gradient
interference summary. Numbers only -- nothing here reads report text or ids.

    python scripts/phase_report.py --root ~/run_<date>_3phase
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

CLASSIFICATION_KEYS = [
    ("macro AUROC", "sp_macro_auroc"),
    ("macro AUPRC", "sp_macro_auprc"),
    ("pos. macro F1 @0.5", "sp_positive_macro_f1"),
]
VIEWS = [("12", ""), ("13", "_13labels"), ("14", "_14labels")]


def _phase_dir(root: Path, phase: str) -> Path | None:
    hits = sorted((root / phase).glob("*/phase_metrics.jsonl")) or sorted(
        (root / phase).glob("*/itc_gate_epoch*.json")
    )
    return hits[0].parent if hits else None


def _rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _val_by_epoch(directory: Path | None) -> dict[int, dict]:
    if directory is None:
        return {}
    out = {}
    for row in _rows(directory / "phase_metrics.jsonl"):
        if row.get("split") == "val" and isinstance(row.get("epoch"), int):
            out[row["epoch"]] = row
    return out


def gate_table(root: Path, phases) -> list[str]:
    lines = [
        "| phase | epoch | pairs | temp | loss_itc | ln N | delta_nats | R@1 i2t/t2i | R@5 i2t/t2i | R@5 chance | pass |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|---:|---|",
    ]
    for phase in phases:
        directory = _phase_dir(root, phase)
        if directory is None:
            continue
        for path in sorted(directory.glob("itc_gate_epoch*.json")):
            g = json.loads(path.read_text())
            lines.append(
                f"| {phase} | {g['epoch']} | {g['pairs']} | {g['temperature']} | "
                f"{g['loss_itc']} | {g['chance_ln_n']} | {g['delta_nats']:+.4f} | "
                f"{g['R@1_i2t']:.4f} / {g['R@1_t2i']:.4f} | "
                f"{g['R@5_i2t']:.4f} / {g['R@5_t2i']:.4f} | {g['R@5_chance']:.4f} | "
                f"{'yes' if g['meets_threshold'] else 'NO'} |"
            )
    return lines


def classification_table(root: Path) -> list[str]:
    ref_rows = _val_by_epoch(_phase_dir(root, "phase1b"))
    joint_rows = _val_by_epoch(_phase_dir(root, "phase1c"))
    if not ref_rows:
        return ["(no phase1b val metrics)"]
    ref = ref_rows[max(ref_rows)]
    lines = [
        "| metric | labels | end of 1b | "
        + " | ".join(f"1c ep {e} (delta)" for e in sorted(joint_rows))
        + " |",
        "|---|---:|---:|" + "---:|" * len(joint_rows),
    ]
    for label, key in CLASSIFICATION_KEYS:
        for view, suffix in VIEWS:
            k = key + suffix
            if k not in ref:
                continue
            cells = []
            for epoch in sorted(joint_rows):
                value = joint_rows[epoch].get(k)
                cells.append(
                    "n/a" if value is None else f"{value:.4f} ({value - ref[k]:+.4f})"
                )
            lines.append(f"| {label} | {view} | {ref[k]:.4f} | " + " | ".join(cells) + " |")
    return lines


def interference_table(root: Path) -> list[str]:
    directory = _phase_dir(root, "phase1c")
    rows = _rows(directory / "grad_interference.jsonl") if directory else []
    if not rows:
        return ["(no gradient-interference measurements)"]
    groups = sorted({k for r in rows for k, v in r.items() if isinstance(v, dict)})
    lines = [
        f"{len(rows)} measurements, updates {rows[0]['update']}..{rows[-1]['update']}",
        "",
        "| parameter group | cosine mean | cosine min | cosine max | norm ratio align/cls mean | median |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for group in groups:
        cos = [r[group]["cosine"] for r in rows if isinstance(r.get(group), dict)]
        ratio = [
            r[group]["norm_ratio_align_over_cls"] for r in rows if isinstance(r.get(group), dict)
        ]
        lines.append(
            f"| {group} | {statistics.fmean(cos):+.4f} | {min(cos):+.4f} | {max(cos):+.4f} | "
            f"{statistics.fmean(ratio):.3f} | {statistics.median(ratio):.3f} |"
        )
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args(argv)
    root = args.root.expanduser()
    out = ["## ITC gate per epoch", "", *gate_table(root, ["phase1a", "phase1c"]), ""]
    out += ["## Classification on val (study_presence, q_pos) vs end of phase 1b", ""]
    out += classification_table(root) + [""]
    out += ["## Gradient interference in phase 1c", ""] + interference_table(root)
    print("\n".join(out))


if __name__ == "__main__":
    main()
