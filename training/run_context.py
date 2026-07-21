"""Immutable Stage-1 identity for a Stage-2 run.

The entry point used to configure the Figure-9 module by assigning to its
globals from outside::

    fig9.RUN_NAME = args.stage1_run
    fig9.THRESHOLDS = fig9.load_thresholds(args.threshold_path)

That made the run un-snapshottable, let two runs in one process interfere, and
meant the evaluation fingerprint depended on import-time state rather than on an
argument. ``Stage1Context`` carries the same four values as one frozen object
that is passed explicitly.

stdlib only, so it stays importable and testable without torch or LAVIS.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType


@dataclass(frozen=True)
class Stage1Context:
    """Which Stage-1 run a Stage-2 run is built on, and how it classifies."""

    run_name: str
    config_path: Path | None = None
    checkpoint_path: Path | None = None
    # Read-only view: a frozen dataclass would otherwise still hand out a dict
    # that a caller could mutate, which is the defect this class exists to fix.
    thresholds: Mapping[str, Mapping[str, float]] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        if not str(self.run_name).strip():
            raise ValueError("run_name must not be empty")
        object.__setattr__(self, "run_name", str(self.run_name))
        if self.config_path is not None:
            object.__setattr__(self, "config_path", Path(self.config_path))
        if self.checkpoint_path is not None:
            object.__setattr__(self, "checkpoint_path", Path(self.checkpoint_path))
        object.__setattr__(
            self,
            "thresholds",
            MappingProxyType(
                {
                    str(name): MappingProxyType(dict(values))
                    for name, values in dict(self.thresholds).items()
                }
            ),
        )

    def resolve_config_path(self, default: Path) -> Path:
        """Explicit --stage1-config wins; otherwise the caller's default."""
        return self.config_path if self.config_path is not None else Path(default)

    def resolve_checkpoint_path(self, checkpoint_root: Path) -> Path:
        """Explicit --stage1-checkpoint wins; otherwise <root>/<run>/checkpoint_best.pth."""
        if self.checkpoint_path is not None:
            return self.checkpoint_path
        return Path(checkpoint_root) / self.run_name / "checkpoint_best.pth"

    def threshold_for(self, abnormality: str) -> Mapping[str, float]:
        return self.thresholds.get(abnormality, MappingProxyType({}))

    def fingerprint_payload(self) -> dict:
        """JSON-safe view for cohort/eval fingerprints."""
        return {
            "run_name": self.run_name,
            "config_path": str(self.config_path) if self.config_path else None,
            "checkpoint_path": (
                str(self.checkpoint_path) if self.checkpoint_path else None
            ),
            "thresholds": {
                name: dict(values) for name, values in self.thresholds.items()
            },
        }
