"""Run state and RNG capture, so that a resumed run continues the original one.

A resume that restores only the optimizer produces a *different* run that merely
starts from the same weights: the data order restarts, dropout masks differ, and
the loss curve diverges from the uninterrupted run. That is hard to notice and
impossible to defend in a paper, so every stream that affects the trajectory is
captured here:

* Python ``random`` -- used by the record subsetting helpers.
* NumPy -- used by the metric code and by anything reaching through pandas.
* torch CPU and CUDA -- dropout, initialisation.
* the DataLoader's own ``torch.Generator`` -- shuffle order.

Provenance travels with the state because "which commit, which config, which
data" is exactly what nobody can reconstruct six months later.
"""

from __future__ import annotations

import random
import subprocess
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

STATE_VERSION = 1


@dataclass
class RngSnapshot:
    """Every RNG stream that influences the training trajectory."""

    python: Any
    numpy: Any
    torch_cpu: Any
    cuda: Any = None
    data_generator: Any = None

    @classmethod
    def capture(cls, data_generator: torch.Generator | None = None) -> RngSnapshot:
        return cls(
            python=random.getstate(),
            numpy=np.random.get_state(),
            torch_cpu=torch.get_rng_state(),
            cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            data_generator=data_generator.get_state() if data_generator is not None else None,
        )

    def restore(self, data_generator: torch.Generator | None = None) -> None:
        random.setstate(self.python)
        np.random.set_state(self.numpy)
        torch.set_rng_state(self.torch_cpu)
        # A checkpoint written on a GPU box and resumed on CPU (or on a machine
        # with a different device count) must not fail here; the CPU streams
        # above are what determine a CPU run's trajectory.
        if self.cuda is not None and torch.cuda.is_available():
            if len(self.cuda) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all(self.cuda)
        if data_generator is not None and self.data_generator is not None:
            data_generator.set_state(self.data_generator)

    def to_dict(self) -> dict:
        return {
            "python": self.python,
            "numpy": self.numpy,
            "torch_cpu": self.torch_cpu,
            "cuda": self.cuda,
            "data_generator": self.data_generator,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> RngSnapshot:
        return cls(**payload)


def git_sha(default: str = "unknown") -> str:
    """Current commit, or ``default`` outside a git checkout."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        )
    except OSError:
        # git is not installed. Recorded as unknown rather than failing a run.
        return default
    return proc.stdout.strip() if proc.returncode == 0 else default


@dataclass
class TrainingState:
    """Where the run is, how well it is doing, and what produced it."""

    epoch: int = 0
    micro_step: int = 0
    global_step: int = 0
    best_score: float = float("inf")
    #: True when a lower score is better (cross-entropy); False for F1/BLEU.
    lower_is_better: bool = True
    bad_epochs: int = 0
    rng: RngSnapshot | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def is_improvement(self, score: float) -> bool:
        return score < self.best_score if self.lower_is_better else score > self.best_score

    def record_score(self, score: float) -> bool:
        """Update best/bad-epoch bookkeeping. Returns whether it improved."""
        if self.is_improvement(score):
            self.best_score = float(score)
            self.bad_epochs = 0
            return True
        self.bad_epochs += 1
        return False

    def should_stop(self, patience: int) -> bool:
        return self.bad_epochs >= patience

    def to_dict(self) -> dict:
        return {
            "state_version": STATE_VERSION,
            "epoch": self.epoch,
            "micro_step": self.micro_step,
            "global_step": self.global_step,
            "best_score": self.best_score,
            "lower_is_better": self.lower_is_better,
            "bad_epochs": self.bad_epochs,
            "rng": self.rng.to_dict() if self.rng else None,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> TrainingState:
        version = payload.get("state_version")
        if version != STATE_VERSION:
            raise ValueError(
                f"trainer state version {version!r} cannot be read by this code "
                f"(expects {STATE_VERSION}). Resuming across an incompatible state "
                "format would silently restart counters and RNG streams."
            )
        rng = payload.get("rng")
        return cls(
            epoch=int(payload["epoch"]),
            micro_step=int(payload["micro_step"]),
            global_step=int(payload["global_step"]),
            best_score=float(payload["best_score"]),
            lower_is_better=bool(payload["lower_is_better"]),
            bad_epochs=int(payload["bad_epochs"]),
            rng=RngSnapshot.from_dict(rng) if rng else None,
            provenance=dict(payload.get("provenance") or {}),
        )


def build_provenance(
    *,
    config_snapshot: dict[str, Any] | None = None,
    dataset_fingerprint: str | None = None,
    model_revision: str | None = None,
    threshold_provenance: str | None = None,
) -> dict[str, Any]:
    """Everything needed to explain, later, what a checkpoint actually is.

    ``threshold_provenance`` is deliberately explicit: image-only argmax and a
    calibrated ``threshold.json`` produce different structured findings, and a
    checkpoint that does not record which it used cannot be compared with one
    that used the other.
    """
    return {
        "git_sha": git_sha(),
        "config_snapshot": config_snapshot or {},
        "dataset_fingerprint": dataset_fingerprint,
        "model_revision": model_revision,
        "threshold_provenance": threshold_provenance or "image_only_argmax",
    }
