"""Atomic checkpoint save/load for resumable training.

Two properties matter more than anything else here:

1. **Atomicity.** A run killed mid-``torch.save`` must not leave a truncated
   file that loads as garbage or half-loads and then throws. Everything is
   written to a temporary path and renamed, which is atomic on POSIX.
2. **Completeness.** A checkpoint that restores weights but not the optimizer,
   scheduler, scaler and RNG streams does not resume a run -- it starts a new
   one from those weights. ``verify_resumable`` refuses a partial directory
   rather than letting the run silently restart its counters.

Knows nothing about MedGemma, LoRA or MIMIC-CXR: it takes state dicts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from training.torch_io import load_torch_checkpoint
from training.trainer.state import RngSnapshot, TrainingState

TRAINER_STATE_FILE = "trainer_state.pt"


def atomic_save(payload: Any, path: Path) -> None:
    """Write via a temporary file and rename, so readers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


class CheckpointManager:
    """Saves and restores the parts of a run that are not model weights.

    Adapter weights are saved separately by the reporter (``save_pretrained``),
    because their format is PEFT's business. This manager owns the optimizer,
    scheduler, AMP scaler and ``TrainingState``.
    """

    def __init__(self, root: Path | str):
        self.root = Path(root)

    def state_path(self, subdir: str = "") -> Path:
        base = self.root / subdir if subdir else self.root
        return base / TRAINER_STATE_FILE

    def save(
        self,
        state: TrainingState,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any = None,
        scaler: Any = None,
        data_generator: torch.Generator | None = None,
        extra: dict[str, Any] | None = None,
        subdir: str = "",
    ) -> Path:
        state.rng = RngSnapshot.capture(data_generator)
        payload = {
            "state": state.to_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            **(extra or {}),
        }
        path = self.state_path(subdir)
        atomic_save(payload, path)
        return path

    def load(
        self,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any = None,
        scaler: Any = None,
        data_generator: torch.Generator | None = None,
        subdir: str = "",
    ) -> TrainingState:
        path = self.state_path(subdir)
        if not path.is_file():
            raise FileNotFoundError(f"no trainer state to resume from: {path}")
        payload = load_torch_checkpoint(path)
        state = TrainingState.from_dict(payload["state"])

        for component, key in (
            (optimizer, "optimizer"),
            (scheduler, "scheduler"),
            (scaler, "scaler"),
        ):
            if component is None:
                continue
            saved = payload.get(key)
            if saved is None:
                raise ValueError(
                    f"checkpoint {path} has no {key} state but one was supplied to "
                    "resume. Continuing would reset it and silently change the "
                    "optimisation trajectory."
                )
            component.load_state_dict(saved)

        if state.rng is not None:
            state.rng.restore(data_generator)
        return state

    def is_resumable(self, subdir: str = "") -> bool:
        return self.state_path(subdir).is_file()

    def verify_resumable(self, required_files: tuple[str, ...] = (), subdir: str = "") -> None:
        """Fail before training starts if the resume directory is incomplete."""
        base = self.root / subdir if subdir else self.root
        missing = [name for name in (TRAINER_STATE_FILE, *required_files) if not (base / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"incomplete resume checkpoint {base}: missing {', '.join(missing)}"
            )
