"""Run identity and progress tracking for resumable inference.

A resumed run must be the *same* run. If the model, revision, generation
settings, split or dataset fingerprint changed, appending to the existing
predictions file would silently blend outputs from two different configurations
into one results set, and no downstream metric could tell them apart. That is
refused rather than merged.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROGRESS_FILENAME = "progress.json"


class ResumeMismatch(RuntimeError):
    """The on-disk run does not match the run being started."""


@dataclass(frozen=True)
class RunIdentity:
    """Everything that must match for a resume to be legitimate."""

    dataset_fingerprint: str
    split: str
    model_id: str
    model_revision: str
    max_new_tokens: int
    do_sample: bool
    num_beams: int
    section: str = "findings"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def assert_matches(self, other: RunIdentity) -> None:
        differences = [
            f"{key}: on-disk {getattr(other, key)!r} != requested {getattr(self, key)!r}"
            for key in self.to_dict()
            if getattr(other, key) != getattr(self, key)
        ]
        if differences:
            raise ResumeMismatch(
                "refusing to resume into an existing run with different settings: "
                + "; ".join(differences)
                + ". Use a different --output-dir, or delete the existing one."
            )


@dataclass
class ProgressFile:
    """Small JSON sidecar recording run identity and cumulative runtime."""

    path: Path
    identity: RunIdentity

    @classmethod
    def open(cls, output_dir: str | Path, identity: RunIdentity) -> ProgressFile:
        """Create or validate the progress file for this run."""
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        progress = cls(path=directory / PROGRESS_FILENAME, identity=identity)
        existing = progress.read()
        if existing is not None:
            identity.assert_matches(RunIdentity(**existing["identity"]))
        return progress

    def read(self) -> dict[str, Any] | None:
        if not self.path.is_file():
            return None
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except json.JSONDecodeError:
            # A progress file killed mid-write is recoverable: the predictions
            # JSONL is the real source of truth for what finished.
            return None

    def prior_elapsed_seconds(self) -> float:
        existing = self.read()
        if not existing:
            return 0.0
        return float(existing.get("elapsed_seconds", 0.0))

    def write(
        self, *, completed_samples: int, elapsed_seconds: float, finished: bool = False
    ) -> None:
        payload = {
            "identity": self.identity.to_dict(),
            "completed_samples": int(completed_samples),
            "elapsed_seconds": round(float(elapsed_seconds), 3),
            "finished": bool(finished),
        }
        # Write-then-rename so a crash never leaves a half-written progress file.
        temporary = self.path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        temporary.replace(self.path)
