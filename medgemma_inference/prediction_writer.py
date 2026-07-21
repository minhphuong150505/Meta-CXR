"""Append-only JSONL prediction writer with crash-safe resume.

Each record is flushed and fsynced as it is written, so a killed run leaves a
file whose complete lines are all valid. If the process died partway through a
line, that trailing fragment is truncated on resume rather than being left to
break the next reader.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

PREDICTIONS_FILENAME = "predictions.jsonl"

#: Never written to an artifact. Kept as an explicit guard so a future edit that
#: adds one of these to the record fails a test instead of leaking quietly.
FORBIDDEN_FIELDS = frozenset(
    {"subject_id", "study_id", "dicom_id", "image_path", "ref", "reference", "report"}
)


class PrivacyViolation(RuntimeError):
    """A record carried an identifier or reference report."""


def assert_publishable(record: dict[str, Any]) -> None:
    """Reject records carrying identifiers, paths or reference text."""
    present = sorted(FORBIDDEN_FIELDS.intersection(record))
    if present:
        raise PrivacyViolation(
            f"prediction record carries restricted field(s): {', '.join(present)}. "
            "MIMIC-CXR identifiers, image paths and reference reports must not be "
            "written to evaluation artifacts."
        )


def read_completed_keys(path: str | Path) -> set[str]:
    """Return sample keys already written, truncating any partial trailing line.

    Repairing the file here (rather than at write time) means the repair happens
    exactly once per resume, and the file is left in a state every reader can
    parse.
    """
    file_path = Path(path)
    if not file_path.is_file():
        return set()
    completed: set[str] = set()
    valid_bytes = 0
    with file_path.open("rb") as handle:
        for raw_line in handle:
            if not raw_line.endswith(b"\n"):
                # Partial final line from a killed process; drop it.
                break
            try:
                record = json.loads(raw_line.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                break
            key = record.get("sample_key")
            if key:
                completed.add(str(key))
            valid_bytes += len(raw_line)
    if valid_bytes != file_path.stat().st_size:
        with file_path.open("r+b") as handle:
            handle.truncate(valid_bytes)
    return completed


class PredictionWriter:
    """Appends prediction records as JSONL, one flushed line at a time."""

    def __init__(self, output_dir: str | Path) -> None:
        self.directory = Path(output_dir)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / PREDICTIONS_FILENAME
        self.completed_keys = read_completed_keys(self.path)
        self._handle = None

    def __enter__(self) -> PredictionWriter:
        self._handle = self.path.open("a", encoding="utf-8")
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()
            self._handle = None

    def already_done(self, sample_key: str) -> bool:
        return sample_key in self.completed_keys

    def write(self, record: dict[str, Any]) -> None:
        if self._handle is None:
            raise RuntimeError("PredictionWriter used outside its context manager.")
        assert_publishable(record)
        self._handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        # Durability per record: the budget may stop this run at any batch
        # boundary, and everything already generated has been paid for.
        self._handle.flush()
        os.fsync(self._handle.fileno())
        key = record.get("sample_key")
        if key:
            self.completed_keys.add(str(key))
