"""CPU-only checks for the read-only training healthcheck."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_healthcheck.sh"


def _run_healthcheck(tmp_path: Path, **extra_env: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "RUN_DIR": str(tmp_path),
            "LOG": "",
            "STALL_MIN": "45",
            **extra_env,
        }
    )
    return subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_expected_run_that_is_absent_is_an_alert(tmp_path: Path):
    result = _run_healthcheck(tmp_path, EXPECT_RUNNING="1")

    assert result.returncode == 3
    assert "NONE RUNNING (but EXPECT_RUNNING=1)" in result.stdout
    assert "=== ALERT: act now ===" in result.stdout


def test_stage2_recovery_artifact_is_tracked_without_fractional_timestamp_error(
    tmp_path: Path,
):
    checkpoint = tmp_path / "checkpoints" / "last" / "trainer_state.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"recovery")

    result = _run_healthcheck(tmp_path)

    assert result.returncode == 4
    assert "last checkpoint" in result.stdout
    assert "trainer_state.pt" in result.stdout
    assert "checkpoints            1 files" in result.stdout
    assert "arithmetic syntax error" not in result.stderr
