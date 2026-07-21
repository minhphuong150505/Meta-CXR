"""Wall-clock GPU budget controller for inference runs.

Cost is derived from elapsed wall-clock time and a configured hourly rate --
the rented GPU bills for time, not for samples, so an idle or stalled run costs
exactly as much as a productive one and must be measured the same way.

The controller only ever *stops* a run. It never escalates: reaching the budget
does not fall back to a cheaper model, and finishing Findings does not enable
Impression.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

SECONDS_PER_HOUR = 3600.0


class BudgetExceeded(RuntimeError):
    """Raised when the projected spend has reached the configured ceiling."""


@dataclass
class BudgetState:
    """Tracks elapsed time and spend for a single inference run.

    ``prior_elapsed_seconds`` carries the runtime already spent by earlier
    attempts at the same run. Without it a run could be resumed indefinitely,
    each attempt starting from a zeroed budget, and the ceiling would never
    bind.
    """

    hourly_cost_usd: float
    budget_limit_usd: float
    max_runtime_hours: float = 0.0
    prior_elapsed_seconds: float = 0.0
    processed_samples: int = 0
    # Injected in tests so budget behaviour is verifiable without real waiting.
    clock: Callable[[], float] = field(default=time.monotonic, repr=False)
    start_time: float | None = None

    def __post_init__(self) -> None:
        if self.start_time is None:
            self.start_time = self.clock()

    @property
    def elapsed_seconds(self) -> float:
        """Cumulative runtime for this run, including previous attempts."""
        session = max(0.0, float(self.clock()) - float(self.start_time))
        return session + max(0.0, float(self.prior_elapsed_seconds))

    @property
    def elapsed_hours(self) -> float:
        return self.elapsed_seconds / SECONDS_PER_HOUR

    @property
    def estimated_cost_usd(self) -> float:
        return self.elapsed_hours * self.hourly_cost_usd

    @property
    def remaining_budget_usd(self) -> float:
        return max(0.0, self.budget_limit_usd - self.estimated_cost_usd)

    @property
    def samples_per_second(self) -> float:
        elapsed = self.elapsed_seconds
        if elapsed <= 0 or self.processed_samples <= 0:
            return 0.0
        return self.processed_samples / elapsed

    def record_samples(self, count: int = 1) -> None:
        self.processed_samples += count

    def assert_within_budget(self) -> None:
        """Raise before starting more work that the budget cannot cover.

        Called *between* batches, never mid-sample, so the caller can always
        flush a complete prediction and a consistent progress file.
        """
        if self.budget_limit_usd > 0 and self.estimated_cost_usd >= self.budget_limit_usd:
            raise BudgetExceeded(
                f"estimated spend ${self.estimated_cost_usd:.2f} reached the "
                f"${self.budget_limit_usd:.2f} limit after "
                f"{self.processed_samples} samples "
                f"({self.elapsed_hours:.2f}h at ${self.hourly_cost_usd:.2f}/h). "
                "Stopping cleanly; rerun to resume from the progress file."
            )
        if self.max_runtime_hours > 0 and self.elapsed_hours >= self.max_runtime_hours:
            raise BudgetExceeded(
                f"elapsed {self.elapsed_hours:.2f}h reached the "
                f"{self.max_runtime_hours:.2f}h runtime ceiling after "
                f"{self.processed_samples} samples. Stopping cleanly."
            )

    def project(self, target_samples: int) -> dict[str, float]:
        """Extrapolate a full-split run from the rate measured so far.

        Returns zeroed projections until at least one sample has completed --
        this project does not publish invented throughput numbers.
        """
        rate = self.samples_per_second
        if rate <= 0 or target_samples <= 0:
            return {
                "samples_per_second": 0.0,
                "projected_hours": 0.0,
                "projected_cost_usd": 0.0,
            }
        projected_hours = (target_samples / rate) / SECONDS_PER_HOUR
        return {
            "samples_per_second": rate,
            "projected_hours": projected_hours,
            "projected_cost_usd": projected_hours * self.hourly_cost_usd,
        }

    def progress_line(self, target_samples: int = 0) -> str:
        """One-line heartbeat for the periodic log."""
        projection = self.project(target_samples)
        return (
            f"[budget] {self.processed_samples} samples "
            f"| {self.elapsed_hours:.3f}h "
            f"| {self.samples_per_second:.3f} sample/s "
            f"| spent ${self.estimated_cost_usd:.3f} "
            f"| remaining ${self.remaining_budget_usd:.3f}"
            + (
                f" | projected {projection['projected_hours']:.2f}h "
                f"/ ${projection['projected_cost_usd']:.2f} for {target_samples}"
                if target_samples > 0
                else ""
            )
        )
