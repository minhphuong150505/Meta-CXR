"""Optional clinical-metric adapters.

**No clinical metric is implemented in this repository.** RadGraph, CheXbert and
CheXpert-labeler all require model weights and, in CheXpert's case, a Java
runtime; none is vendored here and none has been run. This module exists so that
asking for one produces an honest error naming the missing dependency, instead
of either an import-time crash or -- much worse -- a lexical number reported
under a clinical name.

The rules this module enforces:

1. No optional dependency is imported at module import time.
2. A missing dependency raises ``MissingOptionalDependency`` naming the install
   command. It never returns a placeholder score.
3. There is no silent fallback from a clinical metric to a lexical one. A
   lexical metric may only be requested by its own lexical name.
4. Checkpoint selection may not name a metric that is unavailable; config
   validation fails first, so a run cannot train for hours and then discover it
   has nothing to select on.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

WARN = "warn"
ERROR = "error"
SKIP = "skip"
MISSING_DEPENDENCY_POLICIES = (WARN, ERROR, SKIP)


class MissingOptionalDependency(RuntimeError):
    """A clinical metric was requested but its dependency is not installed."""

    def __init__(self, metric: str, package: str, install: str):
        self.metric = metric
        self.package = package
        self.install = install
        super().__init__(
            f"clinical metric {metric!r} requires the {package!r} package, which is "
            f"not installed. Install it with: {install}. This metric has no lexical "
            "fallback -- BLEU/ROUGE/CIDEr/BERTScore measure wording, not clinical "
            "correctness, and reporting one under a clinical name would be wrong."
        )


@runtime_checkable
class ClinicalMetric(Protocol):
    name: str

    def available(self) -> bool: ...

    def compute(
        self, predictions: Sequence[str], references: Sequence[str]
    ) -> dict[str, Any]: ...


@dataclass
class _DependencyBackedMetric:
    """Common shape: a name, an importable package, and an install command."""

    name: str
    package: str
    install: str

    def available(self) -> bool:
        import importlib.util

        return importlib.util.find_spec(self.package) is not None

    def _require(self):
        if not self.available():
            raise MissingOptionalDependency(self.name, self.package, self.install)
        import importlib

        return importlib.import_module(self.package)

    def compute(
        self, predictions: Sequence[str], references: Sequence[str]
    ) -> dict[str, Any]:
        if len(predictions) != len(references):
            raise ValueError(
                f"{self.name}: {len(predictions)} predictions vs {len(references)} references"
            )
        self._require()
        # Reached only when the dependency is genuinely installed. Wiring the
        # real scorer is deliberately left undone rather than approximated:
        # nothing in this repo has ever run these models, and a plausible-looking
        # implementation that has never been executed against reference outputs
        # would be worse than an explicit gap.
        raise NotImplementedError(
            f"{self.package} is installed but the {self.name} adapter has not been "
            "wired to it. Implement it against the installed API and validate "
            "against published reference scores before reporting any number."
        )


class RadGraphMetric(_DependencyBackedMetric):
    def __init__(self) -> None:
        super().__init__(
            name="radgraph",
            package="radgraph",
            install="pip install radgraph",
        )


class CheXbertMetric(_DependencyBackedMetric):
    def __init__(self) -> None:
        super().__init__(
            name="chexbert",
            package="chexbert",
            install="pip install chexbert  # plus the CheXbert checkpoint",
        )


class CheXpertLabelMetric(_DependencyBackedMetric):
    def __init__(self) -> None:
        super().__init__(
            name="chexpert_labeler",
            package="chexpert_labeler",
            install="see docs/SETUP_GUIDE.md -- also needs a Java runtime",
        )


REGISTRY: dict[str, type] = {
    "radgraph": RadGraphMetric,
    "chexbert": CheXbertMetric,
    "chexpert_labeler": CheXpertLabelMetric,
}


def build_metric(name: str) -> ClinicalMetric:
    if name not in REGISTRY:
        raise ValueError(
            f"unknown clinical metric {name!r}; available: {sorted(REGISTRY)}"
        )
    return REGISTRY[name]()


def resolve_metrics(
    names: Sequence[str], policy: str = WARN
) -> tuple[list[ClinicalMetric], list[str]]:
    """Instantiate the requested metrics, applying the missing-dependency policy.

    Returns ``(usable, warnings)``. Under ``error`` a missing dependency raises;
    under ``warn`` and ``skip`` it is dropped, the difference being whether a
    message is produced.
    """
    if policy not in MISSING_DEPENDENCY_POLICIES:
        raise ValueError(
            f"missing_dependency_policy must be one of {MISSING_DEPENDENCY_POLICIES}, "
            f"got {policy!r}"
        )
    usable: list[ClinicalMetric] = []
    warnings: list[str] = []
    for name in names:
        metric = build_metric(name)
        if metric.available():
            usable.append(metric)
            continue
        if policy == ERROR:
            raise MissingOptionalDependency(metric.name, metric.package, metric.install)
        if policy == WARN:
            warnings.append(
                f"clinical metric {name!r} skipped: {metric.package!r} is not installed "
                f"({metric.install})"
            )
    return usable, warnings


def validate_selection_metric(selection_metric: str, available: Sequence[str]) -> None:
    """Fail before training if checkpoints would be selected on a missing metric.

    Discovering at the end of a multi-hour run that the selection metric was
    never computed means the run produced no defensible checkpoint choice.
    """
    if selection_metric in REGISTRY and selection_metric not in available:
        raise ValueError(
            f"selection metric {selection_metric!r} is a clinical metric whose "
            f"dependency is unavailable (usable: {sorted(available)}). Install it or "
            "select on an available metric; refusing to start a run that cannot "
            "evaluate its own checkpoint-selection criterion."
        )
