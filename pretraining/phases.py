"""Three-phase representation learning, as the META-CXR paper describes it.

Edirisinghe et al., IEEE Access 2025, "Model pre-training and fine tuning ->
Representation learning phase":

* **1a** -- MHCAC frozen; only ITC, ITG (= LM here) and ITM train the
  META-Former query tokens.
* **1b** -- MHCAC progressively unfrozen while META-Former is progressively
  frozen; LR warm-up 5e-5 -> 2e-4, L2, dropout 0.2, cosine 2e-4 -> 1e-5 over
  five epochs.
* **1c** -- once classification has converged, META-Former and the encoders'
  projection heads are unfrozen and everything trains on a weighted sum.

The source of truth is the paper, not DasithEdirisinghe/META-CXR, whose code
comments out ITC/ITM/ITG in every commit.

Each phase is ONE launch of ``pretraining.train`` with ``run.phase=<name>``;
``scripts/run_stage1_phases.sh`` chains them. A phase is a block under
``run.phases.<name>`` of the run YAML:

    run:
      phase: null                 # set on the command line
      phases:
        phase1a:
          epochs: 4
          init_from: null         # or {phase: phase1a} -> <run.phase_root>/checkpoint_phase1a.pth
          trainable: [query_tokens, Qformer, ...]   # module-name prefixes
          transition: null        # or {fade_in: [...], fade_out: [...], fraction: 0.1}
          itc_gate: {every_epoch: true, stop_after_epochs: 2}
          grad_interference: null # or {every_updates: 200, groups: {...}}
          run: {...}              # overrides merged into run.*
          model: {...}            # overrides merged into model.*
          datasets: {...}         # overrides merged into datasets.*

With no ``run.phases`` block nothing here runs and the recipe is a single
phase exactly as before. With a block, ``run.phase`` must name one of its
entries -- a phased YAML is never silently trained as one phase.

Pure Python + OmegaConf; torch only inside the functions that touch a model,
so the schedule arithmetic is testable on a CPU box.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PHASE_KEYS = frozenset(
    {
        "epochs",
        "init_from",
        "trainable",
        "transition",
        "itc_gate",
        "grad_interference",
        "unfreeze_encoder_blocks",
        "run",
        "model",
        "datasets",
        "description",
    }
)


class PhaseConfigError(ValueError):
    """The ``run.phases`` block is malformed."""


@dataclass(frozen=True)
class Transition:
    """Phase-1b hand-over: LR of ``fade_in`` 0 -> 1, of ``fade_out`` 1 -> 0.

    Over the first ``fraction`` of the phase's optimizer updates; at the end
    ``fade_out`` is set ``requires_grad=False`` and leaves the optimizer.
    """

    fade_in: tuple[str, ...]
    fade_out: tuple[str, ...]
    fraction: float = 0.1

    def steps(self, total_updates: int) -> int:
        return max(1, int(round(self.fraction * int(total_updates))))


@dataclass(frozen=True)
class PhaseSpec:
    name: str
    epochs: int
    trainable: tuple[str, ...]
    #: Union of every phase's trainable prefixes. Checkpoints keep these even
    #: when frozen now, so e.g. the Q-Former trained in 1a survives the end of
    #: 1b, where it is frozen.
    keep: tuple[str, ...]
    init_from: dict | None = None
    transition: Transition | None = None
    itc_gate: dict | None = None
    grad_interference: dict | None = None
    unfreeze_encoder_blocks: bool = False
    index: int = 0
    names: tuple[str, ...] = field(default_factory=tuple)


def name_matches(name: str, prefix: str) -> bool:
    """``prefix`` is a module path: it matches itself and everything under it."""
    return name == prefix or name.startswith(prefix + ".")


def matches_any(name: str, prefixes) -> bool:
    return any(name_matches(name, p) for p in prefixes)


def _as_tuple(value, what):
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    try:
        items = tuple(str(v) for v in value)
    except TypeError as exc:
        raise PhaseConfigError(f"{what} must be a list of module prefixes") from exc
    if any(not item or item.endswith(".") for item in items):
        raise PhaseConfigError(f"{what}: write prefixes without a trailing dot, got {items}")
    return items


def _to_plain(cfg_node):
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(cfg_node):
            return OmegaConf.to_container(cfg_node, resolve=True)
    except ImportError:  # pragma: no cover
        pass
    return cfg_node


def parse_phases(phases_cfg) -> dict[str, dict]:
    phases = _to_plain(phases_cfg) or {}
    if not isinstance(phases, dict) or not phases:
        raise PhaseConfigError("run.phases must be a non-empty mapping")
    for name, block in phases.items():
        if not isinstance(block, dict):
            raise PhaseConfigError(f"run.phases.{name} must be a mapping")
        unknown = set(block) - PHASE_KEYS
        if unknown:
            raise PhaseConfigError(f"run.phases.{name} has unknown keys {sorted(unknown)}")
        if int(block.get("epochs", 0)) < 1:
            raise PhaseConfigError(f"run.phases.{name}.epochs must be >= 1")
        if not _as_tuple(block.get("trainable"), f"run.phases.{name}.trainable"):
            raise PhaseConfigError(f"run.phases.{name}.trainable is empty")
    return phases


def build_spec(phases_cfg, phase_name: str) -> PhaseSpec:
    phases = parse_phases(phases_cfg)
    if phase_name not in phases:
        raise PhaseConfigError(
            f"run.phase={phase_name!r} is not one of {list(phases)}"
        )
    block = phases[phase_name]
    trainable = _as_tuple(block["trainable"], f"run.phases.{phase_name}.trainable")
    keep = tuple(
        sorted(
            {
                prefix
                for other in phases.values()
                for prefix in _as_tuple(other.get("trainable"), "trainable")
            }
        )
    )
    transition = None
    if block.get("transition"):
        t = block["transition"]
        transition = Transition(
            fade_in=_as_tuple(t.get("fade_in"), "transition.fade_in"),
            fade_out=_as_tuple(t.get("fade_out"), "transition.fade_out"),
            fraction=float(t.get("fraction", 0.1)),
        )
        if not transition.fade_in or not transition.fade_out:
            raise PhaseConfigError("transition needs both fade_in and fade_out")
        if not 0.0 < transition.fraction <= 1.0:
            raise PhaseConfigError("transition.fraction must be in (0, 1]")
        stray = [p for p in transition.fade_in + transition.fade_out if p not in trainable]
        if stray:
            raise PhaseConfigError(
                f"transition names prefixes that are not trainable in {phase_name}: {stray}"
            )
    init_from = block.get("init_from")
    if init_from is not None and not isinstance(init_from, dict):
        raise PhaseConfigError("init_from must be a mapping like {phase: phase1a}")
    if isinstance(init_from, dict) and init_from.get("phase") not in (None, *phases):
        raise PhaseConfigError(f"init_from.phase {init_from.get('phase')!r} is not a phase")
    names = tuple(phases)
    return PhaseSpec(
        name=phase_name,
        epochs=int(block["epochs"]),
        trainable=trainable,
        keep=keep,
        init_from=init_from,
        transition=transition,
        itc_gate=block.get("itc_gate"),
        grad_interference=block.get("grad_interference"),
        unfreeze_encoder_blocks=bool(block.get("unfreeze_encoder_blocks", False)),
        index=names.index(phase_name),
        names=names,
    )


def apply_phase_to_config(config, phase_name: str | None):
    """Merge the chosen phase into an OmegaConf config in place.

    Returns the :class:`PhaseSpec`, or ``None`` for an unphased YAML.
    """
    from omegaconf import OmegaConf

    phases_cfg = OmegaConf.select(config, "run.phases", default=None)
    if phases_cfg is None:
        if phase_name:
            raise PhaseConfigError(f"run.phase={phase_name!r} but the YAML has no run.phases")
        return None
    if not phase_name:
        raise PhaseConfigError(
            "this YAML defines run.phases; pass run.phase=<name> "
            f"(one of {list(_to_plain(phases_cfg))})"
        )
    spec = build_spec(phases_cfg, phase_name)
    block = _to_plain(phases_cfg)[phase_name]
    if block.get("model"):
        config.model = OmegaConf.merge(config.model, OmegaConf.create(block["model"]))
    if block.get("run"):
        config.run = OmegaConf.merge(config.run, OmegaConf.create(block["run"]))
    if block.get("datasets"):
        config.datasets = OmegaConf.merge(
            config.datasets, OmegaConf.create(block["datasets"])
        )
    OmegaConf.update(config, "run.max_epoch", spec.epochs, merge=False)
    if not spec.unfreeze_encoder_blocks:
        # Encoders stay frozen in every phase unless the phase says otherwise:
        # the paper unfreezes projection heads in 1c, not encoder blocks.
        OmegaConf.update(config, "model.encoder_finetune.enabled", False, merge=True)
    return spec


def resolve_init_checkpoint(spec: PhaseSpec, run_root: str | Path) -> Path | None:
    """``init_from: {phase: X}`` -> ``<run.phase_root>/checkpoint_<X>.pth``.

    The runner writes every phase's final checkpoint there. ``init_from:
    {path: ...}`` names a file directly.
    """
    if not spec.init_from:
        return None
    if spec.init_from.get("path"):
        return Path(str(spec.init_from["path"])).expanduser()
    if not run_root:
        raise PhaseConfigError(
            f"phase {spec.name} starts from phase {spec.init_from['phase']!r}; "
            "set run.phase_root to the directory holding its checkpoint"
        )
    source = spec.init_from["phase"]
    return Path(run_root) / f"checkpoint_{source}.pth"


def apply_trainable(model, spec: PhaseSpec, never_train=()):
    """Set ``requires_grad`` from the phase's prefixes; returns counts per module.

    Everything not matched is frozen -- including the vision encoders, which
    therefore stay frozen unless a phase lists them. ``never_train`` wins over
    the prefixes (the pinned ITC temperature must not be switched back on).
    """
    counts: dict[str, int] = {}
    for name, param in model.named_parameters():
        train = matches_any(name, spec.trainable) and not matches_any(name, never_train)
        param.requires_grad_(train)
        if train:
            top = name.split(".", 1)[0]
            counts[top] = counts.get(top, 0) + param.numel()
    if not counts:
        raise PhaseConfigError(f"phase {spec.name} leaves no parameter trainable")
    return counts


def param_role(name: str, transition: Transition | None) -> str:
    if transition is None:
        return "steady"
    if matches_any(name, transition.fade_in):
        return "fade_in"
    if matches_any(name, transition.fade_out):
        return "fade_out"
    return "steady"


def transition_multipliers(update: int, transition_steps: int) -> dict[str, float]:
    """LR multipliers after ``update`` optimizer updates (0-based).

    Linear: fade_in goes 0 -> 1 and fade_out 1 -> 0 over ``transition_steps``;
    both stay at their end value afterwards.
    """
    steps = max(1, int(transition_steps))
    progress = min(max(float(update) / steps, 0.0), 1.0)
    return {"fade_in": progress, "fade_out": 1.0 - progress, "steady": 1.0}


def checkpoint_keep(name: str, requires_grad: bool, spec: PhaseSpec | None) -> bool:
    """Which parameters a checkpoint stores.

    Unphased: exactly what it always did -- trainable parameters only. Phased:
    also anything trainable in ANY phase, so a module frozen by this phase
    (the Q-Former at the end of 1b) is not dropped from the file.
    """
    if requires_grad:
        return True
    return spec is not None and matches_any(name, spec.keep)
