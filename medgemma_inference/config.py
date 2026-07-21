"""Config schema for external-MedGemma inference experiments.

Scope note: this validator runs on ``configs/experiments/*.yaml`` only. Stage-1
META-CXR/MHCAC training configs under ``pretraining/configs/`` are a separate,
still-supported namespace and are never parsed here -- their ``learning_rate``,
``optimizer`` and ``warmup_steps`` keys remain entirely legitimate.

What is rejected here is *MedGemma* fine-tuning configuration, which no longer
has an implementation behind it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PIPELINE_MODE = "pretrained_medgemma_findings_first"

#: MedGemma fine-tuning keys. Silently ignoring these would let someone think
#: they had configured a training run that no longer exists.
OBSOLETE_FINETUNING_KEYS = frozenset(
    {
        "training",
        "epochs",
        "learning_rate",
        "weight_decay",
        "optimizer",
        "scheduler",
        "warmup_steps",
        "gradient_accumulation_steps",
        "max_grad_norm",
        "early_stopping",
        "save_optimizer",
        "save_scheduler",
        "resume_training",
        "lora_rank",
        "lora_alpha",
        "lora_dropout",
        "target_modules",
        "ddp",
        "world_size",
        "local_rank",
    }
)

OBSOLETE_MESSAGE = (
    "This project now runs externally fine-tuned MedGemma checkpoints in "
    "inference-only mode. MedGemma fine-tuning configuration is no longer "
    "supported in the active pipeline. (Stage-1 META-CXR/MHCAC training config "
    "is unaffected and lives under pretraining/configs/.)"
)


class ConfigError(ValueError):
    """The experiment config is structurally invalid."""


class ObsoleteFineTuningConfigError(ConfigError):
    """The config carries MedGemma fine-tuning keys that no longer do anything."""


@dataclass(frozen=True)
class FindingsModelConfig:
    model_id: str = "erjui/medgemma-4b-srrg-findings"
    revision: str = "main"
    enabled: bool = True
    max_new_tokens: int = 512
    do_sample: bool = False
    num_beams: int = 1
    device: str = "auto"
    dtype: str = "auto"
    load_in_4bit: bool = False


@dataclass(frozen=True)
class ImpressionModelConfig:
    model_id: str = "erjui/medgemma-4b-srrg-impression"
    enabled: bool = False
    max_new_tokens: int = 256
    do_sample: bool = False
    num_beams: int = 1


@dataclass(frozen=True)
class RuntimeConfig:
    hourly_cost_usd: float = 0.40
    budget_limit_usd: float = 20.0
    max_runtime_hours: float = 50.0
    stop_on_budget_limit: bool = True
    load_one_model_at_a_time: bool = True
    log_every: int = 10


@dataclass(frozen=True)
class EvaluationConfig:
    section: str = "findings"
    save_predictions: bool = True
    run_counterfactual_audit: bool = True
    run_impression: bool = False


@dataclass(frozen=True)
class PrivacyConfig:
    save_identifiers: bool = False
    save_absolute_paths: bool = False
    save_reference_reports: bool = False


@dataclass(frozen=True)
class ExperimentConfig:
    pipeline_mode: str = PIPELINE_MODE
    findings: FindingsModelConfig = field(default_factory=FindingsModelConfig)
    impression: ImpressionModelConfig = field(default_factory=ImpressionModelConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)


def _find_obsolete_keys(node: Any, path: str = "") -> list[str]:
    """Walk the whole tree; a fine-tuning key is obsolete at any depth."""
    found: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            here = f"{path}.{key}" if path else str(key)
            if str(key) in OBSOLETE_FINETUNING_KEYS:
                found.append(here)
            found.extend(_find_obsolete_keys(value, here))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found.extend(_find_obsolete_keys(value, f"{path}[{index}]"))
    return found


def _subset(cls, data: Any, source: str):
    """Build a frozen dataclass, rejecting unknown keys rather than dropping them."""
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise ConfigError(f"{source} must be a mapping, got {type(data).__name__}.")
    known = {f.name for f in cls.__dataclass_fields__.values()}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ConfigError(
            f"{source} has unknown key(s): {', '.join(unknown)}. "
            f"Known keys: {', '.join(sorted(known))}."
        )
    return cls(**data)


def parse_config(raw: dict[str, Any], source: str = "<dict>") -> ExperimentConfig:
    """Validate a raw config mapping into an ExperimentConfig."""
    if not isinstance(raw, dict):
        raise ConfigError(f"{source} must be a mapping.")

    obsolete = _find_obsolete_keys(raw)
    if obsolete:
        raise ObsoleteFineTuningConfigError(
            f"{source} contains obsolete MedGemma fine-tuning key(s): "
            f"{', '.join(sorted(obsolete))}. {OBSOLETE_MESSAGE}"
        )

    pipeline = raw.get("pipeline") or {}
    if not isinstance(pipeline, dict):
        raise ConfigError(f"{source}: 'pipeline' must be a mapping.")
    mode = pipeline.get("mode", PIPELINE_MODE)
    if mode != PIPELINE_MODE:
        raise ConfigError(
            f"{source}: pipeline.mode is {mode!r}; this runner only implements "
            f"{PIPELINE_MODE!r}."
        )

    models = raw.get("models") or {}
    if not isinstance(models, dict):
        raise ConfigError(f"{source}: 'models' must be a mapping.")

    config = ExperimentConfig(
        pipeline_mode=mode,
        findings=_subset(
            FindingsModelConfig, models.get("findings"), f"{source}: models.findings"
        ),
        impression=_subset(
            ImpressionModelConfig,
            models.get("impression"),
            f"{source}: models.impression",
        ),
        runtime=_subset(RuntimeConfig, raw.get("runtime"), f"{source}: runtime"),
        evaluation=_subset(
            EvaluationConfig, raw.get("evaluation"), f"{source}: evaluation"
        ),
        privacy=_subset(PrivacyConfig, raw.get("privacy"), f"{source}: privacy"),
    )
    validate_config(config, source)
    return config


def validate_config(config: ExperimentConfig, source: str = "<config>") -> None:
    """Enforce the Phase-1 invariants."""
    if not config.findings.enabled:
        raise ConfigError(
            f"{source}: models.findings.enabled must be true -- the Findings "
            "model is the only model this phase runs."
        )
    if config.impression.enabled:
        raise ConfigError(
            f"{source}: models.impression.enabled must be false during the "
            "Findings-first phase."
        )
    if config.evaluation.run_impression:
        raise ConfigError(
            f"{source}: evaluation.run_impression must be false during the "
            "Findings-first phase."
        )
    if config.evaluation.section != "findings":
        raise ConfigError(
            f"{source}: evaluation.section must be 'findings', got "
            f"{config.evaluation.section!r}."
        )
    # PhysioNet DUA: derivatives carrying identifiers or report text are
    # restricted data. The runner has no code path that writes them, so a config
    # asking for them is a mistake worth surfacing rather than ignoring.
    for flag in ("save_identifiers", "save_absolute_paths", "save_reference_reports"):
        if getattr(config.privacy, flag):
            raise ConfigError(
                f"{source}: privacy.{flag} must be false. MIMIC-CXR identifiers "
                "and report text may not be written to evaluation artifacts."
            )
    if config.runtime.hourly_cost_usd < 0 or config.runtime.budget_limit_usd < 0:
        raise ConfigError(f"{source}: runtime costs must be non-negative.")


def load_config(path: str | Path) -> ExperimentConfig:
    """Read and validate an experiment config from disk."""
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return parse_config(raw, source=str(config_path))
