"""Config keys that turned the paper's three-class task into a binary one.

META-CXR classifies every finding into exactly three classes -- Negative,
Positive, Uncertain -- and is evaluated that way (paper Sec. III-D, Fig. 5,
Fig. 10). Between 2026-08 and 2026-09 this repository grew a set of options
that quietly collapsed that into "present / not present": a binary mention
gate, a hierarchical "mention-conditioned" objective, uncertain-folding
policies, and a ``study_presence`` evaluation framing. They were removed on
2026-09-29 at the user's request (D-023), because the binary framing was never
what the paper does.

This module exists so they cannot come back silently. Any config that still
names one of these keys is refused with an explanation, instead of being
ignored (which would let a stale YAML look like it ran the old recipe).

stdlib only: imported by the training entrypoint, the model's ``from_config``
and the Stage-1 evaluation hook.
"""

from __future__ import annotations

from collections.abc import Mapping

#: ``(section, key)`` pairs. ``section`` is where the key used to live.
RETIRED_MODEL_KEYS = (
    ("loss", "lambda_gate"),
    ("loss", "lambda_mention_conditioned_cls"),
    ("mhcac", "gate_class_weights"),
    ("mhcac", "mention_conditioned_pos_weights"),
    ("mhcac", "uncertain_policy"),
)
RETIRED_RUN_KEYS = (
    "uncertain_policy",
    "report_study_presence",
    "include_meta_labels",
    "label_framing",
)

#: Checkpoint entries left by the retired heads/losses. Dropped on load so an
#: old checkpoint still restores everything else.
RETIRED_STATE_PREFIXES = (
    "mhcac.mention_heads.",
    "gate_loss_fn.",
    "mention_conditioned_loss_fn.",
)

EXPLANATION = (
    "was removed on 2026-09-29 (D-023): the model and its evaluation are "
    "three-class P/N/U as in the META-CXR paper, with no binary mention gate, "
    "no uncertain-folding policy and no study_presence framing. Delete the key; "
    "see CLAUDE.md, 'Three classes, never binary'."
)


class RetiredConfigKey(ValueError):
    """A config names an option that no longer exists."""


def _get(mapping, key):
    if mapping is None:
        return None
    if isinstance(mapping, Mapping) or hasattr(mapping, "get"):
        try:
            return mapping.get(key, None)
        except TypeError:
            return None
    return None


def reject_retired_model_keys(model_cfg) -> None:
    """Raise if ``model_cfg`` still carries a retired binary-label key."""
    for section, key in RETIRED_MODEL_KEYS:
        block = _get(model_cfg, section)
        if block is not None and _get(block, key) is not None:
            raise RetiredConfigKey(f"model.{section}.{key} {EXPLANATION}")


#: Opt-in experiment knobs removed on 2026-11-27, when the residual-lateral R1
#: pipeline became the main one. Every one failed its pre-declared pilot rule;
#: the measurements stay in CLAUDE.md and docs/handoff/. ``None`` section = a key
#: directly under ``model``.
REMOVED_EXPERIMENT_MODEL_KEYS = (
    ("view_fusion", "mode"),
    ("view_fusion", "late_anchor_weight"),
    ("view_fusion", "late_detach_aux"),
    ("view_fusion", "late_aux_head"),
    ("view_fusion", "detach_aux"),
    ("loss", "mpc_temperature"),
    ("loss", "lambda_query_diversity"),
    ("mhcac", "asymmetric_loss"),
    (None, "qformer_input_norm"),
    (None, "qformer_dropout"),
    (None, "eval_mode_modules"),
)

REMOVED_EXPERIMENT_EXPLANATION = (
    "was removed on 2026-11-27 with the failed experiments it belonged to (late "
    "/ tokens view fusion, EVOKE flags, asymmetric loss, the phase-1q Q-Former "
    "refit). Delete the key; the results are recorded in CLAUDE.md and "
    "docs/handoff/."
)


def reject_removed_experiment_keys(model_cfg) -> None:
    """Raise if ``model_cfg`` still names a removed experiment knob."""
    for section, key in REMOVED_EXPERIMENT_MODEL_KEYS:
        block = model_cfg if section is None else _get(model_cfg, section)
        if block is not None and _get(block, key) is not None:
            name = key if section is None else f"{section}.{key}"
            raise RetiredConfigKey(f"model.{name} {REMOVED_EXPERIMENT_EXPLANATION}")


def reject_retired_run_keys(run_cfg) -> None:
    """Raise if ``run_cfg`` still carries a retired binary-label key."""
    for key in RETIRED_RUN_KEYS:
        if _get(run_cfg, key) is not None:
            raise RetiredConfigKey(f"run.{key} {EXPLANATION}")


def drop_retired_state(state_dict):
    """Return ``state_dict`` without entries of the retired heads/losses."""
    kept = type(state_dict)(
        (name, value)
        for name, value in state_dict.items()
        if not name.startswith(RETIRED_STATE_PREFIXES)
    )
    metadata = getattr(state_dict, "_metadata", None)
    if metadata is not None:
        kept._metadata = metadata
    return kept
