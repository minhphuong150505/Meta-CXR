"""Post-hoc explainability for the Stage-2 language head.

This package is an OBSERVER. Nothing in the Stage-1 or Stage-2 training path
imports it, and deleting the directory leaves both pipelines byte-for-byte
unchanged. That independence is the point: an explanation layer that can alter
what it explains is not an explanation layer.

Module split, and why it is this split:

``rollout``
    Pure tensor algebra. No model, no ``transformers``, no import from anywhere
    else in this repository. Attention matrices in, attribution vector out, so
    the maths is checkable against hand-computed answers on a CPU box.
``projection``
    Feature-grid coordinates to image coordinates. Also pure tensor.
``sentence_attribution``
    Sentence splitting, per-sentence labels and per-sentence NLL. Standard
    library plus the repository's existing ``safety.claims`` lexicon.
``attention_capture``
    The only module that touches a live model. Everything above it is testable
    without a GPU precisely so that this one stays small.

Nothing here is imported eagerly, so ``import training.explainability`` costs
nothing and never drags in the training stack.
"""

from __future__ import annotations

__all__ = [
    "attention_capture",
    "projection",
    "rollout",
    "sentence_attribution",
]
