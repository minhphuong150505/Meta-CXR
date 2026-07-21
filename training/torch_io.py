"""Checkpoint I/O shared by the Stage-1 and Stage-2 code paths.

Deliberately depends on nothing but torch, so that both
``training/stage1/lavis_loader.py`` (which pulls the whole Stage-1 stack) and
the Stage-2 modules (which must not) can use it without either importing the
other.
"""

from __future__ import annotations

from pathlib import Path

import torch


def load_torch_checkpoint(path: str | Path):
    """``torch.load`` onto CPU, tolerating torch versions without ``weights_only``.

    These checkpoints are produced by this repo's own training runs, so full
    unpickling is intended; the fallback exists only for older torch builds that
    do not accept the keyword.
    """
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")
