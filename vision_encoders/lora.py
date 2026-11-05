"""Minimal LoRA for ``nn.Linear`` (Hu et al. 2021), torch only -- the Stage-1 venv has no ``peft``.

Plan: ``docs/handoff/PLAN-2026-11-04-residual-lateral.md`` (Step 1). ``LoRALinear`` keeps the
wrapped layer frozen and adds ``(dropout(x) @ A^T @ B^T) * alpha / r``. ``A`` is Kaiming-uniform,
``B`` is zero, so at step 0 the output equals the frozen layer's (pinned bit for bit by
``tests/test_lateral_lora.py``). Nothing in the training pipeline imports this module.

    names = apply_lora(swin.model, SWIN_STAGE34_QV, r=8, alpha=16, dropout=0.05)
    opt = AdamW(lora_parameters(swin.model), ...)
    set_lora_train(swin, True)   # SwinEncoder(frozen=True).train() keeps the backbone in eval
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator

import torch
from torch import nn

# MedCLIP-Swin (HF SwinModel inside SwinEncoder.model): query / value of stages 3 and 4.
SWIN_STAGE34_QV = r"^encoder\.layers\.[23]\.blocks\.\d+\.attention\.self\.(query|value)$"


class LoRALinear(nn.Module):
    """``base(x) + scaling * B(A(dropout(x)))`` with ``base`` frozen."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear wraps nn.Linear, got {type(base).__name__}")
        if r <= 0:
            raise ValueError("r must be positive")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r, self.alpha, self.scaling = int(r), float(alpha), float(alpha) / int(r)
        w = base.weight
        self.lora_A = nn.Parameter(torch.empty(self.r, base.in_features, device=w.device, dtype=w.dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.r, device=w.device, dtype=w.dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.merged = False

    @property
    def weight(self) -> torch.Tensor:  # some HF code reads .weight of the attention projections
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def delta_weight(self) -> torch.Tensor:
        return (self.lora_B @ self.lora_A) * self.scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.merged:
            return y
        return y + (self.dropout(x) @ self.lora_A.t().to(x.dtype) @ self.lora_B.t().to(x.dtype)) * self.scaling

    @torch.no_grad()
    def merge(self) -> None:
        if not self.merged:
            self.base.weight += self.delta_weight().to(self.base.weight.dtype)
            self.merged = True


def apply_lora(module: nn.Module, pattern: str, r: int, alpha: float, dropout: float = 0.0) -> list[str]:
    """Wrap every ``nn.Linear`` whose qualified name matches ``pattern`` (``re.search``).

    Freezes every existing parameter of ``module`` first, so only LoRA tensors train. Returns the
    wrapped names; raises if nothing matched (a typo must not silently train nothing).
    """
    for p in module.parameters():
        p.requires_grad = False
    rx = re.compile(pattern)
    targets = [n for n, m in module.named_modules() if isinstance(m, nn.Linear) and rx.search(n)]
    if not targets:
        raise ValueError(f"no nn.Linear matches {pattern!r}")
    for name in targets:
        parent_name, _, child = name.rpartition(".")
        parent = module.get_submodule(parent_name) if parent_name else module
        setattr(parent, child, LoRALinear(getattr(parent, child), r, alpha, dropout))
    return targets


def lora_modules(module: nn.Module) -> Iterator[tuple[str, LoRALinear]]:
    for n, m in module.named_modules():
        if isinstance(m, LoRALinear):
            yield n, m


def lora_parameters(module: nn.Module) -> list[nn.Parameter]:
    return [p for _, m in lora_modules(module) for p in (m.lora_A, m.lora_B)]


def set_lora_train(module: nn.Module, mode: bool = True) -> None:
    """Set only the LoRA wrappers' train flag (their dropout); the backbone keeps its own mode."""
    for _, m in lora_modules(module):
        m.train(mode)


def merge_lora(module: nn.Module) -> int:
    """Fold every ``B A`` into its base weight; returns how many layers were merged."""
    n = 0
    for _, m in lora_modules(module):
        m.merge()
        n += 1
    return n


def lora_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {f"{n}.{k}": getattr(m, k).detach().cpu().clone()
            for n, m in lora_modules(module) for k in ("lora_A", "lora_B")}


@torch.no_grad()
def load_lora_state_dict(module: nn.Module, state: dict[str, torch.Tensor]) -> int:
    """Inverse of ``lora_state_dict`` on a module with the same LoRA layout; returns tensors copied.

    Refuses a missing, extra or shape-mismatched key, so weights can never load partially.
    """
    targets = {f"{n}.{k}": getattr(m, k) for n, m in lora_modules(module) for k in ("lora_A", "lora_B")}
    if set(targets) != set(state):
        missing, extra = sorted(set(targets) - set(state)), sorted(set(state) - set(targets))
        raise KeyError(f"LoRA state does not match the module: missing {missing[:3]}, extra {extra[:3]}")
    for k, p in targets.items():
        if tuple(p.shape) != tuple(state[k].shape):
            raise ValueError(f"{k}: shape {tuple(state[k].shape)} != {tuple(p.shape)}")
        p.copy_(state[k].to(device=p.device, dtype=p.dtype))
    return len(targets)


def base_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    """Every tensor of ``module`` except the LoRA ones (for the 'backbone bit-identical' gate)."""
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()
            if not (k.endswith(".lora_A") or k.endswith(".lora_B"))}
