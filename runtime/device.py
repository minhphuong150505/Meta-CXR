"""Device and dtype selection for external-checkpoint inference.

Nothing here hardcodes ``cuda:0``. The device is resolved from config or from
what the machine actually reports, so the same config runs on any single-GPU
workstation without edits.
"""

from __future__ import annotations

from dataclasses import dataclass


class QuantizationUnavailable(RuntimeError):
    """4-bit was requested but bitsandbytes or CUDA cannot provide it."""


@dataclass(frozen=True)
class DevicePlan:
    """Resolved placement for one model load."""

    device: str
    torch_dtype: str
    load_in_4bit: bool

    @property
    def is_cuda(self) -> bool:
        return self.device.startswith("cuda")


def _torch():
    """Import torch lazily so config parsing works on a machine without it."""
    import torch

    return torch


def resolve_device(requested: str = "auto") -> str:
    """Return a concrete device string.

    ``auto`` picks CUDA when it is genuinely usable and falls back to CPU
    otherwise. An explicitly requested CUDA device is *not* silently downgraded
    to CPU: a 4B multimodal model on CPU is slow enough that a silent fallback
    would look like a hang, and any timing it produced would be a useless basis
    for a cost estimate.
    """
    torch = _torch()
    available = torch.cuda.is_available()
    if requested == "auto":
        return "cuda" if available else "cpu"
    if requested.startswith("cuda") and not available:
        raise RuntimeError(
            f"device {requested!r} was requested but torch reports no CUDA device. "
            "Refusing to fall back to CPU: MedGemma-4B on CPU would not produce a "
            "usable throughput measurement. Set device: cpu explicitly if that is "
            "really what you want."
        )
    return requested


def resolve_dtype(device: str, requested: str = "auto") -> str:
    """Pick bfloat16 where supported, else float16 on GPU, else float32.

    The checkpoints are published in bfloat16, so bf16 is the faithful choice
    wherever the hardware offers it (Ampere and newer, which includes the 3090).
    """
    if requested != "auto":
        return requested
    if not device.startswith("cuda"):
        # fp16 on CPU is emulated and slower than fp32.
        return "float32"
    torch = _torch()
    try:
        if torch.cuda.is_bf16_supported():
            return "bfloat16"
    except Exception:  # pragma: no cover - depends on driver specifics
        pass
    return "float16"


def check_4bit_available(device: str) -> None:
    """Raise unless 4-bit quantised loading can actually run here."""
    if not device.startswith("cuda"):
        raise QuantizationUnavailable(
            f"4-bit loading needs a CUDA device, got {device!r}."
        )
    try:
        import bitsandbytes  # noqa: F401
    except ImportError as exc:
        raise QuantizationUnavailable(
            "load_in_4bit was requested but bitsandbytes is not installed."
        ) from exc


def plan_device(
    *, device: str = "auto", dtype: str = "auto", load_in_4bit: bool = False
) -> DevicePlan:
    """Resolve the full placement plan, failing fast on impossible requests."""
    resolved_device = resolve_device(device)
    if load_in_4bit:
        check_4bit_available(resolved_device)
    return DevicePlan(
        device=resolved_device,
        torch_dtype=resolve_dtype(resolved_device, dtype),
        load_in_4bit=load_in_4bit,
    )
