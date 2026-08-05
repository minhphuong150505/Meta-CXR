"""Small torch device helper kept local to avoid the full hi-ml dependency."""

import torch


def get_module_device(module: torch.nn.Module) -> torch.device:
    """Return the device of a module's first parameter or buffer."""
    parameter = next(module.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(module.buffers(), None)
    if buffer is not None:
        return buffer.device
    return torch.device("cpu")
