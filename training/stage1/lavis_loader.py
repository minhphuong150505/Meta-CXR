"""Every LAVIS/Stage-1 import in the Stage-2 pipeline, isolated to one module.

Importing this module pulls in `model.lavis`, which transitively loads the
vision encoders, MHCAC and the Q-Former. That is the entire point of the file:
by keeping those imports here instead of at the top of the Figure-9 module, a
``medgemma_direct`` run can import the Stage-2 code without the Stage-1 stack
being importable at all.

Nothing here may be imported at module scope by a Stage-2 entrypoint. Import it
inside the branch that has already decided it needs Stage-1.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

PROJECT_DIR = Path(__file__).resolve().parents[2]
for _path in (PROJECT_DIR, PROJECT_DIR / "model"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import model.lavis.tasks as tasks  # noqa: E402
from local_config import VIS_ROOT  # noqa: E402
from model.lavis.common.config import Config  # noqa: E402
from model.lavis.common.registry import registry  # noqa: E402
from model.lavis.data.ReportDataset import MIMIC_CXR_Dataset  # noqa: E402
from training.torch_io import load_torch_checkpoint  # noqa: E402

registry.mapping["paths"]["cache_root"] = "."

__all__ = [
    "Config",
    "PROJECT_DIR",
    "build_cfg",
    "build_stage1_model",
    "default_stage1_config_path",
    "filter_state_dict_for_model",
    "load_state_dict_materializing_meta",
    "load_torch_checkpoint",
    "make_stage1_loader",
]


def default_stage1_config_path(run_name: str) -> Path:
    return PROJECT_DIR / "pretraining/configs/encoder_comparison" / f"{run_name}.yaml"


def build_cfg(context) -> Config:
    cfg_path = context.resolve_config_path(default_stage1_config_path(context.run_name))
    return Config(SimpleNamespace(cfg_path=str(cfg_path), options=None))


def filter_state_dict_for_model(model, state_dict: dict) -> dict:
    model_state = model.state_dict()
    filtered = {}
    for key, value in state_dict.items():
        if (
            key in model_state
            and hasattr(value, "shape")
            and tuple(value.shape) != tuple(model_state[key].shape)
        ):
            continue
        filtered[key] = value
    return filtered


def load_state_dict_materializing_meta(model, state_dict: dict):
    try:
        return model.load_state_dict(state_dict, strict=False, assign=True)
    except TypeError:
        return model.load_state_dict(state_dict, strict=False)


def build_stage1_model(context, checkpoint_root: Path, device: torch.device):
    cfg = build_cfg(context)
    task = tasks.setup_task(cfg)
    model = task.build_model(cfg)
    ckpt_path = context.resolve_checkpoint_path(checkpoint_root)
    ckpt = load_torch_checkpoint(ckpt_path)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    state_dict = filter_state_dict_for_model(model, state_dict)
    missing, unexpected = load_state_dict_materializing_meta(model, state_dict)
    print(f"[stage1] loaded {ckpt_path}; missing={len(missing)} unexpected={len(unexpected)}")
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return cfg, model


def make_stage1_loader(cfg, split: str, sample_limit: int | None, num_workers: int) -> DataLoader:
    dataset = MIMIC_CXR_Dataset(
        vis_processor=None,
        text_processor=None,
        vis_root=VIS_ROOT,
        split=split,
        cfg=cfg,
        truncate=sample_limit if sample_limit and sample_limit > 0 else None,
    )
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=dataset.collater,
    )
