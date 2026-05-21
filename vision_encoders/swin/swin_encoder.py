"""Swin Transformer encoder wrapper used as an optional META-CXR branch."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SwinEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "swin_tiny_patch4_window7_224",
        pretrained: bool = True,
        frozen: bool = True,
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "timm is required for SwinEncoder. Install with `pip install timm>=0.9.0`."
            ) from exc

        self.model_name = model_name
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )
        patch_embed = getattr(self.model, "patch_embed", None)
        input_size = getattr(patch_embed, "img_size", (224, 224))
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        self.input_size = tuple(input_size)
        self.embed_dim = int(getattr(self.model, "num_features", 768))
        self.frozen = frozen
        if frozen:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            self.model.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if tuple(x.shape[-2:]) != self.input_size:
            x = F.interpolate(
                x,
                size=self.input_size,
                mode="bilinear",
                align_corners=False,
            )

        feats = self.model.forward_features(x)
        if feats.dim() == 4:
            b, h, w, c = feats.shape
            feats = feats.reshape(b, h * w, c)
        elif feats.dim() != 3:
            raise RuntimeError(
                f"Unexpected Swin forward_features output shape {tuple(feats.shape)}"
            )
        return feats
