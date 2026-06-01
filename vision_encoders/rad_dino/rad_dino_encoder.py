"""Rad-DINO vision encoder (DINOv2 ViT-B pretrained on chest X-rays).

Optional META-CXR branch that feeds the Q-Former stream with strong
domain-specific CXR features. Mirrors the SwinEncoder interface
(``embed_dim`` / ``input_size`` attributes, ``forward`` returns patch tokens).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class RadDinoEncoder(nn.Module):
    # rad-dino preprocessor stats (grayscale CXR), from the model card's
    # preprocessor_config.json — NOT ImageNet stats.
    DEFAULT_MEAN = (0.5307, 0.5307, 0.5307)
    DEFAULT_STD = (0.2583, 0.2583, 0.2583)

    def __init__(
        self,
        model_name: str = "microsoft/rad-dino",
        pretrained: bool = True,
        frozen: bool = True,
        normalize: bool = True,
    ) -> None:
        super().__init__()

        try:
            from transformers import AutoConfig, AutoModel
        except ImportError as exc:
            raise ImportError(
                "transformers is required for RadDinoEncoder. "
                "Install with `pip install transformers>=4.38`."
            ) from exc

        self.model_name = model_name
        self.frozen = frozen
        self.normalize = normalize

        if pretrained:
            self.model = AutoModel.from_pretrained(model_name)
            config = self.model.config
        else:
            config = AutoConfig.from_pretrained(model_name)
            self.model = AutoModel.from_config(config)

        image_size = getattr(config, "image_size", 518)
        if isinstance(image_size, int):
            self.input_size = (image_size, image_size)
        else:
            self.input_size = tuple(image_size)
        self.embed_dim = int(getattr(config, "hidden_size", 768))

        if self.normalize:
            self.register_buffer(
                "image_mean",
                torch.tensor(self.DEFAULT_MEAN).view(1, 3, 1, 1),
                persistent=False,
            )
            self.register_buffer(
                "image_std",
                torch.tensor(self.DEFAULT_STD).view(1, 3, 1, 1),
                persistent=False,
            )

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

        if self.normalize:
            x = (x - self.image_mean) / self.image_std

        # last_hidden_state: (B, 1 + num_patches, hidden_size); keep CLS + patches.
        return self.model(pixel_values=x).last_hidden_state
