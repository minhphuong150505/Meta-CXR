"""Swin Transformer encoder wrapper used as an optional META-CXR branch."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SwinEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "ChayanM/SwinV2-GPT2_Mimic",
        pretrained: bool = True,
        frozen: bool = True,
        backend: str = "hf",
        normalize: bool | None = None,
        weights_path: str | None = None,
    ) -> None:
        super().__init__()

        self.backend = backend.lower()
        self.model_name = model_name
        self.frozen = frozen
        # medclip: the dataset already emits MedCLIP-normalised 224x224 input
        # (medclip_swin.medclip_preprocess), so nothing is re-normalised here.
        self.normalize = (
            (self.backend in {"hf", "huggingface", "transformers"})
            if normalize is None
            else normalize
        )
        if self.backend == "medclip" and self.normalize:
            raise ValueError(
                "backend 'medclip' takes input already normalised by "
                "medclip_preprocess; set model.swin.normalize: false"
            )
        #: Tokens per image, or None when the layout is not fixed.
        self.num_tokens = None
        #: True when position 0 is a pooled/global vector.
        self.has_global_token = False
        self.load_report = None

        if self.backend == "timm":
            self._init_timm(model_name=model_name, pretrained=pretrained)
        elif self.backend in {"hf", "huggingface", "transformers"}:
            self._init_huggingface(model_name=model_name, pretrained=pretrained)
        elif self.backend == "medclip":
            self._init_medclip(model_name=model_name, weights_path=weights_path)
        else:
            raise ValueError(
                f"Unsupported Swin backend '{backend}'. Use 'medclip', 'timm' or 'hf'."
            )

        if self.normalize:
            self.register_buffer(
                "image_mean",
                torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
                persistent=False,
            )
            self.register_buffer(
                "image_std",
                torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
                persistent=False,
            )

        if frozen:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

    def _init_timm(self, model_name: str, pretrained: bool) -> None:
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "timm is required for SwinEncoder. Install with `pip install timm>=0.9.0`."
            ) from exc

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

    def _init_medclip(self, model_name: str, weights_path: str | None) -> None:
        """MedCLIP Swin-Tiny, the encoder the META-CXR paper uses.

        Emits [B, 50, 768]: ``pooler_output`` at position 0, then the 7x7
        ``last_hidden_state`` patches. The 768 -> 512 ``projection_head`` of
        ``MedCLIPVisionModelViT`` is deliberately not used.
        """
        from vision_encoders.swin.medclip_swin import (
            MEDCLIP_EMBED_DIM,
            MEDCLIP_IMG_SIZE,
            MEDCLIP_NUM_TOKENS,
            MEDCLIP_VIT_TYPE,
            build_medclip_swin,
        )

        if model_name != MEDCLIP_VIT_TYPE:
            raise ValueError(
                f"backend 'medclip' is MedCLIP's {MEDCLIP_VIT_TYPE}; got model_name "
                f"{model_name!r}"
            )
        self.model, self.load_report = build_medclip_swin(weights_path, backbone=model_name)
        self.input_size = (MEDCLIP_IMG_SIZE, MEDCLIP_IMG_SIZE)
        self.embed_dim = MEDCLIP_EMBED_DIM
        self.num_tokens = MEDCLIP_NUM_TOKENS
        self.has_global_token = True
        import logging

        logging.info("MedCLIP Swin loaded: %s", self.load_report)

    def _init_huggingface(self, model_name: str, pretrained: bool) -> None:
        try:
            from transformers import AutoConfig, AutoModel, AutoModelForImageClassification
        except ImportError as exc:
            raise ImportError(
                "transformers is required for Hugging Face SwinEncoder checkpoints."
            ) from exc

        hf_config = AutoConfig.from_pretrained(model_name)

        if getattr(hf_config, "model_type", None) == "vision-encoder-decoder":
            # MIMIC-finetuned SwinV2 is published inside a VisionEncoderDecoder
            # checkpoint; keep only the vision encoder and drop the text decoder.
            from transformers import VisionEncoderDecoderModel

            ved = VisionEncoderDecoderModel.from_pretrained(model_name)
            self.model = ved.encoder
            config = self.model.config
        elif pretrained:
            # Use the classification wrapper first so supervised fine-tuned
            # checkpoints load completely, then keep only the Swin backbone.
            hf_model = AutoModelForImageClassification.from_pretrained(model_name)
            self.model = getattr(
                hf_model,
                "swin",
                getattr(hf_model, "swinv2", getattr(hf_model, "base_model", hf_model)),
            )
            config = getattr(self.model, "config", getattr(hf_model, "config", None))
        else:
            config = hf_config
            self.model = AutoModel.from_config(config)

        image_size = getattr(config, "image_size", 224)
        if isinstance(image_size, int):
            self.input_size = (image_size, image_size)
        else:
            self.input_size = tuple(image_size)
        self.embed_dim = int(getattr(config, "hidden_size", 768))

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            self.model.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backend == "medclip":
            from vision_encoders.swin.medclip_swin import join_pooled_and_patches

            if tuple(x.shape[-2:]) != self.input_size or x.shape[1] != 3:
                raise ValueError(
                    f"backend 'medclip' expects [B, 3, 224, 224] from "
                    f"medclip_preprocess, got {tuple(x.shape)}; it must not be fed "
                    "the BioViL tensor"
                )
            output = self.model(pixel_values=x)
            return join_pooled_and_patches(output.pooler_output, output.last_hidden_state)

        if tuple(x.shape[-2:]) != self.input_size:
            x = F.interpolate(
                x,
                size=self.input_size,
                mode="bilinear",
                align_corners=False,
            )

        if self.normalize:
            x = (x - self.image_mean) / self.image_std

        if self.backend == "timm":
            feats = self.model.forward_features(x)
            if feats.dim() == 4:
                b, h, w, c = feats.shape
                feats = feats.reshape(b, h * w, c)
            elif feats.dim() != 3:
                raise RuntimeError(
                    f"Unexpected Swin forward_features output shape {tuple(feats.shape)}"
                )
        else:
            output = self.model(pixel_values=x)
            feats = output.last_hidden_state
        return feats
