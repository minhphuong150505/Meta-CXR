"""Soft-token injection: substituting projected image features into the LM.

Visual conditioning works by *replacing* the embedding at each
``<qformer_soft_token>`` position with a projected Q-Former vector -- not by
summing into it. Getting the per-sample indexing wrong here is silent: the model
still trains, the loss still falls, and every study is simply described using a
different study's image. So every shape relationship is checked per row and the
module fails closed.

Depends on nothing but torch, which keeps it unit-testable on CPU without
transformers, MedGemma weights or a GPU.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from stage2_utils import validate_soft_token_batch
except ImportError:  # ``python -m training...``
    from training.stage2_utils import validate_soft_token_batch

DEFAULT_NUM_IMG_TOKENS = 32

# The projector itself stays an ordinary ``nn.Linear(768, hidden)`` owned by the
# reporter. Wrapping it in a named module here would rename its state-dict keys
# and silently invalidate every ``img_proj.pt`` already written to disk.


class SoftTokenEmbeddingWrapper(nn.Module):
    """Wraps an LM embedding table, substituting image features at image tokens.

    Installed around the real embedding for the duration of one forward pass and
    then removed, so the underlying table is never mutated.
    """

    def __init__(
        self,
        base_embedding: nn.Module,
        img_token_id: int,
        projected_img_embs: torch.Tensor,
        num_img_tokens: int = DEFAULT_NUM_IMG_TOKENS,
    ):
        super().__init__()
        self.base_embedding = base_embedding
        self.img_token_id = int(img_token_id)
        self.projected_img_embs = projected_img_embs
        self.num_img_tokens = int(num_img_tokens)

    @property
    def weight(self):
        return getattr(self.base_embedding, "weight", None)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeds = self.base_embedding(input_ids)
        mask = input_ids == self.img_token_id
        if not mask.any():
            return embeds

        if self.projected_img_embs.dim() != 3:
            raise ValueError(
                "projected image embeddings must be [B, N, D], got "
                f"{tuple(self.projected_img_embs.shape)}"
            )
        hidden = embeds.shape[-1]
        if self.projected_img_embs.shape[-1] != hidden:
            raise ValueError(
                "projected image embedding width does not match the language model: "
                f"{self.projected_img_embs.shape[-1]} vs {hidden}"
            )

        # Cloning keeps autograd intact for the un-substituted positions while
        # allowing in-place assignment at the image positions.
        embeds = embeds.clone()
        for batch_idx in range(input_ids.shape[0]):
            positions = mask[batch_idx].nonzero(as_tuple=False).flatten()
            # Fails closed on any batch/embedding mismatch. The previous
            # ``min(batch_idx, n - 1)`` clamp silently fed one study's image
            # features to a different study's report.
            validate_soft_token_batch(
                self.projected_img_embs.shape[0],
                input_ids.shape[0],
                len(positions),
                self.num_img_tokens,
            )
            img = self.projected_img_embs[batch_idx]
            embeds[batch_idx, positions, :] = img.to(device=embeds.device, dtype=embeds.dtype)
        return embeds
