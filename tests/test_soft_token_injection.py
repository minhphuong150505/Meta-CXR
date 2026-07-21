"""Soft-token injection must fail closed rather than mix up studies.

These run on real tensors on CPU -- no transformers, no MedGemma weights, no
GPU. The failure this guards against is silent: with a clamped index the model
trains fine and simply describes each study using another study's image.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "training"))

from training.medgemma.soft_tokens import SoftTokenEmbeddingWrapper  # noqa: E402

VOCAB, HIDDEN, IMG_TOKEN_ID, N_TOKENS = 50, 8, 7, 4


def make_embedding() -> nn.Embedding:
    torch.manual_seed(0)
    return nn.Embedding(VOCAB, HIDDEN)


def make_input_ids(batch: int, n_tokens: int = N_TOKENS) -> torch.Tensor:
    """One row per sample: [text, <img> * n, text]."""
    rows = []
    for b in range(batch):
        rows.append([1 + b] + [IMG_TOKEN_ID] * n_tokens + [2 + b])
    return torch.tensor(rows, dtype=torch.long)


def wrap(embedding, projected, n_tokens: int = N_TOKENS) -> SoftTokenEmbeddingWrapper:
    return SoftTokenEmbeddingWrapper(embedding, IMG_TOKEN_ID, projected, n_tokens)


def test_single_sample_substitutes_projected_embeddings():
    embedding = make_embedding()
    projected = torch.randn(1, N_TOKENS, HIDDEN)
    out = wrap(embedding, projected)(make_input_ids(1))

    assert out.shape == (1, N_TOKENS + 2, HIDDEN)
    torch.testing.assert_close(out[0, 1 : 1 + N_TOKENS, :], projected[0])


def test_each_sample_gets_its_own_embeddings():
    """The regression test for the removed ``min(batch_idx, n - 1)`` clamp."""
    embedding = make_embedding()
    projected = torch.stack(
        [torch.full((N_TOKENS, HIDDEN), 1.0), torch.full((N_TOKENS, HIDDEN), -1.0)]
    )
    out = wrap(embedding, projected)(make_input_ids(2))

    torch.testing.assert_close(out[0, 1 : 1 + N_TOKENS, :], projected[0])
    torch.testing.assert_close(out[1, 1 : 1 + N_TOKENS, :], projected[1])
    # Explicitly: row 1 must not have received row 0's features.
    assert not torch.allclose(out[1, 1 : 1 + N_TOKENS, :], projected[0])


def test_non_image_positions_are_untouched():
    embedding = make_embedding()
    input_ids = make_input_ids(2)
    reference = embedding(input_ids)
    out = wrap(embedding, torch.randn(2, N_TOKENS, HIDDEN))(input_ids)

    torch.testing.assert_close(out[:, 0, :], reference[:, 0, :])
    torch.testing.assert_close(out[:, -1, :], reference[:, -1, :])


def test_batch_size_mismatch_raises():
    embedding = make_embedding()
    projected = torch.randn(3, N_TOKENS, HIDDEN)  # 3 embeddings, 2 sequences
    with pytest.raises(ValueError, match="do not match the batch"):
        wrap(embedding, projected)(make_input_ids(2))


def test_too_few_placeholders_raises():
    embedding = make_embedding()
    projected = torch.randn(1, N_TOKENS, HIDDEN)
    input_ids = make_input_ids(1, n_tokens=N_TOKENS - 1)
    with pytest.raises(ValueError, match=f"expected {N_TOKENS} image tokens"):
        wrap(embedding, projected)(input_ids)


def test_too_many_placeholders_raises():
    embedding = make_embedding()
    projected = torch.randn(1, N_TOKENS, HIDDEN)
    input_ids = make_input_ids(1, n_tokens=N_TOKENS + 1)
    with pytest.raises(ValueError, match=f"expected {N_TOKENS} image tokens"):
        wrap(embedding, projected)(input_ids)


def test_one_row_missing_placeholders_raises():
    """A batch where only some rows carry image tokens must not pass silently."""
    embedding = make_embedding()
    projected = torch.randn(2, N_TOKENS, HIDDEN)
    input_ids = torch.tensor(
        [
            [1] + [IMG_TOKEN_ID] * N_TOKENS + [2],
            [1] + [3] * N_TOKENS + [2],  # no image tokens at all
        ],
        dtype=torch.long,
    )
    with pytest.raises(ValueError, match="expected"):
        wrap(embedding, projected)(input_ids)


def test_hidden_size_mismatch_raises():
    embedding = make_embedding()
    projected = torch.randn(1, N_TOKENS, HIDDEN + 3)
    with pytest.raises(ValueError, match="width does not match"):
        wrap(embedding, projected)(make_input_ids(1))


def test_wrong_rank_raises():
    embedding = make_embedding()
    projected = torch.randn(N_TOKENS, HIDDEN)  # missing batch axis
    with pytest.raises(ValueError, match=r"must be \[B, N, D\]"):
        wrap(embedding, projected)(make_input_ids(1))


def test_no_image_tokens_is_a_passthrough():
    """A text-only batch must return the base embeddings unchanged."""
    embedding = make_embedding()
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    projected = torch.randn(1, N_TOKENS, HIDDEN)
    torch.testing.assert_close(wrap(embedding, projected)(input_ids), embedding(input_ids))


def test_gradient_flows_to_the_projector():
    """The projector must receive gradient from the LM loss.

    If substitution detached the graph, the projector would silently never
    train and the soft tokens would stay at their random initialisation.
    """
    embedding = make_embedding()
    projector = nn.Linear(6, HIDDEN)
    qformer = torch.randn(2, N_TOKENS, 6)
    projected = projector(qformer)

    out = wrap(embedding, projected)(make_input_ids(2))
    out.sum().backward()

    assert projector.weight.grad is not None
    assert torch.any(projector.weight.grad != 0)


def test_gradient_also_reaches_the_base_embedding():
    """Substitution must not sever the un-substituted text positions."""
    embedding = make_embedding()
    projected = torch.randn(1, N_TOKENS, HIDDEN, requires_grad=True)
    wrap(embedding, projected)(make_input_ids(1)).sum().backward()

    assert embedding.weight.grad is not None
    assert torch.any(embedding.weight.grad != 0)


def test_projected_dtype_is_cast_to_the_embedding_dtype():
    """Mixed precision: an fp32 projector output feeds a half-precision LM."""
    embedding = make_embedding().to(torch.float16)
    projected = torch.randn(1, N_TOKENS, HIDDEN, dtype=torch.float32)
    out = wrap(embedding, projected)(make_input_ids(1))

    assert out.dtype == torch.float16
    torch.testing.assert_close(
        out[0, 1 : 1 + N_TOKENS, :], projected[0].to(torch.float16)
    )


def test_weight_property_exposes_the_base_table():
    """HF code reads ``.weight`` off the embedding module; the wrapper must proxy it."""
    embedding = make_embedding()
    wrapper = wrap(embedding, torch.randn(1, N_TOKENS, HIDDEN))
    assert wrapper.weight is embedding.weight
