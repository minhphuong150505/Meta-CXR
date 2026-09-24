"""GradCache, Q-Former gradient checkpointing and SigLIP (D-021, 2026-09-25).

The claims pinned here are the ones that would fail SILENTLY:

* GradCache must give the gradient of the full-batch loss: chunking may change
  memory, never the update;
* with checkpointing the query KV cache is gone, so the LM term re-feeds the
  queries and the image -- it must equal the cached computation, or the LM
  trains without the image;
* checkpointing must not change gradients;
* SigLIP must be the published loss, over valid pairs only.

The model-level tests build a tiny real Q-Former (LAVIS fork) and need
transformers plus the bert-base-uncased tokenizer: they run on the training
host and skip on the CPU box.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from mhcac.loss import siglip_loss

# --------------------------------------------------------------------------
# SigLIP (CPU)
# --------------------------------------------------------------------------


def test_siglip_matches_the_published_formula():
    torch.manual_seed(0)
    sim = torch.randn(5, 5)
    t, b = torch.tensor(math.log(10.0)), torch.tensor(-10.0)
    ours = siglip_loss(sim, t, b)
    z = 2 * torch.eye(5) - 1
    ref = -F.logsigmoid(z * (sim * 10.0 - 10.0)).sum() / 5
    assert float(ours) == pytest.approx(float(ref), rel=1e-6)


def test_siglip_scores_valid_studies_only():
    torch.manual_seed(0)
    sim = torch.randn(4, 4)
    valid = torch.tensor([True, False, True, True])
    t, b = torch.tensor(1.0), torch.tensor(0.0)
    keep = valid.nonzero().flatten()
    expected = siglip_loss(sim[keep][:, keep], t, b)
    assert float(siglip_loss(sim, t, b, valid)) == pytest.approx(float(expected))
    assert float(siglip_loss(sim, t, b, torch.zeros(4, dtype=torch.bool))) == 0.0


def test_siglip_rewards_alignment_and_trains_its_scale_and_bias():
    t = torch.tensor(math.log(10.0), requires_grad=True)
    b = torch.tensor(-10.0, requires_grad=True)
    aligned = siglip_loss(torch.eye(8), t, b)
    shuffled = siglip_loss(torch.eye(8).roll(1, dims=1), t, b)
    assert float(aligned) < float(shuffled)
    aligned.backward()
    assert t.grad is not None and b.grad is not None


def test_shipped_config_uses_siglip_checkpointing_and_gradcache():
    from pathlib import Path

    import yaml

    cfg = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "pretraining/configs/mimic_cxr_full.yaml").read_text()
    )
    assert cfg["model"]["loss"]["itc_loss"] == "sigmoid"
    assert cfg["model"]["qformer_grad_checkpointing"] is True
    assert cfg["model"]["gradcache_chunk_size"] == 0, "only phase 1a caches"
    phase1a = cfg["run"]["phases"]["phase1a"]
    assert phase1a["model"]["gradcache_chunk_size"] > 0
    assert phase1a["run"]["batch_size_train"] >= 32
    assert phase1a["run"]["batch_size_train"] % phase1a["model"]["gradcache_chunk_size"] == 0
    for name in ("phase1a", "phase1b", "phase1c"):
        assert "siglip_bias" in cfg["run"]["phases"][name]["trainable"]


# --------------------------------------------------------------------------
# Tiny real Q-Former (host only)
# --------------------------------------------------------------------------

VIS_DIM = 16
HIDDEN = 32
QUERIES = 4


def _tiny_model(itc_loss="sigmoid", checkpointing=False, chunk=0):
    pytest.importorskip("transformers")
    try:
        from model.lavis.models.blip2_models.blip2_qformer import Blip2Qformer
        from model.lavis.models.blip2_models.Qformer import BertConfig, BertLMHeadModel
        from vision_encoders.shared_visual_tokens import SharedVisualTokens
    except Exception as exc:
        pytest.skip(f"Q-Former stack not importable here ({type(exc).__name__})")
    try:
        tokenizer = Blip2Qformer.init_tokenizer()
    except Exception as exc:  # no network and no cached tokenizer
        pytest.skip(f"bert-base-uncased tokenizer unavailable ({type(exc).__name__})")

    torch.manual_seed(0)
    config = BertConfig(
        vocab_size=len(tokenizer), hidden_size=HIDDEN, num_hidden_layers=2,
        num_attention_heads=2, intermediate_size=64, hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0, max_position_embeddings=64,
    )
    config.encoder_width = VIS_DIM
    config.add_cross_attention = True
    config.cross_attention_freq = 1
    config.query_length = QUERIES
    config.gradient_checkpointing = checkpointing

    model = Blip2Qformer.__new__(Blip2Qformer)
    nn.Module.__init__(model)
    model.tokenizer = tokenizer
    model.Qformer = BertLMHeadModel(config)
    model.query_tokens = nn.Parameter(torch.randn(1, QUERIES, HIDDEN) * 0.02)
    model.vision_proj = nn.Linear(HIDDEN, 8)
    model.text_proj = nn.Linear(HIDDEN, 8)
    model.itm_head = nn.Linear(HIDDEN, 2)
    model.temp = nn.Parameter(torch.tensor(0.07))
    model.register_buffer("itc_temp_fixed", torch.tensor(0.07), persistent=False)
    model.itc_temp_learnable = True
    model.itc_loss = itc_loss
    model.siglip_logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
    model.siglip_bias = nn.Parameter(torch.tensor(-10.0))
    model.itc_label_smoothing = 0.1
    model.itc_queue_size = 0
    model.register_buffer("itc_queue_filled", torch.zeros((), dtype=torch.long), persistent=False)
    model.max_txt_len = 12
    model.gradcache_chunk_size = chunk
    for name in ("lambda_cls", "lambda_teacher_cls", "lambda_distill", "lambda_mhcac_contrastive",
                 "lambda_orthogonality", "lambda_sparsity", "lambda_gate",
                 "lambda_mention_conditioned_cls", "lambda_view_consistency",
                 "lambda_explanation", "lambda_explanation_strong"):
        setattr(model, name, 0.0)
    model.lambda_itc = model.lambda_itm = model.lambda_lm = 1.0
    model.projector = nn.Linear(VIS_DIM, VIS_DIM)  # stands in for the shared projection

    def encode_samples(samples):
        tokens = model.projector(samples["feat"])
        return SharedVisualTokens(tokens=tokens, spans={"biovil": slice(0, tokens.shape[1])})

    model.encode_samples = encode_samples
    return model


def _batch(n=6, seed=1):
    g = torch.Generator().manual_seed(seed)
    words = ["normal heart", "left effusion", "no pneumothorax", "clear lungs",
             "mild edema", "right opacity", "stable tube", "small nodule"]
    return {
        "feat": torch.randn(n, 5, VIS_DIM, generator=g),
        "text_output": [words[i % len(words)] for i in range(n)],
        "generation_mask": torch.tensor([True] * (n - 1) + [False]),
    }


def _grads(model):
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


@pytest.mark.parametrize("itc_loss", ["sigmoid", "softmax"])
def test_gradcache_chunks_give_the_full_batch_gradient(itc_loss):
    batch = _batch()
    reference = _tiny_model(itc_loss, chunk=6).train()
    torch.manual_seed(5)
    out_full = reference.forward_gradcache(batch)
    g_full = _grads(reference)

    chunked = _tiny_model(itc_loss, chunk=2).train()
    chunked.load_state_dict(reference.state_dict())
    torch.manual_seed(5)
    out_chunked = chunked.forward_gradcache(batch)
    g_chunked = _grads(chunked)

    for key in ("loss_itc", "loss_itm", "loss_lm", "loss"):
        assert float(out_chunked[key]) == pytest.approx(float(out_full[key]), rel=1e-4, abs=1e-6)
    assert set(g_full) == set(g_chunked)
    assert "projector.weight" in g_full, "the shared projection must receive gradient"
    for name in g_full:
        # Relative to the tensor's own scale: chunking reorders float sums, so
        # entries that are ~0 in exact arithmetic may differ by rounding.
        diff = (g_chunked[name] - g_full[name]).abs().max().item()
        scale = g_full[name].abs().max().item()
        assert diff <= 1e-6 + 1e-4 * scale, f"{name}: max |diff| {diff:.3g} at scale {scale:.3g}"


def test_gradcache_matches_an_ordinary_backward_on_the_same_loss():
    """With one chunk, the surrogate trick must equal autograd on the full loss."""
    batch = _batch()
    model = _tiny_model("sigmoid", chunk=6).train()
    torch.manual_seed(5)
    model.forward_gradcache(batch)
    g_cache = _grads(model)

    plain = _tiny_model("sigmoid", chunk=6).train()
    plain.load_state_dict(model.state_dict())
    torch.manual_seed(5)
    from types import SimpleNamespace as NS

    tokens = plain.tokenizer(batch["text_output"], padding="max_length", truncation=True,
                             max_length=plain.max_txt_len, return_tensors="pt")
    tok = NS(input_ids=tokens.input_ids, attention_mask=tokens.attention_mask)
    mask = batch["generation_mask"]
    img_f, txt_f, embeds, q_tokens, q_out = plain._alignment_features(batch, tok)
    loss_itc, sim_i2t, sim_t2i, valid_all = plain._image_text_contrastive(img_f, txt_f, mask)
    loss_lm = plain._language_modeling(tok, q_tokens, q_out, mask, image_embeds=embeds)
    (loss_itc + loss_lm).backward()
    g_plain = _grads(plain)
    # ITM draws random negatives, so compare on the ITM-free parameters' ITC+LM
    # part by re-running GradCache with ITM off.
    model2 = _tiny_model("sigmoid", chunk=6).train()
    model2.load_state_dict(model.state_dict())
    model2.lambda_itm = 0.0
    torch.manual_seed(5)
    model2.forward_gradcache(batch)
    g2 = _grads(model2)
    for name, grad in g_plain.items():
        if name.startswith("itm_head"):
            continue
        diff = (g2[name] - grad).abs().max().item()
        scale = grad.abs().max().item()
        assert diff <= 1e-6 + 1e-3 * scale, f"{name}: max |diff| {diff:.3g} at scale {scale:.3g}"
    assert g_cache  # the ITM-on run produced gradients too


def test_lm_without_the_kv_cache_equals_the_cached_lm():
    model = _tiny_model().eval()
    batch = _batch()
    tokens = model.tokenizer(batch["text_output"], padding="max_length", truncation=True,
                             max_length=model.max_txt_len, return_tensors="pt")
    tok = SimpleNamespace(input_ids=tokens.input_ids, attention_mask=tokens.attention_mask)
    with torch.no_grad():
        _, _, embeds, q_tokens, q_out = model._alignment_features(batch, tok)
        assert q_out.past_key_values is not None
        cached = model._language_modeling(tok, q_tokens, q_out, batch["generation_mask"])
        no_cache = SimpleNamespace(last_hidden_state=q_out.last_hidden_state, past_key_values=None)
        recomputed = model._language_modeling(
            tok, q_tokens, no_cache, batch["generation_mask"], image_embeds=embeds
        )
        with pytest.raises(ValueError, match="image_embeds"):
            model._language_modeling(tok, q_tokens, no_cache, batch["generation_mask"])
    assert float(recomputed) == pytest.approx(float(cached), rel=1e-5)


def test_gradient_checkpointing_does_not_change_gradients():
    batch = _batch()
    plain = _tiny_model(checkpointing=False, chunk=6).train()
    plain.lambda_itm = 0.0
    torch.manual_seed(3)
    plain.forward_gradcache(batch)
    g_plain = _grads(plain)

    ckpt = _tiny_model(checkpointing=True, chunk=6).train()
    ckpt.load_state_dict(plain.state_dict())
    ckpt.lambda_itm = 0.0
    torch.manual_seed(3)
    ckpt.forward_gradcache(batch)
    g_ckpt = _grads(ckpt)
    assert set(g_plain) == set(g_ckpt)
    for name in g_plain:
        assert torch.allclose(g_ckpt[name], g_plain[name], rtol=1e-4, atol=1e-7), name


def test_gradcache_refuses_classification_objectives():
    model = _tiny_model(chunk=2)
    model.lambda_cls = 1.0
    with pytest.raises(ValueError, match="alignment-only"):
        model.forward_gradcache(_batch())
