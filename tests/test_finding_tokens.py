"""Learnable finding tokens — shapes, ordering, masking, gradients, parity.

EXPERIMENTAL branch (``--finding-tokens``, default ``off``). Everything here is
arithmetic over tensors plus a torch-free prompt builder, so it runs on a CPU
box without MedGemma weights or a GPU.

The tests are grouped by the way this could silently produce a wrong result
rather than a crash: wrong label order (the token says Edema, the numbers are
Pneumothorax's), wrong row (study A's predictions on study B's report), an
un-normalised feature the projection learns to ignore, or a default path that
quietly changed and invalidated the arms it is being compared against.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

torch = pytest.importorskip("torch")

from stage2.prompts.ontology import MODELED_FINDINGS  # noqa: E402
from training.medgemma.finding_tokens import (  # noqa: E402
    FEATURE_ABLATION_SHUFFLE,
    FEATURE_ABLATION_ZERO,
    FINDING_TOKEN,
    FINDING_TOKENS_FULL,
    FINDING_TOKENS_OFF,
    FINDING_TOKENS_Q_ONLY,
    NUM_FINDING_TOKENS,
    FindingTokenEmbeddingWrapper,
    FindingTokenEncoder,
    apply_finding_feature_ablation,
    feature_width,
    finding_features,
)

N_LABELS = 14


def _logits(seed: int = 0) -> torch.Tensor:
    return torch.randn(N_LABELS, 3, generator=torch.Generator().manual_seed(seed))


def _mention(seed: int = 1) -> torch.Tensor:
    return torch.randn(N_LABELS, generator=torch.Generator().manual_seed(seed))


# -- features ---------------------------------------------------------------


@pytest.mark.parametrize(
    "mode,width", [(FINDING_TOKENS_Q_ONLY, 3), (FINDING_TOKENS_FULL, 4)]
)
def test_feature_shape_is_thirteen_by_width(mode, width):
    features = finding_features(_logits(), _mention(), mode)
    assert features.shape == (NUM_FINDING_TOKENS, width)
    assert feature_width(mode) == width


def test_no_finding_is_dropped_and_the_rest_keep_classifier_order():
    """Row i must be MODELED_FINDINGS[i], i.e. ABNORMALITIES_14[i + 1].

    Order is checked by construction rather than by name, because the features
    carry no names: a silent off-by-one here would attach every finding's
    numbers to its neighbour and nothing downstream could notice.
    """
    fig9 = pytest.importorskip("training.train_eval_figure9_llm_variants_200")
    assert fig9.ABNORMALITIES_14[0] == "No Finding"
    assert tuple(fig9.ABNORMALITIES_14[1:]) == MODELED_FINDINGS
    assert len(MODELED_FINDINGS) == NUM_FINDING_TOKENS

    # A one-hot probe: make label k the only one with a distinctive q.
    for k in range(1, N_LABELS):
        logits = torch.zeros(N_LABELS, 3)
        logits[k, 1] = 20.0  # positive, essentially probability 1
        features = finding_features(logits, None, FINDING_TOKENS_Q_ONLY)
        hot = int(features[:, 1].argmax())
        assert hot == k - 1, f"label {k} landed on row {hot}"


def test_q_only_rows_are_a_probability_distribution():
    features = finding_features(_logits(), _mention(), FINDING_TOKENS_Q_ONLY)
    assert torch.allclose(features.sum(dim=-1), torch.ones(NUM_FINDING_TOKENS), atol=1e-5)
    assert (features >= 0).all()


def test_full_features_decompose_the_mention_gate():
    """m*q_pos + m*q_neg + m*q_unc == m, exactly the identity that makes m redundant."""
    features = finding_features(_logits(), _mention(), FINDING_TOKENS_FULL)
    m, pos, neg, unc = features.unbind(dim=-1)
    assert torch.allclose(pos + neg + unc, m, atol=1e-5)
    assert ((m >= 0) & (m <= 1)).all()


def test_full_features_match_sigmoid_times_softmax():
    logits, mention = _logits(3), _mention(4)
    features = finding_features(logits, mention, FINDING_TOKENS_FULL)
    q = torch.softmax(logits, dim=-1)
    m = torch.sigmoid(mention)
    for row, label in enumerate(range(1, N_LABELS)):
        assert features[row, 0].item() == pytest.approx(m[label].item(), abs=1e-6)
        assert features[row, 1].item() == pytest.approx(
            (m[label] * q[label, 1]).item(), abs=1e-6
        )


def test_low_mention_never_becomes_a_negative_assertion():
    """A near-zero gate must shrink all three polarity numbers, not flip one.

    "Not mentioned" is not "absent" -- q is undefined when the finding was never
    mentioned, and encoding that as a confident negative is the single most
    harmful thing this channel could do.
    """
    logits = torch.zeros(N_LABELS, 3)
    logits[:, 0] = 10.0  # q says negative
    mention = torch.full((N_LABELS,), -20.0)  # but it was almost certainly not mentioned
    features = finding_features(logits, mention, FINDING_TOKENS_FULL)
    assert features.abs().max() < 1e-6


def test_q_only_refuses_nothing_and_full_refuses_missing_mention():
    logits = _logits()
    finding_features(logits, None, FINDING_TOKENS_Q_ONLY)  # no mention needed
    with pytest.raises(ValueError, match="mention_logits"):
        finding_features(logits, None, FINDING_TOKENS_FULL)


def test_wrong_label_count_raises():
    with pytest.raises(ValueError, match="labels"):
        finding_features(torch.randn(13, 3), torch.randn(13), FINDING_TOKENS_FULL)
    with pytest.raises(ValueError, match="mention_logits has"):
        finding_features(_logits(), torch.randn(13), FINDING_TOKENS_FULL)


def test_off_has_no_features():
    with pytest.raises(ValueError):
        finding_features(_logits(), _mention(), FINDING_TOKENS_OFF)
    with pytest.raises(ValueError):
        feature_width(FINDING_TOKENS_OFF)


# -- encoder ----------------------------------------------------------------


def test_encoder_maps_to_the_language_hidden_size():
    encoder = FindingTokenEncoder(FINDING_TOKENS_FULL, hidden=32)
    out = encoder(torch.rand(2, NUM_FINDING_TOKENS, 4))
    assert out.shape == (2, NUM_FINDING_TOKENS, 32)


def test_encoder_rejects_the_wrong_feature_width():
    encoder = FindingTokenEncoder(FINDING_TOKENS_Q_ONLY, hidden=16)
    with pytest.raises(ValueError, match="wide"):
        encoder(torch.rand(1, NUM_FINDING_TOKENS, 4))
    with pytest.raises(ValueError, match="finding tokens"):
        encoder(torch.rand(1, 7, 3))


def test_identical_features_still_give_distinct_tokens():
    """Identity comes from the embedding, not from the numbers.

    If two findings share their Stage-1 numbers the tokens must still differ,
    or the model cannot tell which finding a token is about.
    """
    encoder = FindingTokenEncoder(FINDING_TOKENS_FULL, hidden=16)
    flat = torch.full((1, NUM_FINDING_TOKENS, 4), 0.25)
    out = encoder(flat)[0]
    pairwise = torch.cdist(out, out) + torch.eye(NUM_FINDING_TOKENS) * 1e3
    assert pairwise.min() > 1e-4


def test_the_numbers_change_the_token():
    """The counterpart: identical identity, different numbers, different token.

    Without the LayerNorm this is the test that fails once the identity
    embedding grows during training.
    """
    encoder = FindingTokenEncoder(FINDING_TOKENS_FULL, hidden=16)
    a = encoder(torch.zeros(1, NUM_FINDING_TOKENS, 4))
    b = encoder(torch.full((1, NUM_FINDING_TOKENS, 4), 0.9))
    assert (a - b).abs().max() > 1e-4


def test_output_scale_calibrates_to_the_embedding_output_not_its_weight():
    """Gemma scales inside the embedding forward; matching `weight` is ~60x wrong."""
    table = torch.nn.Embedding(64, 32)
    torch.nn.init.normal_(table.weight, std=1.0)
    multiplier = 5.0

    class Scaled(torch.nn.Embedding):
        def forward(self, ids):
            return super().forward(ids) * multiplier

    scaled = Scaled(64, 32)
    scaled.weight.data.copy_(table.weight.data)
    encoder = FindingTokenEncoder(FINDING_TOKENS_FULL, hidden=32)
    rms = encoder.calibrate_output_scale(scaled, torch.arange(64))
    weight_rms = table.weight.pow(2).mean().sqrt().item()
    assert rms == pytest.approx(weight_rms * multiplier, rel=1e-3)
    out = encoder(torch.rand(1, NUM_FINDING_TOKENS, 4))
    assert out.pow(2).mean().sqrt().item() == pytest.approx(rms, rel=0.05)


def test_gradient_reaches_the_encoder_and_stops_at_the_features():
    """Stage 1 is frozen: the features are data, not a differentiable input."""
    encoder = FindingTokenEncoder(FINDING_TOKENS_FULL, hidden=8)
    features = finding_features(_logits(), _mention(), FINDING_TOKENS_FULL).unsqueeze(0)
    assert features.grad_fn is None and not features.requires_grad
    encoder(features).sum().backward()
    for name, param in encoder.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
    assert encoder.identity.weight.grad.abs().sum() > 0
    assert encoder.output_scale.grad is not None


# -- substitution -----------------------------------------------------------


class _Base(torch.nn.Module):
    """Deterministic stand-in for the LM embedding table."""

    def __init__(self, vocab=64, hidden=8):
        super().__init__()
        self.table = torch.nn.Embedding(vocab, hidden)
        torch.nn.init.normal_(self.table.weight, std=1.0)

    @property
    def weight(self):
        return self.table.weight

    def forward(self, ids):
        return self.table(ids)


def _ids_with_findings(batch=2, prefix=3, token_id=50):
    seq = []
    for _ in range(batch):
        seq.append([1] * prefix + [token_id] * NUM_FINDING_TOKENS + [2, 3])
    return torch.tensor(seq)


def test_substitution_hits_exactly_the_finding_positions():
    base, token_id = _Base(), 50
    ids = _ids_with_findings(token_id=token_id)
    values = torch.arange(2 * NUM_FINDING_TOKENS * 8, dtype=torch.float32).reshape(
        2, NUM_FINDING_TOKENS, 8
    )
    wrapper = FindingTokenEmbeddingWrapper(base, token_id, values)
    out = wrapper(ids)
    plain = base(ids)
    mask = ids == token_id
    assert torch.equal(out[mask], values.reshape(-1, 8))
    assert torch.equal(out[~mask], plain[~mask])


def test_each_row_gets_its_own_predictions():
    """The failure this fails closed on: study A's cues on study B's report."""
    base, token_id = _Base(), 50
    ids = _ids_with_findings(batch=3, token_id=token_id)
    values = torch.stack(
        [torch.full((NUM_FINDING_TOKENS, 8), float(i)) for i in range(3)]
    )
    out = FindingTokenEmbeddingWrapper(base, token_id, values)(ids)
    for row in range(3):
        got = out[row][ids[row] == token_id]
        assert torch.equal(got, torch.full((NUM_FINDING_TOKENS, 8), float(row)))


def test_wrong_batch_count_and_wrong_position_count_raise():
    base, token_id = _Base(), 50
    ids = _ids_with_findings(batch=2, token_id=token_id)
    with pytest.raises(ValueError, match="do not match the batch"):
        FindingTokenEmbeddingWrapper(base, token_id, torch.zeros(1, NUM_FINDING_TOKENS, 8))(ids)
    short = torch.tensor([[1, token_id, token_id, 2]])
    with pytest.raises(ValueError, match="expected 13 finding tokens"):
        FindingTokenEmbeddingWrapper(base, token_id, torch.zeros(1, NUM_FINDING_TOKENS, 8))(short)


def test_wrong_hidden_width_raises():
    base, token_id = _Base(), 50
    ids = _ids_with_findings(token_id=token_id)
    with pytest.raises(ValueError, match="width does not match"):
        FindingTokenEmbeddingWrapper(base, token_id, torch.zeros(2, NUM_FINDING_TOKENS, 5))(ids)


def test_a_sequence_without_finding_tokens_is_untouched():
    base, token_id = _Base(), 50
    ids = torch.tensor([[1, 2, 3, 4]])
    wrapper = FindingTokenEmbeddingWrapper(base, token_id, torch.zeros(1, NUM_FINDING_TOKENS, 8))
    assert torch.equal(wrapper(ids), base(ids))


def test_composition_with_the_soft_token_wrapper_substitutes_both():
    """Composition, not modification: the soft-token wrapper is untouched code."""
    from training.medgemma.soft_tokens import SoftTokenEmbeddingWrapper

    base, soft_id, find_id, n_soft = _Base(vocab=64, hidden=8), 40, 50, 4
    ids = torch.tensor([[1] + [soft_id] * n_soft + [find_id] * NUM_FINDING_TOKENS + [2]])
    soft_values = torch.full((1, n_soft, 8), 7.0)
    find_values = torch.full((1, NUM_FINDING_TOKENS, 8), -3.0)
    inner = SoftTokenEmbeddingWrapper(base, soft_id, soft_values, n_soft)
    outer = FindingTokenEmbeddingWrapper(inner, find_id, find_values)
    out = outer(ids)[0]
    assert torch.equal(out[ids[0] == soft_id], soft_values[0])
    assert torch.equal(out[ids[0] == find_id], find_values[0])
    assert torch.equal(out[0], base(ids)[0][0])


def test_gradient_flows_through_the_substituted_positions():
    base, token_id = _Base(), 50
    ids = _ids_with_findings(batch=1, token_id=token_id)
    encoder = FindingTokenEncoder(FINDING_TOKENS_FULL, hidden=8)
    projected = encoder(torch.rand(1, NUM_FINDING_TOKENS, 4))
    FindingTokenEmbeddingWrapper(base, token_id, projected)(ids).sum().backward()
    assert encoder.proj.weight.grad is not None
    assert encoder.proj.weight.grad.abs().sum() > 0


# -- inference ablations ----------------------------------------------------


def test_zero_ablation_keeps_the_shape_and_drops_the_numbers():
    features = finding_features(_logits(), _mention(), FINDING_TOKENS_FULL)
    out = apply_finding_feature_ablation(features, FEATURE_ABLATION_ZERO)
    assert out.shape == features.shape
    assert out.abs().max() == 0


def test_shuffle_ablation_permutes_rows_and_is_deterministic():
    features = finding_features(_logits(), _mention(), FINDING_TOKENS_FULL)
    a = apply_finding_feature_ablation(features, FEATURE_ABLATION_SHUFFLE, seed=16)
    b = apply_finding_feature_ablation(features, FEATURE_ABLATION_SHUFFLE, seed=16)
    assert torch.equal(a, b)
    assert sorted(a.sum(dim=-1).tolist()) == pytest.approx(
        sorted(features.sum(dim=-1).tolist())
    )
    assert not torch.equal(a, features)


def test_no_ablation_is_the_identity():
    features = finding_features(_logits(), _mention(), FINDING_TOKENS_FULL)
    assert torch.equal(apply_finding_feature_ablation(features, None), features)


# -- prompt integration -----------------------------------------------------


def _guided_config():
    from stage2.prompts.schemas import VisualMode
    from stage2.prompts.validation import PromptConfig

    return PromptConfig(visual_mode=VisualMode.NATIVE_QFORMER_GUIDED)


def _context(finding_token_count=None):
    from stage2.prompts.schemas import PromptContext, VisualMode

    return PromptContext(
        study_id="SYNTHETIC",
        visual_mode=VisualMode.NATIVE_QFORMER_GUIDED,
        positive_findings=("Edema",),
        qformer_token_count=32,
        finding_token_count=finding_token_count,
    )


def test_the_default_prompt_is_unchanged_when_the_branch_is_off():
    """Arms A and B must execute the prompt that produced the recorded numbers."""
    from stage2.prompts.builder import PromptBuilder
    from stage2.prompts.schemas import PartKind

    rendered = PromptBuilder(_guided_config()).build(_context(None))
    assert not any(part.kind is PartKind.FINDING_TOKENS for part in rendered.parts)
    assert FINDING_TOKEN not in rendered.user_text()


def test_thirteen_placeholders_are_emitted_when_the_branch_is_on():
    from stage2.prompts.builder import PromptBuilder
    from stage2.prompts.schemas import PartKind

    rendered = PromptBuilder(_guided_config()).build(_context(NUM_FINDING_TOKENS))
    parts = [part for part in rendered.parts if part.kind is PartKind.FINDING_TOKENS]
    assert len(parts) == 1 and parts[0].count == NUM_FINDING_TOKENS
    assert parts[0].budget_priority == 0  # never dropped by truncation
    assert rendered.user_text().count(FINDING_TOKEN) == NUM_FINDING_TOKENS


def test_finding_tokens_sit_after_the_soft_tokens_and_before_the_instruction():
    """Causal decoder: the instruction and the target must be able to see them."""
    from stage2.prompts.builder import PromptBuilder
    from stage2.prompts.schemas import PartKind

    parts = PromptBuilder(_guided_config()).build(_context(NUM_FINDING_TOKENS)).parts
    kinds = [part.kind for part in parts]
    assert kinds.index(PartKind.SOFT_TOKENS) < kinds.index(PartKind.FINDING_TOKENS)
    assert kinds.index(PartKind.FINDING_TOKENS) < len(kinds) - 1


def test_the_template_hash_moves_only_when_the_branch_is_on():
    from stage2.prompts.builder import PromptBuilder

    off = PromptBuilder(_guided_config()).build(_context(None))
    on = PromptBuilder(_guided_config()).build(_context(NUM_FINDING_TOKENS))
    assert off.template_hash != on.template_hash
    assert off.config_hash == on.config_hash  # the prompt YAML did not change


def test_the_placeholder_string_is_distinct_from_the_soft_token():
    from stage2.prompts.schemas import PromptContext  # noqa: F401

    assert FINDING_TOKEN != "<qformer_soft_token>"
    assert FINDING_TOKEN not in "<qformer_soft_token>"


# -- wiring: artifacts, cache identity, CLI refusals ------------------------


def _fake_adapter(tmp_path: Path, *, with_finding_encoder: bool) -> Path:
    import json

    adapter = tmp_path / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"")
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "meta.json").write_text("{}")
    from training.stage2_utils import SCHEMA_VERSION

    (adapter / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "status": "complete",
                "image_mode": "native_qformer",
            }
        )
    )
    torch.save({}, adapter / "trainer_state.pt")
    torch.save({}, adapter / "img_proj.pt")
    if with_finding_encoder:
        torch.save({"mode": FINDING_TOKENS_FULL, "state_dict": {}}, adapter / "finding_tokens.pt")
    return adapter


def test_an_adapter_without_the_encoder_is_incomplete_only_when_the_branch_is_on(tmp_path):
    """A missing encoder is exactly as silent as a missing img_proj.pt."""
    from training.stage2_utils import adapter_is_complete

    adapter = _fake_adapter(tmp_path, with_finding_encoder=False)
    assert adapter_is_complete(adapter, "native_qformer", FINDING_TOKENS_OFF)
    assert not adapter_is_complete(adapter, "native_qformer", FINDING_TOKENS_FULL)

    full = _fake_adapter(tmp_path / "b", with_finding_encoder=True)
    assert adapter_is_complete(full, "native_qformer", FINDING_TOKENS_FULL)


def test_resume_refuses_a_checkpoint_without_the_encoder(tmp_path):
    run = pytest.importorskip("training.run_medgemma_qlora")
    adapter = _fake_adapter(tmp_path, with_finding_encoder=False)
    assert run.resumable_adapter(adapter, "native_qformer", FINDING_TOKENS_OFF)
    assert not run.resumable_adapter(adapter, "native_qformer", FINDING_TOKENS_FULL)


def test_cache_identity_changes_only_when_the_branch_is_on(tmp_path):
    """Records built before this branch carry no class_logits.

    The fingerprint must therefore differ for a finding-token run -- and must
    NOT differ otherwise, or every existing 10 GiB cache would be rebuilt.
    """
    fig9 = pytest.importorskip("training.train_eval_figure9_llm_variants_200")
    from training.run_context import Stage1Context

    context = Stage1Context(run_name="mimic_cxr_full_blip2", thresholds={})
    args = (context, tmp_path, "val", None)
    off, off_payload = fig9.stage1_cohort_fingerprint(*args, "none", FINDING_TOKENS_OFF)
    on, on_payload = fig9.stage1_cohort_fingerprint(*args, "none", FINDING_TOKENS_FULL)
    assert off != on
    assert "record_features" not in off_payload
    assert on_payload["record_features"] == "with_class_logits"
    # q_only and full read the same cached tensors, so they share an identity.
    q_only, _ = fig9.stage1_cohort_fingerprint(*args, "none", FINDING_TOKENS_Q_ONLY)
    assert q_only == on


def test_records_without_class_logits_raise_for_a_finding_token_run():
    fig9 = pytest.importorskip("training.train_eval_figure9_llm_variants_200")

    stale = [{"sample_key": "k", "mention_logits": torch.zeros(14)}]
    fig9.assert_class_logits_present(stale, FINDING_TOKENS_OFF)  # off: no opinion
    with pytest.raises(RuntimeError, match="class_logits"):
        fig9.assert_class_logits_present(stale, FINDING_TOKENS_FULL)

    good = [{"sample_key": "k", "class_logits": torch.zeros(14, 3)}]
    fig9.assert_class_logits_present(good, FINDING_TOKENS_FULL)
    wrong = [{"sample_key": "k", "class_logits": torch.zeros(13, 3)}]
    with pytest.raises(RuntimeError, match="expected"):
        fig9.assert_class_logits_present(wrong, FINDING_TOKENS_FULL)


def _gen_args(**overrides):
    import argparse

    base = dict(
        cue_rule="none",
        finding_tokens=FINDING_TOKENS_OFF,
        finding_feature_ablation=None,
        prompt_config=None,
        adapter=None,
        manifest=Path("m.csv"),
        image_root=Path("images"),
        checkpoint_root=Path("."),
        stage1_checkpoint=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_generation_refuses_finding_tokens_without_a_trained_encoder(tmp_path):
    """The zero-shot form does not exist: the encoder is trained in Stage 2."""
    gen = pytest.importorskip("scripts.generate_stage2_reports")
    from training.pipeline_modes import resolve_pipeline_modes

    mode = resolve_pipeline_modes("meta_cxr_native_qformer_guided")[0]
    cfg = tmp_path / "prompt.yaml"
    cfg.write_text("visual_mode: native_qformer_guided\n")

    with pytest.raises(SystemExit, match="Stage-1 pipeline mode"):
        gen.validate_invocation(
            _gen_args(finding_tokens=FINDING_TOKENS_FULL),
            resolve_pipeline_modes("medgemma_direct")[0],
        )
    with pytest.raises(SystemExit, match="guided --prompt-config"):
        gen.validate_invocation(_gen_args(finding_tokens=FINDING_TOKENS_FULL), mode)
    with pytest.raises(SystemExit, match="no zero-shot form"):
        gen.validate_invocation(
            _gen_args(finding_tokens=FINDING_TOKENS_FULL, prompt_config=cfg), mode
        )
    adapter = _fake_adapter(tmp_path, with_finding_encoder=False)
    with pytest.raises(SystemExit, match="finding_tokens.pt"):
        gen.validate_invocation(
            _gen_args(finding_tokens=FINDING_TOKENS_FULL, prompt_config=cfg, adapter=adapter),
            mode,
        )


def test_an_ablation_without_the_branch_is_refused(tmp_path):
    gen = pytest.importorskip("scripts.generate_stage2_reports")
    from training.pipeline_modes import resolve_pipeline_modes

    mode = resolve_pipeline_modes("meta_cxr_native_qformer_guided")[0]
    with pytest.raises(SystemExit, match="needs --finding-tokens"):
        gen.validate_invocation(
            _gen_args(finding_feature_ablation="zero", cue_rule="conditional_positive"),
            mode,
        )


def test_permute_across_gives_every_study_someone_elses_predictions():
    gen = pytest.importorskip("scripts.generate_stage2_reports")

    records = [
        {"sample_key": str(i), "class_logits": torch.full((14, 3), float(i)),
         "mention_logits": torch.full((14,), float(i))}
        for i in range(5)
    ]
    out = gen.permute_finding_features_across_studies(records, seed=16)
    assert len(out) == len(records)
    for original, permuted in zip(records, out, strict=True):
        assert permuted["sample_key"] == original["sample_key"]
        assert not torch.equal(permuted["class_logits"], original["class_logits"])
    # It is a derangement of the SAME set, not new numbers.
    donated = sorted(float(r["class_logits"][0, 0]) for r in out)
    assert donated == sorted(float(r["class_logits"][0, 0]) for r in records)
