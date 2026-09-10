"""Learnable finding tokens — Stage-1 mention/polarity as embedding-space tokens.

EXPERIMENTAL. Every entry point here is reached only when ``--finding-tokens``
is set to something other than ``off``; with the flag off nothing in this module
is imported at runtime and the default Stage-2 path is byte-identical.

Stage 1 gives two numbers per finding: a mention gate ``m = sigmoid(mention_logits)``
answering "will the report mention this at all", and a polarity distribution
``q = softmax(classification_logits)`` over {negative, positive, uncertain} that
is CONDITIONAL on mention -- 79.5% of the CheXpert matrix is blank and masked
out of the classification loss, so ``q`` never saw "absent from the report".
Every recorded experiment passed that pair to Stage 2 through a hard threshold
and an English sentence. This module passes it as 13 continuous tokens instead.

**"Not mentioned" is never turned into "negative".** A low ``m`` scales all three
polarity numbers toward zero, which is the honest encoding of "Stage 1 has no
opinion here"; it never produces a negative assertion.

Depends on nothing but torch, so the shape, ordering, masking and gradient
contracts are unit-testable on a CPU box without MedGemma weights or a GPU.
"""

from __future__ import annotations

import torch
import torch.nn as nn

#: Placeholder the prompt carries; substituted the way ``<qformer_soft_token>`` is.
FINDING_TOKEN = "<finding_token>"

#: The 13 reportable findings, i.e. ``ABNORMALITIES_14`` without ``No Finding``.
#: Mirrored here (and pinned by a test against the classifier) because this
#: module must import without torch/transformers-heavy Stage-1 code.
NUM_FINDING_TOKENS = 13

#: Index of ``No Finding`` in ``ABNORMALITIES_14``. It is excluded because it has
#: zero negatives in every split by construction, so its ``q`` can only be a
#: constant -- see CLAUDE.md, "``No Finding`` is back in the classification head".
NO_FINDING_INDEX = 0

FINDING_TOKENS_OFF = "off"
FINDING_TOKENS_Q_ONLY = "q_only"
FINDING_TOKENS_FULL = "full"
FINDING_TOKEN_MODES = (FINDING_TOKENS_OFF, FINDING_TOKENS_Q_ONLY, FINDING_TOKENS_FULL)

#: ``CLASS_MAP`` in the Stage-2 runner: negative=0, positive=1, uncertain=2.
CLASS_NEGATIVE, CLASS_POSITIVE, CLASS_UNCERTAIN = 0, 1, 2

#: Numeric feature width per mode. ``q_only`` is the control that isolates the
#: mention contribution: same identity, same projection, same 13 positions, only
#: ``m`` removed.
FEATURE_WIDTHS = {FINDING_TOKENS_Q_ONLY: 3, FINDING_TOKENS_FULL: 4}

DEFAULT_IDENTITY_DIM = 64


def validate_mode(mode: str) -> str:
    if mode not in FINDING_TOKEN_MODES:
        raise ValueError(f"finding_tokens must be one of {FINDING_TOKEN_MODES}, got {mode!r}")
    return mode


def feature_width(mode: str) -> int:
    """Numeric features per finding token. Raises for ``off`` -- there are none."""
    validate_mode(mode)
    if mode == FINDING_TOKENS_OFF:
        raise ValueError("finding_tokens='off' has no feature width")
    return FEATURE_WIDTHS[mode]


def finding_features(
    class_logits: torch.Tensor,
    mention_logits: torch.Tensor | None,
    mode: str,
) -> torch.Tensor:
    """``[14, 3]`` + ``[14]`` Stage-1 outputs -> ``[13, k]`` token features.

    Row ``i`` of the result is ``ABNORMALITIES_14[i + 1]``: ``No Finding`` is
    dropped and the remaining order is preserved exactly, so it matches
    ``stage2.prompts.ontology.MODELED_FINDINGS`` position for position.

    ``q_only``  -> ``[q_neg, q_pos, q_unc]``
    ``full``    -> ``[m, m*q_pos, m*q_neg, m*q_unc]``

    ``m`` is linearly redundant in ``full`` (the three products sum to it); it is
    kept so a single linear projection has a direct route to "was it mentioned"
    without having to sum three inputs.

    ⚠ ``m`` is NOT a calibrated mention probability -- the gate trains with
    inverse-frequency weights capped at 10, so its odds are inflated. That is
    fine for a learned projection, which can absorb a monotone rescaling, but it
    means these numbers must never be reported as probabilities.
    """
    validate_mode(mode)
    if mode == FINDING_TOKENS_OFF:
        raise ValueError("finding_tokens='off' produces no features")

    logits = torch.as_tensor(class_logits).float()
    if logits.dim() != 2 or logits.shape[-1] != 3:
        raise ValueError(f"class_logits must be [labels, 3], got {tuple(logits.shape)}")
    n_labels = logits.shape[0]
    if n_labels != NUM_FINDING_TOKENS + 1:
        raise ValueError(
            f"class_logits has {n_labels} labels, expected {NUM_FINDING_TOKENS + 1} "
            "(ABNORMALITIES_14)"
        )
    q = torch.softmax(logits, dim=-1)

    if mode == FINDING_TOKENS_Q_ONLY:
        features = q
    else:
        if mention_logits is None:
            raise ValueError(
                "finding_tokens='full' needs mention_logits. Records built before "
                "the gate was threaded through carry none; rebuild them rather "
                "than silently falling back to q-only features."
            )
        gate = torch.as_tensor(mention_logits).reshape(-1).float()
        if gate.numel() != n_labels:
            raise ValueError(
                f"mention_logits has {gate.numel()} entries, expected {n_labels}"
            )
        m = torch.sigmoid(gate).unsqueeze(-1)
        features = torch.cat(
            [
                m,
                m * q[:, CLASS_POSITIVE : CLASS_POSITIVE + 1],
                m * q[:, CLASS_NEGATIVE : CLASS_NEGATIVE + 1],
                m * q[:, CLASS_UNCERTAIN : CLASS_UNCERTAIN + 1],
            ],
            dim=-1,
        )

    keep = [i for i in range(n_labels) if i != NO_FINDING_INDEX]
    out = features[keep].contiguous()
    if out.shape != (NUM_FINDING_TOKENS, feature_width(mode)):
        raise RuntimeError(f"finding features came out {tuple(out.shape)}")
    return out


class FindingTokenEncoder(nn.Module):
    """``[B, 13, k]`` Stage-1 features -> ``[B, 13, hidden]`` LM-space vectors.

    ``token_i = gamma * rms_norm( W @ LayerNorm(concat(E[i], f_i)) + b )``.

    Three deliberate choices, each of which is a way this could have silently
    measured nothing:

    * **Identity comes from a learnable embedding, not from position.** The
      ordering is fixed and pinned by a test, but the model is not asked to infer
      which finding a token means from where it sits. ``W`` is therefore SHARED
      across findings, so the numeric features carry one consistent meaning
      instead of 13 independently-learned ones.
    * **LayerNorm before the projection.** ``E[i]`` is free to grow during
      training while ``f_i`` is bounded in [0, 1]; without normalisation the
      numbers would become a vanishing fraction of the input norm and the
      projection could learn to ignore them -- which would then be misreported
      as "mention information does not help".
    * **RMS rescaling to the embedding table's own scale.** Substitution happens
      inside ``get_input_embeddings()``, i.e. AFTER Gemma's scaled-word-embedding
      multiplier has been applied to real tokens but not to substituted vectors.
      ``gamma`` starts at the measured RMS of the base embedding output so the
      finding tokens are in-distribution from step 0.
    """

    def __init__(
        self,
        mode: str,
        hidden: int,
        identity_dim: int = DEFAULT_IDENTITY_DIM,
        num_findings: int = NUM_FINDING_TOKENS,
    ):
        super().__init__()
        self.mode = validate_mode(mode)
        if mode == FINDING_TOKENS_OFF:
            raise ValueError("FindingTokenEncoder cannot be built for finding_tokens='off'")
        self.num_findings = int(num_findings)
        self.feature_width = feature_width(mode)
        self.hidden = int(hidden)
        width = identity_dim + self.feature_width
        self.identity = nn.Embedding(self.num_findings, identity_dim)
        nn.init.normal_(self.identity.weight, mean=0.0, std=0.02)
        self.norm = nn.LayerNorm(width)
        self.proj = nn.Linear(width, self.hidden)
        # Scalar, learnable. Initialised to 1.0 and overwritten by
        # calibrate_output_scale() once the real embedding table is available.
        self.output_scale = nn.Parameter(torch.ones(()))

    @torch.no_grad()
    def calibrate_output_scale(self, base_embedding: nn.Module, sample_ids: torch.Tensor) -> float:
        """Set ``gamma`` to the RMS of the base embedding's OUTPUT.

        The output, not ``weight``: Gemma applies its ``sqrt(hidden)`` multiplier
        inside the embedding module's forward, so the weight's RMS is off by that
        factor and a token built to match it would arrive ~60x too small.
        """
        ids = torch.as_tensor(sample_ids).reshape(-1)
        embeds = base_embedding(ids.to(next(base_embedding.parameters()).device))
        rms = embeds.float().pow(2).mean().sqrt().item()
        if not (rms > 0) or rms != rms:  # zero or NaN
            raise RuntimeError(f"base embedding RMS came out {rms}; refusing to calibrate")
        self.output_scale.fill_(rms)
        return rms

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() != 3:
            raise ValueError(f"finding features must be [B, N, k], got {tuple(features.shape)}")
        if features.shape[1] != self.num_findings:
            raise ValueError(
                f"expected {self.num_findings} finding tokens, got {features.shape[1]}"
            )
        if features.shape[2] != self.feature_width:
            raise ValueError(
                f"finding features are {features.shape[2]} wide, mode {self.mode!r} "
                f"expects {self.feature_width}"
            )
        batch = features.shape[0]
        ids = torch.arange(self.num_findings, device=features.device)
        identity = self.identity(ids).unsqueeze(0).expand(batch, -1, -1)
        x = torch.cat([identity, features.float()], dim=-1)
        x = self.proj(self.norm(x))
        rms = x.pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
        return self.output_scale * x / rms


class FindingTokenEmbeddingWrapper(nn.Module):
    """Substitutes finding-token vectors at ``<finding_token>`` positions.

    Composes with ``SoftTokenEmbeddingWrapper`` rather than modifying it: the
    inner wrapper substitutes the 32 Q-Former soft tokens, this one substitutes
    the 13 finding tokens on top of whatever the inner call returned. That keeps
    the arms without finding tokens executing exactly the code that produced
    every recorded result.

    Fails closed on any batch/count/width mismatch, for the same reason the soft
    token wrapper does: a clamp here would feed one study's Stage-1 predictions
    to another study's report with no error anywhere.
    """

    def __init__(
        self,
        base_embedding: nn.Module,
        finding_token_id: int,
        projected_finding_embs: torch.Tensor,
        num_finding_tokens: int = NUM_FINDING_TOKENS,
    ):
        super().__init__()
        self.base_embedding = base_embedding
        self.finding_token_id = int(finding_token_id)
        self.projected_finding_embs = projected_finding_embs
        self.num_finding_tokens = int(num_finding_tokens)

    @property
    def weight(self):
        return getattr(self.base_embedding, "weight", None)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeds = self.base_embedding(input_ids)
        mask = input_ids == self.finding_token_id
        if not mask.any():
            return embeds

        if self.projected_finding_embs.dim() != 3:
            raise ValueError(
                "projected finding embeddings must be [B, N, D], got "
                f"{tuple(self.projected_finding_embs.shape)}"
            )
        hidden = embeds.shape[-1]
        if self.projected_finding_embs.shape[-1] != hidden:
            raise ValueError(
                "projected finding embedding width does not match the language model: "
                f"{self.projected_finding_embs.shape[-1]} vs {hidden}"
            )
        if self.projected_finding_embs.shape[0] != input_ids.shape[0]:
            raise ValueError(
                "projected finding embeddings do not match the batch: "
                f"{self.projected_finding_embs.shape[0]} for {input_ids.shape[0]} sequences"
            )

        embeds = embeds.clone()
        for batch_idx in range(input_ids.shape[0]):
            positions = mask[batch_idx].nonzero(as_tuple=False).flatten()
            if len(positions) != self.num_finding_tokens:
                raise ValueError(
                    f"expected {self.num_finding_tokens} finding tokens, got {len(positions)}"
                )
            row = self.projected_finding_embs[batch_idx]
            embeds[batch_idx, positions, :] = row.to(device=embeds.device, dtype=embeds.dtype)
        return embeds


#: Inference-only interventions used to ask whether the model actually reads the
#: mention channel. ⚠ These are OUT-OF-DISTRIBUTION at inference and are a
#: mechanism probe, never a substitute for the separately-trained ``q_only`` arm:
#: a model can degrade under a corrupted input while still not benefiting from a
#: correct one.
FEATURE_ABLATION_NONE = None
FEATURE_ABLATION_ZERO = "zero"
FEATURE_ABLATION_SHUFFLE = "shuffle_within"
#: Applied by the caller, not here: it needs a second study's features.
FEATURE_ABLATION_PERMUTE = "permute_across"
FEATURE_ABLATIONS = (
    FEATURE_ABLATION_ZERO,
    FEATURE_ABLATION_SHUFFLE,
    FEATURE_ABLATION_PERMUTE,
)


def apply_finding_feature_ablation(
    features: torch.Tensor, ablation: str | None, seed: int = 16
) -> torch.Tensor:
    """Corrupt the numeric features while leaving identity and count intact.

    ``zero`` removes the Stage-1 numbers entirely; ``shuffle_within`` keeps them
    but attaches each to the wrong finding, so the token says "Edema" while
    carrying Pneumothorax's numbers. The permutation is fixed by ``seed`` and
    identical for every study, so the intervention is reproducible.
    """
    if ablation is None or ablation == FEATURE_ABLATION_PERMUTE:
        return features
    if ablation == FEATURE_ABLATION_ZERO:
        return torch.zeros_like(features)
    if ablation == FEATURE_ABLATION_SHUFFLE:
        generator = torch.Generator().manual_seed(int(seed))
        order = torch.randperm(features.shape[0], generator=generator)
        return features[order].contiguous()
    raise ValueError(f"unknown finding-feature ablation {ablation!r}")
