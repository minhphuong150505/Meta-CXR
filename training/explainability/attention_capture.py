"""The only module here that touches a live model.

Everything in :mod:`training.explainability.rollout`,
:mod:`~.projection` and :mod:`~.sentence_attribution` is pure enough to test on
a CPU box precisely so that this module can stay small and be the single place
where a real forward pass happens.

It never mutates the model it is given beyond the duration of one call: hooks
are removed and every flag is restored in ``finally``. It imports nothing from
the Stage-1 or Stage-2 training path, and nothing there imports it.

Four memory facts, each of which cost an OOM to learn on the 16 GB RTX 5060 Ti
(measured 2026-08-30, ``google/medgemma-1.5-4b-it``):

1. **Never pass ``output_attentions=True``.** It propagates into the SigLIP
   vision tower, where 27 layers x 16 heads x 4096^2 in fp32 is roughly 27 GiB
   of retained tensors. Hook the language model's attention modules instead.
2. **The vision tower must use ``sdpa``.** Even without ``output_attentions``,
   eager SigLIP retains its own softmax for backward and OOMs the same way.
   ``attn_implementation={"text_config": "eager", "vision_config": "sdpa"}`` is
   accepted by transformers 4.53 and is what this module asks for.
3. **Freeze every parameter and use ``torch.autograd.grad``.** A plain
   ``.backward()`` allocates a gradient for all ~4B parameters -- about 8 GiB
   that nothing reads. With the weights frozen, the input embeddings carry
   ``requires_grad`` instead, which is what keeps the graph alive at all.
4. **Ask for as few logit rows as possible** (``logits_to_keep``). The full
   tensor is ``[1, S, 262208]``.

With all four applied: peak 9.39 GiB of 15.48, gradient reaching 34 of 34
layers with non-zero values in every one. So the gradient-weighted rollout runs
in bf16 and NF4 is not needed -- which removes the "backward through NF4"
risk entirely on this hardware.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch

try:
    from training.explainability import projection, rollout
except ImportError:  # pragma: no cover - direct-script execution
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from training.explainability import projection, rollout

MEDGEMMA_MODEL_ID = "google/medgemma-1.5-4b-it"

#: The only splits an explanation may be computed on. ``train`` is refused, not
#: warned about: the training transform applies RandomAffine, so a map computed
#: there is registered to a geometry that exists for exactly one sample draw and
#: cannot be laid over the image again.
ALLOWED_SPLITS = ("val", "test")

SOURCE_MEDGEMMA_IMAGE = "medgemma_image_grid"
SOURCE_QFORMER_SOFT_TOKEN = "qformer_soft_token"


class TrainSplitRefused(RuntimeError):
    """Raised when an explanation is requested on the training split."""


class SoftTokenAblationFailed(RuntimeError):
    """The visual tokens did not measurably affect the output.

    Its own type because it is a STOP condition, not a degraded result: either
    the token positions are wrong, or the model is ignoring its visual input.
    Both need a human before anything downstream is believed.
    """


@dataclass(frozen=True)
class CaptureTrace:
    """Everything about how a capture ran that could change its meaning."""

    model_id: str
    split: str
    dtype: str
    num_layers: int
    num_heads: int
    sequence_length: int
    gradient_weighted: bool
    gradient_fallback_reason: str | None
    gradient_checkpointing_disabled: bool
    attn_implementation: dict[str, str]
    peak_vram_bytes: int | None

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "split": self.split,
            "dtype": self.dtype,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "sequence_length": self.sequence_length,
            "gradient_weighted": self.gradient_weighted,
            "gradient_fallback_reason": self.gradient_fallback_reason,
            "gradient_checkpointing_disabled": self.gradient_checkpointing_disabled,
            "attn_implementation": dict(self.attn_implementation),
            "peak_vram_bytes": self.peak_vram_bytes,
        }


@dataclass(frozen=True)
class VisualSpan:
    """Where the visual tokens sit in the language model's sequence."""

    start: int
    end: int  # half-open
    source: str
    grid: projection.GridSpec | None = None

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError(f"empty visual span [{self.start}, {self.end})")

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def positions(self) -> list[int]:
        return list(range(self.start, self.end))

    def to_dict(self) -> dict[str, object]:
        return {
            "start": self.start,
            "end": self.end,
            "source": self.source,
            "grid": self.grid.to_dict() if self.grid is not None else None,
        }


@dataclass
class CapturedAttention:
    """One teacher-forced pass: attention, gradients, and where things are."""

    attention: torch.Tensor  # [L, H, S, S]
    visual_span: VisualSpan
    target_positions: tuple[int, ...]
    token_texts: tuple[str, ...]
    token_nll: tuple[float, ...]
    trace: CaptureTrace
    gradients: torch.Tensor | None = None
    _graph: object = field(default=None, repr=False)


def assert_split_allowed(split: str) -> str:
    """Refuse the training split outright.

    A warning would not do. The train transform applies RandomAffine with
    ``affine_p > 0``, so the image a train-split map describes is a one-off
    random warp of the radiograph. The map cannot be registered back to
    anything, and a warning in a log is not a barrier to someone publishing it.
    """
    value = str(split).strip().lower()
    if value == "train":
        raise TrainSplitRefused(
            "explanations may not be computed on the training split: its "
            "transform applies RandomAffine, so the geometry a map is "
            f"registered to exists only for that sample draw. Use one of {ALLOWED_SPLITS}"
        )
    if value not in ALLOWED_SPLITS:
        raise ValueError(f"split must be one of {ALLOWED_SPLITS} (got {split!r})")
    return value


def disable_gradient_checkpointing(model) -> bool:
    """Turn checkpointing off for the explanation pass, and say whether it was on.

    Not a nicety. With checkpointing on, the forward that a hook observes is a
    RECOMPUTED forward, so the attention captured during backward need not be
    the attention that produced the logits being explained. This path is
    inference-only and has no memory pressure worth trading for that ambiguity:
    the measured peak is 9.39 of 15.48 GiB.
    """
    was_enabled = bool(getattr(model, "is_gradient_checkpointing", False))
    disable = getattr(model, "gradient_checkpointing_disable", None)
    if was_enabled and callable(disable):
        disable()
    if getattr(model, "config", None) is not None:
        model.config.use_cache = False
    return was_enabled


def language_attention_modules(model) -> list[tuple[str, object]]:
    """The language model's self-attention modules, in layer order.

    Selected by name rather than by type so the vision tower can never be
    included by accident -- capturing SigLIP is the 27 GiB mistake.
    """
    found = []
    for name, module in model.named_modules():
        if not name.endswith(".self_attn"):
            continue
        if ".language_model.layers." not in f".{name}":
            continue
        index_text = name.split(".layers.")[1].split(".")[0]
        if not index_text.isdigit():
            continue
        found.append((int(index_text), name, module))
    if not found:
        raise RuntimeError(
            "no language-model self-attention modules found; the module naming "
            "has changed and capturing the wrong ones would silently explain "
            "the vision tower instead"
        )
    found.sort(key=lambda item: item[0])
    return [(name, module) for _index, name, module in found]


@contextmanager
def capture_attention(model):
    """Collect eager attention weights from the language model only.

    Yields a dict that fills during the forward. Hooks are always removed, so a
    failed pass cannot leave the model instrumented for the next one.
    """
    modules = language_attention_modules(model)
    captured: dict[int, torch.Tensor] = {}

    def make_hook(index: int):
        def hook(_module, _inputs, output):
            # eager attention returns (attn_output, attn_weights); sdpa returns
            # None in the second slot, which is the loud symptom of the vision
            # tower's implementation having been applied to the language model.
            if isinstance(output, tuple) and len(output) > 1 and torch.is_tensor(output[1]):
                captured[index] = output[1]

        return hook

    handles = [module.register_forward_hook(make_hook(i)) for i, (_n, module) in enumerate(modules)]
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


def stack_captured(captured: dict[int, torch.Tensor], expected_layers: int) -> torch.Tensor:
    """``{layer: [B,H,S,S]}`` -> ``[L,H,S,S]`` fp32, batch row 0.

    Cast to fp32 here and not later: the weights come back bf16, where a
    softmax row sums to 1.0014 rather than 1.0, and the Chefer recurrence
    multiplies those errors layer over layer.
    """
    if len(captured) != expected_layers:
        raise RuntimeError(
            f"captured {len(captured)} attention layers but the model has "
            f"{expected_layers}; a silently partial capture would produce a "
            "rollout over the wrong depth"
        )
    ordered = [captured[index] for index in sorted(captured)]
    return rollout.stack_layers(ordered, batch_index=0).to(torch.float32)


def locate_visual_tokens(
    input_ids: torch.Tensor,
    token_id: int,
    *,
    expected_count: int,
    source: str,
    grid: projection.GridSpec | None = None,
    require_contiguous: bool = True,
) -> VisualSpan:
    """Find the visual-token block and refuse anything unexpected.

    Verified on the host: MedGemma puts exactly 256 contiguous image tokens
    (id 262144) at positions [5, 260] of a 282-token sequence. Every property
    asserted here held; they are asserted anyway, because each one failing
    silently produces a map of the wrong region rather than an error.
    """
    if not torch.is_tensor(input_ids):
        raise TypeError("input_ids must be a torch.Tensor")
    ids = input_ids[0] if input_ids.ndim == 2 else input_ids
    if ids.ndim != 1:
        raise ValueError(f"input_ids must be [S] or [1, S], got {tuple(input_ids.shape)}")

    positions = (ids == int(token_id)).nonzero(as_tuple=False).flatten()
    if positions.numel() == 0:
        raise RuntimeError(
            f"no visual token (id {token_id}) in the sequence; the prompt was built "
            "without an image, so there is nothing to attribute to"
        )
    if positions.numel() != expected_count:
        raise RuntimeError(
            f"expected {expected_count} visual tokens, found {positions.numel()}; the "
            "processor's image_seq_length and the configured grid disagree"
        )
    if require_contiguous and positions.numel() > 1:
        gaps = positions.diff()
        if not bool((gaps == 1).all()):
            raise RuntimeError(
                "visual tokens are not contiguous; a grid reshape over a split span "
                "would scramble the map"
            )
    if grid is not None and grid.num_tokens != expected_count:
        raise ValueError(
            f"grid declares {grid.num_tokens} cells but {expected_count} visual tokens "
            "were requested"
        )
    return VisualSpan(
        start=int(positions[0]),
        end=int(positions[-1]) + 1,
        source=source,
        grid=grid,
    )


def per_token_nll(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Negative log-likelihood of each supervised token, shifted by one.

    ``logits[t]`` predicts ``labels[t + 1]``, so the value returned at index
    ``t`` is the NLL of the token AT position ``t``, which is what a reader
    means by "how surprised was the model by this word". Positions with label
    ``-100`` come back as ``nan`` and must be dropped, never averaged in.
    """
    if logits.ndim == 3:
        logits = logits[0]
    if labels.ndim == 2:
        labels = labels[0]
    if logits.ndim != 2 or labels.ndim != 1:
        raise ValueError("expected logits [S, V] and labels [S]")
    if logits.shape[0] != labels.shape[0]:
        raise ValueError(
            f"logits cover {logits.shape[0]} positions but labels cover {labels.shape[0]}"
        )

    values = torch.full((labels.shape[0],), float("nan"), dtype=torch.float32)
    predicted = logits[:-1].to(torch.float32)
    targets = labels[1:]
    valid = targets >= 0
    if bool(valid.any()):
        log_probs = torch.log_softmax(predicted[valid], dim=-1)
        picked = log_probs.gather(1, targets[valid, None]).squeeze(1)
        index = torch.arange(1, labels.shape[0], device=labels.device)[valid]
        values[index.cpu()] = (-picked).detach().float().cpu()
    return values


def gradient_weighted_layers(
    score: torch.Tensor,
    captured: dict[int, torch.Tensor],
    *,
    retain_graph: bool = True,
    allow_fallback: bool = False,
) -> tuple[torch.Tensor | None, str | None]:
    """``d score / d attention`` for every layer, or an explicit fallback.

    Returns ``(gradients, fallback_reason)``. ``gradients`` is ``None`` only
    when the gradient could not be taken AND ``allow_fallback`` was set, in
    which case the reason is a string that the caller MUST record -- a
    gradient-free rollout is a weaker measurement and may never be reported as
    if it were the full one.

    ``allow_fallback`` defaults to ``False`` on purpose. Measured peak here is
    9.39 GiB of 15.48, so an OOM means something changed and is worth a human
    looking, not an automatic downgrade.
    """
    tensors = [captured[index] for index in sorted(captured)]
    try:
        grads = torch.autograd.grad(
            outputs=score,
            inputs=tensors,
            retain_graph=retain_graph,
            allow_unused=False,
        )
    except torch.OutOfMemoryError as exc:  # pragma: no cover - GPU only
        if not allow_fallback:
            raise
        torch.cuda.empty_cache()
        return None, f"OutOfMemoryError during autograd.grad: {str(exc)[:200]}"
    except RuntimeError as exc:  # pragma: no cover - GPU only
        if not allow_fallback:
            raise
        return None, f"RuntimeError during autograd.grad: {str(exc)[:200]}"
    # ``grads`` already mirror the captured tensors' [B, H, S, S]; wrapping them
    # again would make them 5-D.
    return rollout.stack_layers(list(grads), batch_index=0).to(torch.float32), None


def attribute_visual_tokens(
    attention: torch.Tensor,
    visual_span: VisualSpan,
    query_positions: Sequence[int],
    *,
    gradients: torch.Tensor | None = None,
    method: str = rollout.METHOD_CHEFER,
    normalize: bool = True,
) -> tuple[torch.Tensor, rollout.RolloutTrace]:
    """Roll attention from some generated tokens back to the visual block."""
    fused = rollout.fuse_heads(attention, gradients)
    matrix, trace = rollout.rollout(
        fused,
        method=method,
        gradient_weighted=gradients is not None,
    )
    values = rollout.span_attribution(
        matrix,
        query_positions,
        visual_span.positions,
        normalize=normalize,
    )
    return values, trace


@dataclass(frozen=True)
class AblationResult:
    """A paired ablation, with enough statistics to say whether it means anything."""

    condition: str
    num_studies: int
    mean_delta: float
    ci_low: float
    ci_high: float
    fraction_worse: float
    established: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "condition": self.condition,
            "num_studies": self.num_studies,
            "mean_delta": self.mean_delta,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "fraction_worse": self.fraction_worse,
            "established": self.established,
        }


def score_ablation(
    baseline_per_study: Sequence[float],
    ablated_per_study: Sequence[float],
    *,
    condition: str,
    min_mean_increase: float = 0.05,
    resamples: int = 20000,
    seed: int = 16,
) -> AblationResult:
    """Paired per-study NLL deltas, with a bootstrap CI over STUDIES.

    ``established`` requires BOTH that the mean increase clears
    ``min_mean_increase`` AND that the 95% CI excludes zero. A threshold on a
    point estimate alone is not enough, and this is not a hypothetical: on 12
    real studies the zero-ablation returned +0.0535, which clears 0.05 and has
    a CI of [-0.0283, +0.1314]. It would have "passed" a threshold-only gate on
    evidence indistinguishable from no effect at all.

    Resampling is over studies, not tokens: tokens within one report are not
    independent, so a token-level interval would be far too narrow.
    """
    import random
    import statistics

    base = [float(v) for v in baseline_per_study]
    ablated = [float(v) for v in ablated_per_study]
    if len(base) != len(ablated) or len(base) < 2:
        raise ValueError(
            f"need at least 2 paired studies, got {len(base)} and {len(ablated)}"
        )
    deltas = [a - b for a, b in zip(ablated, base, strict=True)]
    mean_delta = statistics.mean(deltas)

    rng = random.Random(seed)
    count = len(deltas)
    means = sorted(
        statistics.mean(rng.choices(deltas, k=count)) for _ in range(int(resamples))
    )
    low = means[int(0.025 * len(means))]
    high = means[int(0.975 * len(means))]

    return AblationResult(
        condition=condition,
        num_studies=count,
        mean_delta=mean_delta,
        ci_low=low,
        ci_high=high,
        fraction_worse=sum(1 for d in deltas if d > 0) / count,
        established=bool(mean_delta >= min_mean_increase and low > 0.0),
    )


def assert_visual_tokens_matter(result: AblationResult) -> AblationResult:
    """STOP unless removing the image measurably and reliably hurts.

    If it does not, one of two things is true and both need a human: the visual
    span is in the wrong place, or the model is not using the image. Either way
    every map downstream would be meaningless, so this raises rather than warns.

    Note what "measurably" now means. A mean increase that clears the threshold
    while its CI crosses zero is NOT a pass -- see :func:`score_ablation`.
    """
    if not isinstance(result, AblationResult):
        raise TypeError("result must come from score_ablation")
    if result.established:
        return result
    raise SoftTokenAblationFailed(
        f"ablation {result.condition!r} is not established over "
        f"{result.num_studies} studies: mean NLL change {result.mean_delta:+.4f}, "
        f"95% CI [{result.ci_low:+.4f}, {result.ci_high:+.4f}], "
        f"{result.fraction_worse:.0%} of studies worse. Either the visual span "
        "is located wrongly, or the model is not using the image. Both "
        "invalidate every attribution map from this configuration -- stop and "
        "investigate"
    )


# ---------------------------------------------------------------------------
# The second gate: weight randomization (Adebayo et al. 2018)
# ---------------------------------------------------------------------------
#
# The ablation gate asks whether the model uses the image. This one asks a
# different and equally necessary question: whether the MAP depends on the
# model's learned weights at all. Adebayo et al., "Sanity Checks for Saliency
# Maps", showed that several popular methods produce nearly the same picture
# after the network is randomised -- which means they are reading the input and
# the architecture, not what the model learned.
#
# Cascading randomisation re-initialises the language layers from the last
# backwards, recomputing the attribution at each step. A method that reflects
# the model must lose its correlation with the original map. One that does not
# is an edge detector with extra steps.


class RandomizationSanityFailed(RuntimeError):
    """The attribution survived randomising the model. A STOP condition.

    Its own type, like :class:`SoftTokenAblationFailed`, because it invalidates
    every map from this configuration rather than degrading them.
    """


@dataclass(frozen=True)
class RandomizationResult:
    """Rank correlation between the original map and progressively broken ones."""

    steps: tuple[int, ...]
    correlations: tuple[float, ...]
    final_correlation: float
    max_correlation: float
    degrades: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "steps": list(self.steps),
            "correlations": list(self.correlations),
            "final_correlation": self.final_correlation,
            "max_correlation_threshold": self.max_correlation,
            "degrades": self.degrades,
        }


def spearman_correlation(first: torch.Tensor, second: torch.Tensor) -> float:
    """Rank correlation of two flat maps, ties averaged.

    Rank rather than Pearson because the question is whether the map still
    RANKS regions the same way; a monotone rescale of an unchanged map is not a
    degradation. Ties are averaged so a map with large flat regions -- common
    after randomisation -- does not get an arbitrary ordering imposed on it.
    """
    a = first.detach().reshape(-1).to(torch.float64)
    b = second.detach().reshape(-1).to(torch.float64)
    if a.numel() != b.numel():
        raise ValueError(f"maps differ in size: {a.numel()} vs {b.numel()}")
    if a.numel() < 2:
        raise ValueError("need at least two values to correlate")

    def ranks(values: torch.Tensor) -> torch.Tensor:
        order = torch.argsort(values)
        result = torch.empty_like(values)
        result[order] = torch.arange(values.numel(), dtype=values.dtype)
        # average ties, or a constant map gets a spurious ordering
        unique, inverse, counts = torch.unique(
            values, return_inverse=True, return_counts=True
        )
        if int(counts.max()) > 1:
            sums = torch.zeros(unique.numel(), dtype=values.dtype)
            sums.index_add_(0, inverse, result)
            result = (sums / counts.to(values.dtype))[inverse]
        return result

    ra, rb = ranks(a), ranks(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denominator = torch.sqrt((ra * ra).sum() * (rb * rb).sum())
    if float(denominator) == 0.0:
        # One side is constant: there is no ordering to agree with. Report 0,
        # which is the honest "no relationship", not 1.
        return 0.0
    return float((ra * rb).sum() / denominator)


def randomize_layers(model, layer_indices: Sequence[int], *, seed: int = 16):
    """Re-initialise the named language layers; returns a restore callable.

    Parameters are copied to CPU before being overwritten and copied back by
    the returned callable, so a caller that runs this inside ``try/finally``
    always gets the trained model back. Re-initialisation uses the model's own
    ``initializer_range`` rather than an arbitrary scale, so the randomised
    network stays in the regime the architecture expects.
    """
    modules = language_attention_modules(model)
    layers = sorted({int(i) for i in layer_indices})
    if not layers:
        raise ValueError("layer_indices must not be empty")
    if min(layers) < 0 or max(layers) >= len(modules):
        raise IndexError(
            f"layer index outside [0, {len(modules)}): {layers[0]}..{layers[-1]}"
        )

    text_config = getattr(getattr(model, "config", None), "text_config", None)
    std = float(getattr(text_config, "initializer_range", 0.02) or 0.02)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    saved: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
    for index in layers:
        _name, module = modules[index]
        for parameter in module.parameters():
            saved.append((parameter, parameter.detach().to("cpu", copy=True)))
            # Built on CPU with an explicit generator, so a rerun randomises
            # identically and a correlation can be compared across runs.
            replacement = torch.empty(
                parameter.shape, dtype=torch.float32, device="cpu"
            ).normal_(0.0, std, generator=generator)
            with torch.no_grad():
                parameter.copy_(replacement.to(parameter.dtype))

    def restore() -> None:
        with torch.no_grad():
            for parameter, original in saved:
                parameter.copy_(original.to(parameter.device, parameter.dtype))

    return restore


def cascading_randomization(
    attribute,
    num_layers: int,
    *,
    steps: Sequence[int] | None = None,
    max_correlation: float = 0.5,
) -> RandomizationResult:
    """Randomise the last k layers for increasing k, correlating each map.

    ``attribute(layer_indices)`` must return the attribution map produced with
    those layer indices randomised; ``attribute(())`` returns the original. The
    caller owns loading and restoring, which keeps this function free of any
    model type and therefore testable on CPU.

    Layers are taken from the LAST backwards, which is Adebayo's cascading
    order: the last layers are closest to the output and the first to matter.
    """
    if num_layers < 1:
        raise ValueError("num_layers must be positive")
    if steps is None:
        candidates = [1, 2, 4, 8, 16, num_layers]
        steps = sorted({min(int(k), num_layers) for k in candidates})
    steps = tuple(int(k) for k in steps)
    if any(k < 1 or k > num_layers for k in steps):
        raise ValueError(f"every step must be in [1, {num_layers}]")

    original = attribute(())
    correlations = []
    for k in steps:
        indices = tuple(range(num_layers - k, num_layers))
        correlations.append(spearman_correlation(original, attribute(indices)))

    final = correlations[-1]
    return RandomizationResult(
        steps=steps,
        correlations=tuple(correlations),
        final_correlation=final,
        max_correlation=float(max_correlation),
        degrades=bool(abs(final) < float(max_correlation)),
    )


def assert_randomization_degrades(result: RandomizationResult) -> RandomizationResult:
    """STOP unless randomising the model actually changes the map.

    A map that survives randomisation is not explaining this model. It may
    still be a picture of the input, which is exactly the failure Adebayo et
    al. found in methods that looked convincing.
    """
    if not isinstance(result, RandomizationResult):
        raise TypeError("result must come from cascading_randomization")
    if result.degrades:
        return result
    raise RandomizationSanityFailed(
        f"the attribution survived randomising every language layer: rank "
        f"correlation with the original is {result.final_correlation:+.4f} "
        f"(need |rho| < {result.max_correlation}). The map does not depend on "
        f"what the model learned, so it does not explain this model -- stop "
        f"and investigate. Correlations by step "
        f"{dict(zip(result.steps, [round(c, 4) for c in result.correlations], strict=True))}"
    )


# ---------------------------------------------------------------------------
# Stage 2 of the pipeline: Q-Former cross-attention. INTERFACE ONLY.
# ---------------------------------------------------------------------------


class QFormerCrossAttentionUnavailable(NotImplementedError):
    """The Q-Former stage is declared but deliberately not implemented here."""


def qformer_cross_attention(*_args, **_kwargs):
    """Would map a soft token to image regions. Not implemented in this branch.

    Kept as a named entry point so callers can be written against it, and so the
    reason is recorded where someone will look for it rather than in a commit
    message.

    The 32 soft tokens are ``query_output.last_hidden_state``: each has already
    cross-attended over all 246 visual tokens, and those cross-attention weights
    are NOT in the cached Stage-2 record, which holds only ``[32, 768]``.
    Producing a region map therefore means re-running Stage 1 with its own
    hooks -- a separate piece of work, not a variation on this one.
    """
    raise QFormerCrossAttentionUnavailable(
        "Q-Former cross-attention capture is not implemented in this branch. "
        "Rolling to soft tokens says WHICH soft token a sentence used, not which "
        "image region; the weights that would relate the two are not in the "
        "Stage-2 record and need a Stage-1 re-run"
    )


def warn_if_soft_token_source(source: str) -> None:
    """A soft-token attribution is not a heatmap. Say so, once, at the source."""
    if source == SOURCE_QFORMER_SOFT_TOKEN:
        warnings.warn(
            "attributing to Q-Former soft tokens: this identifies which soft "
            "token a sentence used, NOT which image region. Do not render it "
            "over an image.",
            RuntimeWarning,
            stacklevel=2,
        )


# ---------------------------------------------------------------------------
# Loading and the teacher-forced pass. transformers is imported LAZILY, inside
# these functions, so the module above stays importable -- and testable -- on a
# box with neither transformers nor a GPU.
# ---------------------------------------------------------------------------

#: Measured, not guessed. eager for the language model is what makes the
#: attention weights observable at all; sdpa for the vision tower is what stops
#: SigLIP retaining 27 GiB of 4096x4096 softmax for a backward nothing reads.
ATTN_IMPLEMENTATION = {"text_config": "eager", "vision_config": "sdpa"}


def load_medgemma_for_explanation(
    model_id: str = MEDGEMMA_MODEL_ID,
    *,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    """Load MedGemma frozen, in bf16, instrumented for explanation.

    bf16 rather than NF4 on purpose. The 4B model is 8.02 GiB of weights and the
    whole capture peaks at 9.39 of 15.48 GiB, so quantising buys nothing and
    would reintroduce the "gradient through NF4" question this configuration
    avoids outright.

    Every parameter is frozen. That is not a safety flourish: ``.backward()``
    over ~4B trainable parameters allocates about 8 GiB of gradients that
    nothing here reads, and it is what turned a comfortable pass into an OOM.
    """
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=dtype,
        attn_implementation=dict(ATTN_IMPLEMENTATION),
    ).to(torch.device(device))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    disable_gradient_checkpointing(model)
    return processor, model


def build_visual_inputs(
    model,
    batch: dict,
    visual_span: VisualSpan,
    *,
    visual_features: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the vision tower under ``no_grad`` and scatter its output into embeds.

    The vision tower is deliberately kept OUT of the autograd graph: the rollout
    stops at the visual tokens in the language sequence and never reaches into
    SigLIP, so retaining its activations is pure cost.

    ``visual_features`` overrides what the tower produced. That is what makes the
    MISMATCHED-image control possible: same prompt, same target, same positions,
    another study's picture. It is a sharper question than zeroing -- zeros are
    out of distribution and a model may simply be confused by them, whereas a
    real image of the wrong patient asks precisely "is THIS image being used?".

    The scatter is a SUBSTITUTION at the visual positions -- the same operation
    ``training/medgemma/soft_tokens.py`` performs for Q-Former tokens, and the
    same failure mode applies: get the per-row indexing wrong and every study is
    described using another study's image, with no error and a loss that still
    looks fine. Hence the explicit shape check.
    """
    if visual_features is None:
        with torch.no_grad():
            visual_features = model.get_image_features(pixel_values=batch["pixel_values"])
    flat = visual_features.reshape(-1, visual_features.shape[-1])
    if flat.shape[0] != visual_span.length:
        raise RuntimeError(
            f"vision tower produced {flat.shape[0]} tokens but the located span holds "
            f"{visual_span.length}; refusing to scatter a mismatched block"
        )

    embeds = model.get_input_embeddings()(batch["input_ids"]).clone()
    if embeds.shape[0] != 1:
        raise NotImplementedError(
            "explanation capture is single-study by design; batching would make the "
            "per-row scatter another place to silently cross studies"
        )
    embeds[0, visual_span.start : visual_span.end, :] = flat.to(embeds.dtype)
    # requires_grad on the INPUT is what keeps the graph alive once every weight
    # is frozen; without it there is no graph and autograd.grad has nothing.
    return embeds.detach().requires_grad_(True)


def teacher_forced_forward(
    model,
    batch: dict,
    visual_span: VisualSpan,
    *,
    labels: torch.Tensor | None = None,
    ablate_visual: bool = False,
    visual_features: torch.Tensor | None = None,
    logits_to_keep: int = 0,
):
    """One teacher-forced pass with the KV cache off and attention captured.

    Two controls, deliberately different questions. ``ablate_visual`` zeroes the
    scattered embeddings -- "does anything visual matter?" -- and
    ``visual_features`` substitutes another study's features -- "does THIS
    image matter?". The second is the stronger test, because a zero vector is
    not "no information" but an out-of-distribution point, and a model may
    perform worse OR better when handed one.

    Returns ``(outputs, captured, embeds)``. ``captured`` is still attached to
    the graph, so the caller may take gradients before letting it go.
    """
    embeds = build_visual_inputs(model, batch, visual_span, visual_features=visual_features)
    if ablate_visual:
        with torch.no_grad():
            embeds[0, visual_span.start : visual_span.end, :] = 0.0

    kwargs = {
        "inputs_embeds": embeds,
        "attention_mask": batch.get("attention_mask"),
        # Teacher forcing with no cache: every position is computed in one pass,
        # from the real previous tokens, so the attention captured is the
        # attention that produced the logits being explained.
        "use_cache": False,
        "return_dict": True,
    }
    if labels is not None:
        kwargs["labels"] = labels
    if logits_to_keep:
        kwargs["logits_to_keep"] = int(logits_to_keep)

    with capture_attention(model) as captured:
        outputs = model(**{k: v for k, v in kwargs.items() if v is not None})
        snapshot = dict(captured)
    return outputs, snapshot, embeds
