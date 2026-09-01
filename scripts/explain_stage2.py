#!/usr/bin/env python3
"""Explain a Stage-2 report: which sentence leaned on which part of the image.

Writes one JSONL line per study plus one NPZ of attribution maps per study, and
a run-level summary. The maps are stored at the model's NATIVE grid resolution
(16x16 for MedGemma) and never as an upsampled PNG: a rendered overlay is a
patient-derived image, and this command's job is to produce evidence, not
pictures.

The gate runs FIRST and can abort the whole run. If substituting another
study's image does not measurably degrade the report, every map that follows
would be meaningless -- either the visual span is located wrongly or the model
is not using the image -- so the run stops instead of producing a directory of
plausible-looking artifacts. ``--skip-ablation-gate`` exists, warns loudly, and
records the skip in the summary.

Privacy. Output is PhysioNet credentialed derivative data:
  * the destination is refused unless it is outside the repository or Git
    confirms it is ignored (the same check ``evaluate_explanation.py`` makes);
  * filenames carry a sequential index, never an identifier;
  * the JSONL carries a ``sample_key`` fingerprint rather than the real ids.
    ``--write-key-map`` emits the join separately and is the single most
    sensitive artifact this command can produce.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from hashlib import blake2b
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.evaluate_explanation import _assert_private_output_location  # noqa: E402
from training.explainability import attention_capture as capture  # noqa: E402
from training.explainability import projection, rollout  # noqa: E402
from training.explainability.sentence_attribution import (  # noqa: E402
    DEFAULT_LABELER_NAME,
    LABELERS,
    attribute_sentences,
    dataset_parse_coverage,
    lexicon_metadata,
)

LOGGER = logging.getLogger("explain_stage2")

PROMPT = "Describe the chest radiograph."
SCHEMA_VERSION = 1

#: The CLI says ``val``; MIMIC-CXR's official split column says ``validate``.
#: Filtering on the CLI name alone silently matched nothing, which surfaced as
#: "no study matches the selection" and would have been an empty run had the
#: guard not been there. Accept both, and never guess.
MANIFEST_SPLIT_ALIASES = {"val": ("val", "validate"), "test": ("test",)}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", required=True, type=Path, help="split CSV")
    parser.add_argument("--image-root", required=True, type=Path,
                        help="directory that directly contains files/")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", choices=capture.ALLOWED_SPLITS, default="test")
    parser.add_argument("--limit", type=int, default=20,
                        help="studies to explain; 0 means the whole split")
    parser.add_argument("--ablation-studies", type=int, default=12,
                        help="studies used for the gate before any map is written")
    parser.add_argument("--skip-ablation-gate", action="store_true",
                        help="run without the gate. Recorded in the summary.")
    parser.add_argument("--skip-randomization-gate", action="store_true",
                        help="run without the Adebayo sanity check. Recorded in "
                             "the summary.")
    # v1 stays the default so the n=1513 val run remains reproducible. v2 adds
    # findings outside the CheXpert 14, which raises coverage without adding
    # anything to verify them against.
    parser.add_argument("--labeler", choices=sorted(LABELERS),
                        default=DEFAULT_LABELER_NAME,
                        help="lexicon_v2 is the default; pass lexicon_v1 to reproduce "
                             "runs recorded before 2026-08-30")
    parser.add_argument("--model-id", default=capture.MEDGEMMA_MODEL_ID)
    parser.add_argument(
        "--adapter",
        type=Path,
        default=None,
        help=(
            "Stage-2 QLoRA adapter directory, merged into the base weights. "
            "WITHOUT IT THIS EXPLAINS THE UN-FINETUNED MODEL -- a zero-shot "
            "baseline, not this project's Stage 2."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=16)
    # No default bounds. The previous 30-90 silently excluded the longest
    # reports -- val's longest carry 121-138 findings tokens -- so the worst
    # case was never exercised and the cohort was quietly narrowed.
    parser.add_argument("--min-findings-tokens", type=int, default=None)
    parser.add_argument("--max-findings-tokens", type=int, default=None)
    parser.add_argument("--graph-mode", choices=GRAPH_MODES, default=GRAPH_AUTO,
                        help="auto retries a study per-sentence if the shared "
                             "graph OOMs; measured worst case in val is 13.49 "
                             "GiB of 15.48 with the shared graph")
    parser.add_argument("--no-gradient-weight", action="store_true",
                        help="plain rollout, no gradient term. Recorded in every record.")
    parser.add_argument("--write-key-map", action="store_true",
                        help="also write sample_key -> identifiers. Most sensitive output.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def sample_key(value: str) -> str:
    """Stable, non-reversible handle for one study."""
    return blake2b(str(value).encode("utf-8"), digest_size=12).hexdigest()


def select_studies(manifest: Path, split: str, args) -> list:
    import pandas as pd

    frame = pd.read_csv(manifest)
    if "split" in frame.columns:
        present = sorted(frame["split"].dropna().unique().tolist())
        aliases = MANIFEST_SPLIT_ALIASES.get(split, (split,))
        frame = frame[frame["split"].isin(aliases)]
        if frame.empty:
            raise SystemExit(
                f"the manifest holds no rows for split {split!r} (accepted "
                f"{aliases}); its split column contains {present}"
            )
    frame = frame[frame["target_valid"] & frame["ViewPosition"].isin(["PA", "AP"])]
    if args.min_findings_tokens is not None:
        frame = frame[frame["findings_token_count"] >= args.min_findings_tokens]
    if args.max_findings_tokens is not None:
        frame = frame[frame["findings_token_count"] <= args.max_findings_tokens]
    if frame.empty:
        raise SystemExit(
            f"no study in split {split!r} matches ViewPosition in (PA, AP) with "
            f"findings tokens in [{args.min_findings_tokens}, "
            f"{args.max_findings_tokens}]; widen or drop the bounds"
        )
    # +1 so the last study still has a partner for the mismatch control.
    if args.limit and args.limit > 0:
        wanted = min(len(frame), max(args.limit, args.ablation_studies) + 1)
    else:
        wanted = len(frame)  # --limit 0 == the whole split
    return frame.sample(n=wanted, random_state=args.seed).reset_index(drop=True)


def build_batch(processor, model, row, image_root: Path, device):
    import torch
    from PIL import Image

    with Image.open(image_root / row.image_path) as handle:
        image = handle.convert("RGB").copy()
    messages = [
        {"role": "user", "content": [{"type": "image", "image": image},
                                     {"type": "text", "text": PROMPT}]},
        {"role": "assistant",
         "content": [{"type": "text", "text": str(row.findings_clean).strip()}]},
    ]
    encoded = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        return_dict=True, return_tensors="pt",
    )
    return {
        key: (value.to(device, model.dtype) if value.is_floating_point() else value.to(device))
        for key, value in encoded.items()
        if torch.is_tensor(value)
    }


def locate(model, batch):
    return capture.locate_visual_tokens(
        batch["input_ids"],
        model.config.image_token_index,
        expected_count=projection.MEDGEMMA_GRID.num_tokens,
        source=capture.SOURCE_MEDGEMMA_IMAGE,
        grid=projection.MEDGEMMA_GRID,
    )


def target_positions(batch, span) -> list[int]:
    """Positions of the assistant turn, i.e. what the model had to produce."""
    total = int(batch["input_ids"].shape[1])
    start = span.end + 4  # the turn markers the chat template inserts after the image
    if start >= total:
        raise RuntimeError("no target tokens after the visual span; the target was truncated")
    return list(range(start, total))


def mean_token_nll_for(model, batch, span, positions, **kwargs) -> list[float]:
    import torch

    labels = batch["input_ids"].clone()
    labels[:, : positions[0]] = -100
    with torch.no_grad():
        outputs, _captured, _embeds = capture.teacher_forced_forward(
            model, batch, span, **kwargs
        )
    values = capture.per_token_nll(outputs.logits, labels)
    del outputs
    return [float(v) for v in values if v == v]


def run_ablation_gate(model, processor, frame, args, image_root, device) -> dict:
    """Substitute another study's image and require an established degradation."""
    import torch

    baseline, mismatched = [], []
    count = min(args.ablation_studies, len(frame) - 1)
    for index in range(count):
        batch = build_batch(processor, model, frame.iloc[index], image_root, device)
        span = locate(model, batch)
        positions = target_positions(batch, span)
        values = mean_token_nll_for(model, batch, span, positions)
        baseline.append(sum(values) / len(values))

        other = build_batch(processor, model, frame.iloc[index + 1], image_root, device)
        with torch.no_grad():
            features = model.get_image_features(pixel_values=other["pixel_values"])
        values = mean_token_nll_for(
            model, batch, span, positions, visual_features=features
        )
        mismatched.append(sum(values) / len(values))
        del batch, other, features
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        LOGGER.info("gate %d/%d", index + 1, count)

    result = capture.score_ablation(baseline, mismatched, condition="mismatched_image")
    LOGGER.info(
        "ablation %s (n=%d studies): mean %+0.4f CI [%+0.4f, %+0.4f] worse %.0f%% "
        "established=%s",
        result.condition, result.num_studies, result.mean_delta, result.ci_low,
        result.ci_high, 100 * result.fraction_worse, result.established,
    )
    return result


GRAPH_SHARED = "shared"
GRAPH_PER_SENTENCE = "per-sentence"
GRAPH_AUTO = "auto"
GRAPH_MODES = (GRAPH_AUTO, GRAPH_SHARED, GRAPH_PER_SENTENCE)


def _sentence_score(logits, positions, token_indices):
    """Teacher-forced: the logits at t-1 produced the token at t."""
    rows = [positions[i] for i in token_indices]
    return logits[0, [r - 1 for r in rows], :].max(dim=-1).values.sum(), rows


def _attribute_shared(model, batch, span, attributed, positions, outputs, captured, args):
    """One forward for the study, one gradient per sentence, graph retained.

    Cheapest, and what the smoke run used. Measured on the six worst studies in
    val (12-14 sentences): peak 13.49 GiB of 15.48, i.e. 2.0 GiB of headroom.
    It survives val's worst case and does not have much left over, which is why
    the auto mode exists.
    """
    import torch

    attention = capture.stack_captured(
        captured, expected_layers=len(capture.language_attention_modules(model))
    )
    maps, flags = [], []
    for sentence in attributed.sentences:
        if not sentence.token_indices:
            maps.append(torch.zeros(span.length))
            flags.append(False)
            continue
        score, rows = _sentence_score(outputs.logits, positions, sentence.token_indices)
        gradients = None
        if not args.no_gradient_weight:
            gradients, reason = capture.gradient_weighted_layers(
                score, captured, retain_graph=True
            )
            if reason:
                LOGGER.warning("gradient fallback: %s", reason)
        values, trace = capture.attribute_visual_tokens(
            attention, span, rows, gradients=gradients
        )
        maps.append(values)
        flags.append(bool(trace.gradient_weighted))
        del gradients
    del attention
    return maps, flags


def _attribute_per_sentence(model, batch, span, attributed, positions, args):
    """One forward AND one backward per sentence, graph freed in between.

    Costs a forward per sentence instead of per study, and in exchange the peak
    does not grow with the number of sentences. This is the mode to use when
    the shared graph runs out of room; it is slower, not worse.
    """
    import torch

    maps, flags = [], []
    for sentence in attributed.sentences:
        if not sentence.token_indices:
            maps.append(torch.zeros(span.length))
            flags.append(False)
            continue
        outputs, captured, _embeds = capture.teacher_forced_forward(model, batch, span)
        attention = capture.stack_captured(
            captured, expected_layers=len(capture.language_attention_modules(model))
        )
        score, rows = _sentence_score(outputs.logits, positions, sentence.token_indices)
        gradients = None
        if not args.no_gradient_weight:
            gradients, reason = capture.gradient_weighted_layers(
                score, captured, retain_graph=False
            )
            if reason:
                LOGGER.warning("gradient fallback: %s", reason)
        values, trace = capture.attribute_visual_tokens(
            attention, span, rows, gradients=gradients
        )
        maps.append(values)
        flags.append(bool(trace.gradient_weighted))
        del outputs, captured, attention, gradients
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return maps, flags


def run_randomization_gate(model, processor, frame, args, image_root, device) -> dict:
    """Adebayo's cascading randomization, on one real study.

    Asks a different question from the ablation gate: not "does the model use
    the image" but "does the MAP depend on what the model learned". A map that
    survives randomising the network is a function of the input and the
    architecture, which is the failure Adebayo et al. 2018 found in several
    methods that looked convincing.

    Weights are restored in ``finally`` at every step, so a failure here leaves
    the model trained.
    """
    import torch

    batch = build_batch(processor, model, frame.iloc[0], image_root, device)
    span = locate(model, batch)
    positions = target_positions(batch, span)
    num_layers = len(capture.language_attention_modules(model))

    def attribute(layer_indices):
        restore = None
        try:
            if layer_indices:
                restore = capture.randomize_layers(model, layer_indices, seed=args.seed)
            outputs, captured, _embeds = capture.teacher_forced_forward(model, batch, span)
            attention = capture.stack_captured(captured, expected_layers=num_layers)
            score = outputs.logits[0, [p - 1 for p in positions], :].max(dim=-1).values.sum()
            gradients = None
            if not args.no_gradient_weight:
                gradients, _reason = capture.gradient_weighted_layers(
                    score, captured, retain_graph=False
                )
            values, _trace = capture.attribute_visual_tokens(
                attention, span, positions, gradients=gradients
            )
            result = values.detach().cpu().clone()
            del outputs, captured, attention, gradients
            return result
        finally:
            if restore is not None:
                restore()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    result = capture.cascading_randomization(attribute, num_layers)
    LOGGER.info(
        "randomization (n=1 study): rho by step %s -> final %+0.4f, degrades=%s",
        dict(zip(result.steps, [round(c, 4) for c in result.correlations], strict=True)),
        result.final_correlation, result.degrades,
    )
    return result


def explain_study(model, batch, span, study, args) -> tuple[list, dict]:
    """Sentence-level attribution for one study.

    ``--graph-mode auto`` runs the shared-graph path and, on an OOM, redoes the
    study one sentence at a time. An OOM here is a capacity finding about a
    long report, not a defect, and it is recorded per study so a run cannot
    quietly become half one mode and half the other without saying so.
    """
    import torch

    positions = target_positions(batch, span)
    labels = batch["input_ids"].clone()
    labels[:, : positions[0]] = -100

    outputs, captured, _embeds = capture.teacher_forced_forward(model, batch, span)
    nll = capture.per_token_nll(outputs.logits, labels)
    tokenizer = model._explain_tokenizer
    token_texts = [tokenizer.decode([int(t)]) for t in batch["input_ids"][0, positions]]
    attributed = attribute_sentences(
        str(study.findings_clean).strip(),
        token_texts=token_texts,
        token_nll=[float(nll[p]) for p in positions],
        labeler=LABELERS[args.labeler](),
    )

    mode = args.graph_mode
    used = mode
    try:
        if mode == GRAPH_PER_SENTENCE:
            del outputs, captured
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            maps, flags = _attribute_per_sentence(
                model, batch, span, attributed, positions, args
            )
        else:
            used = GRAPH_SHARED
            maps, flags = _attribute_shared(
                model, batch, span, attributed, positions, outputs, captured, args
            )
    except torch.OutOfMemoryError:
        if mode != GRAPH_AUTO:
            raise
        LOGGER.warning(
            "shared graph ran out of memory on a %d-sentence study; redoing it "
            "one sentence at a time", len(attributed.sentences),
        )
        try:
            del outputs, captured
        except NameError:  # pragma: no cover
            pass
        torch.cuda.empty_cache()
        used = GRAPH_PER_SENTENCE
        maps, flags = _attribute_per_sentence(
            model, batch, span, attributed, positions, args
        )

    grid = span.grid
    records = []
    for index, sentence in enumerate(attributed.sentences):
        record = sentence.to_dict()
        record["attribution_index"] = index
        record["gradient_weighted"] = flags[index]
        records.append(record)
    arrays = [m.detach().cpu().reshape(grid.height, grid.width).numpy() for m in maps]

    return arrays, {
        "sentences": records,
        "parse_coverage": attributed.parse_coverage,
        "labeler": attributed.labeler,
        "unparsed_sentences": list(attributed.unparsed),
        "graph_mode": used,
        "study_summary": attributed,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )
    split = capture.assert_split_allowed(args.split)
    output_dir = _assert_private_output_location(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    maps_dir = output_dir / "maps"
    maps_dir.mkdir(exist_ok=True)

    import numpy as np
    import torch

    frame = select_studies(args.manifest, split, args)
    mode = "medgemma_direct_finetuned" if args.adapter else "medgemma_direct_zeroshot"
    if args.adapter is None:
        LOGGER.warning(
            "No --adapter: explaining the base model. These maps describe "
            "%s out of the box, NOT this project's fine-tuned Stage 2.",
            args.model_id,
        )
    processor, model = capture.load_medgemma_for_explanation(
        args.model_id, device=args.device, adapter=args.adapter
    )
    model._explain_tokenizer = getattr(processor, "tokenizer", processor)
    device = next(model.parameters()).device

    gate = None
    if args.skip_ablation_gate:
        LOGGER.warning(
            "ABLATION GATE SKIPPED. Nothing has checked that the model uses the "
            "image; every map below may be meaningless."
        )
    else:
        gate = run_ablation_gate(model, processor, frame, args, args.image_root, device)
        capture.assert_visual_tokens_matter(gate)

    randomization = None
    if args.skip_randomization_gate:
        LOGGER.warning(
            "RANDOMIZATION GATE SKIPPED. Nothing has checked that the maps "
            "depend on the model's learned weights rather than on the input "
            "and the architecture."
        )
    else:
        randomization = run_randomization_gate(
            model, processor, frame, args, args.image_root, device
        )
        capture.assert_randomization_degrades(randomization)

    planned = len(frame) if not args.limit or args.limit <= 0 else min(
        args.limit, len(frame) - 1
    )
    LOGGER.info("explaining n=%d studies from split %r", planned, split)

    jsonl_path = output_dir / f"explanations_{split}.jsonl"
    keymap_path = output_dir / f"keymap_{split}.jsonl"
    studies, written = [], 0
    with jsonl_path.open("w", encoding="utf-8") as handle:
        keymap = keymap_path.open("w", encoding="utf-8") if args.write_key_map else None
        try:
            for index in range(planned):
                study = frame.iloc[index]
                key = sample_key(study.dicom_id)
                batch = build_batch(processor, model, study, args.image_root, device)
                span = locate(model, batch)
                maps, payload = explain_study(model, batch, span, study, args)
                studies.append(payload.pop("study_summary"))

                # Native grid only. An upsampled render is a patient image.
                name = f"study_{index:05d}.npz"
                np.savez_compressed(
                    maps_dir / name,
                    maps=np.stack(maps).astype(np.float32),
                    grid=np.array([span.grid.height, span.grid.width], dtype=np.int32),
                )
                handle.write(json.dumps({
                    "schema_version": SCHEMA_VERSION,
                    "sample_key": key,
                    "split": split,
                    "attribution_map": f"maps/{name}",
                    "attribution_grid": span.grid.to_dict(),
                    "visual_span": span.to_dict(),
                    "rollout_method": rollout.METHOD_CHEFER,
                    **payload,
                }, ensure_ascii=False) + "\n")
                if keymap is not None:
                    keymap.write(json.dumps({
                        "sample_key": key, "dicom_id": str(study.dicom_id),
                        "study_id": str(study.study_id), "subject_id": str(study.subject_id),
                    }) + "\n")
                written += 1
                del batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                LOGGER.info("explained %d/%d studies (n target=%d)",
                            written, planned, planned)
        finally:
            if keymap is not None:
                keymap.close()

    coverage = dataset_parse_coverage(studies)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "split": split,
        "model_id": args.model_id,
        "mode": mode,
        "adapter": str(args.adapter) if args.adapter else None,
        "n_studies": written,
        "studies_written": written,
        "n_ablation_studies": gate.num_studies if gate is not None else 0,
        "n_randomization_studies": 0 if randomization is None else 1,
        "gradient_weighted": not args.no_gradient_weight,
        "graph_mode_requested": args.graph_mode,
        "rollout_method": rollout.METHOD_CHEFER,
        "attention_implementation": dict(capture.ATTN_IMPLEMENTATION),
        **lexicon_metadata(args.labeler),
        "coverage": coverage,
        "ablation_gate": gate.to_dict() if gate is not None else None,
        "ablation_gate_skipped": bool(args.skip_ablation_gate),
        "randomization_gate": randomization.to_dict() if randomization is not None else None,
        "randomization_gate_skipped": bool(args.skip_randomization_gate),
        "peak_vram_bytes": (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"n={written} studies written to {jsonl_path}")
    if gate is not None:
        print(
            f"ablation n={gate.num_studies}: {gate.mean_delta:+.4f} "
            f"[{gate.ci_low:+.4f}, {gate.ci_high:+.4f}], established={gate.established}"
        )
    if randomization is not None:
        curve = ", ".join(
            f"{k}:{c:+.4f}"
            for k, c in zip(randomization.steps, randomization.correlations, strict=True)
        )
        print(f"randomization n=1, rho by layers randomized -> {curve}")
    print(
        f"parse_coverage {coverage['parse_coverage']:.3f} over "
        f"n={coverage['num_sentences']} sentences in n={coverage['num_studies']} "
        f"studies -- quote n beside any number from this run"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
