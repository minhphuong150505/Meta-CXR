"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from model.lavis.common.registry import registry
from model.lavis.tasks.base_task import BaseTask
from model.lavis.datasets.data_utils import move_to_cuda
from model.lavis.common.dist_utils import is_dist_avail_and_initialized
import logging
import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


@registry.register_task("image_text_pretrain_eval")
class ImageTextPretrainTask(BaseTask):
    """Stage-1 validation, scored with the META-CXR paper's protocol.

    Every scored epoch reports the paper's three-class numbers
    (``training/evaluation/classification_metrics.py``): argmax over
    Negative / Positive / Uncertain, per-finding sklearn-weighted precision /
    recall / F1 averaged over the 14 findings, the five-finding mean F1 of
    Tables 5 and 7, and one-vs-rest AUROC per class. There are no binary /
    positive-only metrics and no uncertain policy (removed 2026-09-29, D-023).

    ``run.selection_metric`` may be ``loss`` or any of
    ``classification_metrics.AGGREGATE_METRICS``. Set
    ``run.save_predictions: true`` to also write a prediction ``.npz`` for
    ``scripts/evaluate_stage1.py``.
    """

    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = getattr(cfg, "run_cfg", cfg)
        from pretraining.retired_keys import reject_retired_run_keys

        reject_retired_run_keys(self.cfg)
        self.eval_split = "validation"
        self.eval_epoch = "unknown"

    @classmethod
    def setup_task(cls, cfg=None, **kwargs):
        """Keep the run config: Stage-1 evaluation has configurable metrics."""
        return cls(cfg=cfg)

    def set_evaluation_context(self, split_name, epoch):
        """Give prediction artifacts a stable split/epoch-specific name."""
        self.eval_split = str(split_name)
        self.eval_epoch = str(epoch)

    def evaluation(self, model, data_loader, cuda_enabled=True):
        loss_sums = {}
        example_count = 0

        run_cfg = self.cfg
        save_predictions = bool(
            run_cfg.get("save_predictions", False) if run_cfg is not None else False
        )
        collected_logits = []
        collected_labels = []
        collected_keys = []
        collected_num_views = []

        for batch in data_loader:
            if cuda_enabled:
                batch = move_to_cuda(batch)
            output = model(batch)

            labels = batch.get("classification_labels")
            batch_size = labels.shape[0] if labels is not None else len(batch["text_output"])
            example_count += batch_size
            for name, value in output.items():
                if "loss" in name and value is not None:
                    loss_sums[name] = loss_sums.get(name, 0.0) + float(value) * batch_size

            logits = output.get("classification_logits")
            if logits is None or labels is None:
                continue
            if logits.ndim != 3 or labels.shape != logits.shape[:2] or logits.shape[-1] != 3:
                raise ValueError(
                    "classification logits/labels must be [B, abnormalities, 3] and "
                    f"[B, abnormalities], got {tuple(logits.shape)} and {tuple(labels.shape)}"
                )

            sample_mask = output.get("classification_mask")
            if sample_mask is None:
                sample_mask = batch.get("classification_mask")
            if sample_mask is None:
                sample_mask = torch.ones(labels.shape[0], dtype=torch.bool, device=labels.device)
            sample_mask = torch.as_tensor(
                sample_mask, dtype=torch.bool, device=labels.device
            ).reshape(-1)

            # A study with no CheXpert information carries -1 in the exported
            # labels, so every metric skips exactly those cells.
            valid = sample_mask[:, None] & (labels >= 0) & (labels < 3)
            masked_labels = labels.clone()
            masked_labels[~valid] = -1
            collected_logits.append(logits.detach().float().cpu())
            collected_labels.append(masked_labels.detach().cpu())
            collected_keys.extend(_sample_keys(batch, labels.shape[0]))
            collected_num_views.append(_num_views(batch, labels.shape[0]))

        device = next(model.parameters()).device
        loss_names = sorted(loss_sums)
        totals = torch.tensor(
            [loss_sums[name] for name in loss_names] + [example_count],
            dtype=torch.float64, device=device,
        )
        if is_dist_avail_and_initialized():
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)

        denom = max(totals[-1].item(), 1.0)
        stats = {
            name: totals[index].item() / denom
            for index, name in enumerate(loss_names)
        }

        if collected_logits:
            if is_dist_avail_and_initialized() and dist.get_world_size() > 1:
                raise RuntimeError("Stage-1 evaluation supports single-GPU runs only")
            from training.evaluation.classification_metrics import (
                evaluate_classification,
            )

            predictions = self._build_predictions(
                collected_logits, collected_labels, collected_keys,
                collected_num_views,
            )
            predictions.metadata["split"] = self.eval_split
            report = evaluate_classification(predictions)
            stats.update({k: float(v) for k, v in report.aggregates.items()})

            if save_predictions:
                self._save_predictions(predictions)

        return stats

    @staticmethod
    def _build_predictions(logits_chunks, label_chunks, keys, num_view_chunks=None):
        from model.lavis.models.blip2_models.blip2_qformer import chexpert_cols
        from training.evaluation.schemas import (
            ClassificationPredictions,
            build_sample_keys,
        )

        logits = torch.cat(logits_chunks, dim=0).numpy()
        labels = torch.cat(label_chunks, dim=0).numpy()
        probabilities = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()

        names = tuple(chexpert_cols)
        if len(names) != labels.shape[1]:
            names = tuple(f"abnormality_{i}" for i in range(labels.shape[1]))
            logger.warning(
                "chexpert_cols has %d entries but logits have %d columns; "
                "falling back to positional names",
                len(chexpert_cols),
                labels.shape[1],
            )

        return ClassificationPredictions(
            labels=labels,
            probabilities=probabilities,
            logits=logits,
            pathology_names=names,
            sample_keys=build_sample_keys(list(keys[: labels.shape[0]])),
            num_views=_concat_num_views(num_view_chunks, labels.shape[0]),
        )

    def _save_predictions(self, predictions):
        """Write a prediction .npz for offline re-evaluation.

        Best-effort: a failure here must not lose a completed validation pass,
        so it is logged rather than raised.
        """
        try:
            if is_dist_avail_and_initialized() and dist.get_world_size() > 1:
                logger.warning(
                    "save_predictions collects rank-local batches only; with "
                    "world_size>1 the written file covers this rank's shard"
                )

            registry_output = registry.get_path("result_dir") or "."
            safe_split = self.eval_split.replace("/", "_")
            safe_epoch = self.eval_epoch.replace("/", "_")
            path = (
                f"{registry_output}/{safe_split}_predictions_epoch_{safe_epoch}.npz"
            )
            predictions.save(path)
            logger.info("saved %d predictions to %s", predictions.num_samples, path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not save predictions (metrics unaffected): %s", exc)


def _num_views(batch, batch_size):
    """Images the model actually saw per study: the anchor plus real auxiliaries.

    ``None`` when the batch carries no ``aux_mask`` (multi-view off), so the
    exported file then has no ``num_views`` rather than a fabricated all-ones.
    """
    aux_mask = batch.get("aux_mask")
    if aux_mask is None:
        return None
    return (1 + aux_mask.detach().to(torch.long).sum(dim=1)).cpu()[:batch_size]


def _concat_num_views(chunks, total):
    if not chunks or any(chunk is None for chunk in chunks):
        return None
    counts = torch.cat(chunks).numpy()
    return counts if counts.shape[0] == total else None


def _sample_keys(batch, batch_size):
    """Opaque per-study keys, preferring an identifier the batch already carries."""
    for field in ("study_id", "image_id", "dicom_id"):
        values = batch.get(field)
        if values is None:
            continue
        if torch.is_tensor(values):
            values = values.detach().cpu().tolist()
        return [str(v) for v in list(values)[:batch_size]]
    return [""] * batch_size
