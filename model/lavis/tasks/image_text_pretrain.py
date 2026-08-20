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
    """Stage-1 validation.

    The confusion-matrix metrics below are kept **bit-identical** to their
    historical behaviour. ``f1_positive_macro`` remains available for old
    runs, while production selects checkpoints with threshold-free
    ``macro_auprc``. The legacy metric's known defect -- a pathology with no positive
    samples contributes 0 to the macro instead of being excluded -- is fixed in
    ``training/evaluation/classification_metrics.py`` and surfaced here under
    separate ``*_defined_only`` keys.

    Set ``run.save_predictions: true`` in the run config to additionally write a
    prediction ``.npz``, which lets the full offline evaluator
    (``scripts/evaluate_stage1.py``) recompute AUROC/AUPRC, calibrate
    thresholds and bootstrap without another GPU pass.
    """

    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = getattr(cfg, "run_cfg", cfg)
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
        confusion = None

        run_cfg = self.cfg
        save_predictions = bool(
            run_cfg.get("save_predictions", False) if run_cfg is not None else False
        )
        selection_metric = str(
            run_cfg.get("selection_metric", "f1_positive_macro")
            if run_cfg is not None
            else "f1_positive_macro"
        )
        probability_metrics = {"macro_auprc", "macro_auroc"}
        collect_predictions = save_predictions or selection_metric in probability_metrics
        collected_logits = []
        collected_labels = []
        collected_keys = []
        # Mention-gate logits, kept alongside the polarity logits so the offline
        # evaluator can form P(present) = sigmoid(mention) * q_positive. A run
        # whose model predates this field simply writes no gate array, and
        # label_framing.presence_scores then refuses 'marginal_presence' rather
        # than silently substituting the conditional score.
        collected_mention = []

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
            if logits.ndim != 3 or labels.shape != logits.shape[:2]:
                raise ValueError(
                    "classification logits/labels must be [B, abnormalities, classes] "
                    f"and [B, abnormalities], got {tuple(logits.shape)} and {tuple(labels.shape)}"
                )

            sample_mask = output.get("classification_mask")
            if sample_mask is None:
                sample_mask = batch.get("classification_mask")
            if sample_mask is None:
                sample_mask = batch.get("has_chexpert_label")
            if sample_mask is None:
                sample_mask = torch.ones(labels.shape[0], dtype=torch.bool, device=labels.device)
            sample_mask = torch.as_tensor(
                sample_mask, dtype=torch.bool, device=labels.device
            ).reshape(-1)

            num_abnormalities, num_classes = logits.shape[1:]
            if confusion is None:
                confusion = torch.zeros(
                    num_abnormalities,
                    num_classes,
                    num_classes,
                    dtype=torch.float64,
                    device=logits.device,
                )
            predictions = logits.argmax(dim=-1)
            valid = sample_mask[:, None] & (labels >= 0) & (labels < num_classes)
            abnormality_idx = torch.arange(num_abnormalities, device=labels.device)[None, :]
            flat_index = (
                abnormality_idx * num_classes * num_classes
                + labels.long() * num_classes
                + predictions.long()
            )
            counts = torch.bincount(
                flat_index[valid], minlength=num_abnormalities * num_classes * num_classes
            )
            confusion += counts.reshape(num_abnormalities, num_classes, num_classes)

            if collect_predictions:
                # Labels are masked to -1 where the sample carries no CheXpert
                # annotation, so the offline evaluator skips exactly the rows
                # this loop skipped. Without it, an unlabelled row would be
                # read back as a true negative.
                masked_labels = labels.clone()
                masked_labels[~valid] = -1
                collected_logits.append(logits.detach().float().cpu())
                collected_labels.append(masked_labels.detach().cpu())
                collected_keys.extend(_sample_keys(batch, labels.shape[0]))
                mention_logits = output.get("mention_logits")
                if mention_logits is None:
                    if not collected_mention and example_count <= batch_size:
                        logger.info(
                            "model emits no mention_logits; the prediction file "
                            "will carry no gate and 'marginal_presence' scoring "
                            "will be unavailable offline"
                        )
                elif mention_logits.shape != labels.shape:
                    raise ValueError(
                        "mention logits must be [B, abnormalities] to line up "
                        f"with the labels, got {tuple(mention_logits.shape)} vs "
                        f"{tuple(labels.shape)}"
                    )
                else:
                    collected_mention.append(mention_logits.detach().float().cpu())

        device = next(model.parameters()).device
        loss_names = sorted(loss_sums)
        totals = torch.tensor(
            [loss_sums[name] for name in loss_names] + [example_count],
            dtype=torch.float64, device=device,
        )
        if is_dist_avail_and_initialized():
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            if confusion is not None:
                dist.all_reduce(confusion, op=dist.ReduceOp.SUM)

        denom = max(totals[-1].item(), 1.0)
        stats = {
            name: totals[index].item() / denom
            for index, name in enumerate(loss_names)
        }

        if confusion is not None:
            true_positive = confusion.diagonal(dim1=1, dim2=2)
            support = confusion.sum(dim=2)
            predicted = confusion.sum(dim=1)
            precision = true_positive / predicted.clamp_min(1.0)
            recall = true_positive / support.clamp_min(1.0)
            f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
            stats.update(
                {
                    "precision_macro": precision.mean().item(),
                    "recall_macro": recall.mean().item(),
                    "f1_macro": f1.mean().item(),
                    "f1_weighted": (
                        (f1 * support).sum() / support.sum().clamp_min(1.0)
                    ).item(),
                    "accuracy": (
                        true_positive.sum() / confusion.sum().clamp_min(1.0)
                    ).item(),
                }
            )
            # CheXpert class index 1 is the positive finding class. Reporting
            # this separately avoids a deceptively high F1 dominated by common
            # negatives and is the primary classification-quality signal.
            if confusion.shape[-1] > 1:
                stats.update(
                    {
                        "precision_positive_macro": precision[:, 1].mean().item(),
                        "recall_positive_macro": recall[:, 1].mean().item(),
                        "f1_positive_macro": f1[:, 1].mean().item(),
                    }
                )

                # Same quantities, averaged over pathologies that actually have
                # positive samples in this split. The keys above dilute the
                # macro with a hard 0 for every pathology with zero positive
                # support, which makes them shift with split composition; these
                # do not. Reported alongside rather than instead, so the
                # selection metric and every historical log stay comparable.
                has_positive = support[:, 1] > 0
                num_defined = int(has_positive.sum().item())
                if num_defined:
                    stats.update(
                        {
                            "precision_positive_macro_defined_only": precision[has_positive, 1].mean().item(),
                            "recall_positive_macro_defined_only": recall[has_positive, 1].mean().item(),
                            "f1_positive_macro_defined_only": f1[has_positive, 1].mean().item(),
                            "num_pathologies_with_positives": float(num_defined),
                        }
                    )
                if num_defined < confusion.shape[0]:
                    logger.info(
                        "%d/%d pathologies have no positive samples in this split; "
                        "they drag f1_positive_macro down but are excluded from "
                        "f1_positive_macro_defined_only",
                        confusion.shape[0] - num_defined,
                        confusion.shape[0],
                    )

        if collected_logits:
            predictions = self._build_predictions(
                collected_logits, collected_labels, collected_keys, collected_mention
            )
            if selection_metric in probability_metrics:
                if is_dist_avail_and_initialized() and dist.get_world_size() > 1:
                    raise RuntimeError(
                        f"{selection_metric} checkpoint selection currently supports "
                        "single-GPU evaluation only"
                    )
                from training.evaluation.classification_metrics import (
                    evaluate_classification,
                )

                uncertain_policy = (
                    str(run_cfg.get("uncertain_policy", "three_class"))
                    if run_cfg is not None
                    else "three_class"
                )
                include_meta_labels = bool(
                    run_cfg.get("include_meta_labels", False)
                    if run_cfg is not None
                    else False
                )
                report = evaluate_classification(
                    predictions,
                    uncertain_policy=uncertain_policy,
                    include_meta_labels=include_meta_labels,
                )
                for key in (
                    "macro_auprc",
                    "macro_auroc",
                    "positive_macro_f1",
                    "positive_micro_f1",
                    "balanced_accuracy",
                ):
                    stats[key] = float(report.aggregates[key])

            if save_predictions:
                self._save_predictions(predictions)

        return stats

    @staticmethod
    def _build_predictions(logits_chunks, label_chunks, keys, mention_chunks=None):
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

        mention_probabilities = None
        if mention_chunks:
            mention = torch.cat(mention_chunks, dim=0)
            if mention.shape == labels.shape:
                mention_probabilities = torch.sigmoid(mention).numpy()
            else:
                logger.warning(
                    "mention logits have shape %s but labels have %s; "
                    "not writing mention_probabilities",
                    tuple(mention.shape),
                    labels.shape,
                )

        return ClassificationPredictions(
            labels=labels,
            probabilities=probabilities,
            logits=logits,
            pathology_names=names,
            sample_keys=build_sample_keys(list(keys[: labels.shape[0]])),
            mention_probabilities=mention_probabilities,
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
