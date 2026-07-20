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
import torch
import torch.distributed as dist


@registry.register_task("image_text_pretrain_eval")
class ImageTextPretrainTask(BaseTask):
    def __init__(self):
        super().__init__()

    def evaluation(self, model, data_loader, cuda_enabled=True):
        loss_sums = {}
        example_count = 0
        confusion = None

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

        return stats
