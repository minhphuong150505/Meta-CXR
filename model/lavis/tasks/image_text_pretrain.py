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
        loss = 0.0
        precision = 0.0
        recall = 0.0
        f1_score = 0.0
        accuracy = 0.0
        dataloader_len = len(data_loader)
        for batch in data_loader:
            if cuda_enabled:
                batch = move_to_cuda(batch)
            loss_dict = model(batch)
            loss += loss_dict["loss"].item()
            
            if "average_precision" in loss_dict:
                precision += loss_dict["average_precision"].item()
            if "average_recall" in loss_dict:
                recall += loss_dict["average_recall"].item()
            if "average_f1_score" in loss_dict:
                f1_score += loss_dict["average_f1_score"].item()
            if "average_accuracy" in loss_dict:
                accuracy += loss_dict["average_accuracy"].item()
        device = next(model.parameters()).device
        totals = torch.tensor(
            [loss, precision, recall, f1_score, accuracy, dataloader_len],
            dtype=torch.float64,
            device=device,
        )
        if is_dist_avail_and_initialized():
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)

        denom = totals[5].item()
        stats = {
            "loss": totals[0].item() / denom,
            "precision": totals[1].item() / denom,
            "recall": totals[2].item() / denom,
            "f1_score": totals[3].item() / denom,
            "accuracy": totals[4].item() / denom,
        }

        print(
            f"Average Loss: {stats['loss']} | "
            f"Average Precision: {stats['precision']} | "
            f"Average Recall: {stats['recall']} | "
            f"Average f1 score: {stats['f1_score']} | "
            f"Average Accuracy: {stats['accuracy']}"
        )
        return stats
