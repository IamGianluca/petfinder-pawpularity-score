from typing import Any, Callable, Optional

import torch
import torchmetrics
from torch import Tensor
from torchmetrics.functional.regression.mean_squared_error import (
    mean_squared_error,
)
from torchmetrics.metric import Metric


def metric_factory(cfg):
    if cfg.metric == "auc":
        # return torchmetrics.AUROC(pos_label=1)
        return torchmetrics.AUROC(pos_label=1, num_classes=cfg.num_classes)
    elif cfg.metric == "mse":
        return torchmetrics.MeanSquaredError(squared=True)
    elif cfg.metric == "rmse":
        return RootMeanSquaredError()
    else:
        raise ValueError("Metric not supported yet.")


class RootMeanSquaredError(Metric):
    def __init__(
        self,
        compute_on_step: bool = False,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ) -> None:
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.target = torch.empty(0).cuda()
        self.preds = torch.empty(0).cuda()

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        self.target = torch.cat([self.target, target], 0)
        self.preds = torch.cat([self.preds, preds], 0)

    def compute(self) -> Tensor:
        """Computes mean squared error over state."""
        preds = self.preds
        target = self.target

        # reset for next epoch
        self.preds = torch.empty(0).cuda()
        self.target = torch.empty(0).cuda()

        return mean_squared_error(preds=preds, target=target, squared=False)

    @property
    def is_differentiable(self) -> bool:
        return True
