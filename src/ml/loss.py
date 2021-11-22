import torch
import torch.nn as nn
import torch.nn.functional as F


def loss_factory(name):
    if name == "bce":
        return nn.BCELoss()
    elif name == "bce_with_logits":
        return nn.BCEWithLogitsLoss()
    elif name == "mixup_bce_with_logits":
        return MixUpBCEWithLogitsLoss()
    elif name == "mixup_ce_with_logits":
        return MixUpCrossEntropy()
    else:
        raise ValueError(f"{name} loss not supported yet.")


class MixUpBCEWithLogitsLoss:
    "Cross entropy that works if there is a probability of MixUp being applied."

    def __init__(self, reduction: bool = True):
        """
        Args:
            reduction (bool): True if mean is applied after loss.
        """
        self.reduction = "mean" if reduction else "none"
        self.criterion = nn.BCEWithLogitsLoss()

    def __call__(self, logits: torch.Tensor, y: torch.LongTensor):
        """
        Args:
            logits (torch.Tensor): Output of the model.
            y (torch.LongTensor): Targets of shape (batch_size, 1) or (batch_size, 3).
        """
        assert (
            y.shape[1] == 1 or y.shape[1] == 3
        ), f"Invalid shape for targets {y.shape}."

        if y.shape[1] == 1:
            loss = self.criterion(logits, y)

        elif y.shape[1] == 3:
            loss_a = self.criterion(logits, y[:, 0].reshape(-1, 1))
            loss_b = self.criterion(logits, y[:, 1].reshape(-1, 1))
            loss = (1 - y[:, 2]) * loss_a + y[:, 2] * loss_b

        if self.reduction == "mean":
            return loss.mean()
        return loss


class MixUpCrossEntropy:
    "Cross entropy that works if there is a probability of MixUp being applied."

    def __init__(self, reduction: bool = True):
        """
        Args:
            reduction (bool): True if mean is applied after loss.
        """
        self.reduction = "mean" if reduction else "none"
        self.criterion = F.cross_entropy

    def __call__(self, logits: torch.Tensor, y: torch.LongTensor):
        """
        Args:
            logits (torch.Tensor): Output of the model.
            y (torch.LongTensor): Targets of shape (batch_size) or (batch_size, 3).
        """
        assert len(y.shape) == 1 or y.shape[1] == 3, "Invalid targets."

        if len(y.shape) == 1:
            loss = self.criterion(logits, y, reduction=self.reduction)

        elif y.shape[1] == 3:
            loss_a = self.criterion(
                logits, y[:, 0].long(), reduction=self.reduction
            )
            loss_b = self.criterion(
                logits, y[:, 1].long(), reduction=self.reduction
            )
            loss = (1 - y[:, 2]) * loss_a + y[:, 2] * loss_b

        if self.reduction == "mean":
            return loss.mean()
        return loss
