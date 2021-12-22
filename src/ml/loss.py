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
    elif name == "crossentropy":
        return nn.CrossEntropyLoss()
    elif name == "focal_loss":
        return BinaryFocalLossWithLogits()
    else:
        raise ValueError(f"{name} loss not supported yet.")


class BinaryFocalLossWithLogits(nn.Module):
    def __init__(
        self, alpha: float = 0.25, gamma: float = 2, reduction: bool = True
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # first compute binary cross-entropy
        bce = self.criterion(inputs, targets)
        bce_exp = torch.exp(-bce)
        focal_loss = self.alpha * (1 - bce_exp) ** self.gamma * bce

        if self.reduction:
            return focal_loss.mean()
        else:
            return focal_loss


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
        if y.shape[1] == 1:  # no mixup
            loss = self.criterion(logits, y)
        elif y.shape[1] == 3:  # mixup
            lam = y[:, 2]
            loss_a = self.criterion(logits, y[:, 0].view(-1, 1))
            loss_b = self.criterion(logits, y[:, 1].view(-1, 1))
            loss = (1 - lam) * loss_a + lam * loss_b
        else:
            raise ValueError(
                f"y tensor should be of shape (batch_size, 1), found {y.shape}"
            )

        if self.reduction:
            return loss.mean()


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
