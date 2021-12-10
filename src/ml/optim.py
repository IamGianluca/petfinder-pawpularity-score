from typing import Iterable

import torch
import transformers
from torch import optim
from torch.optim import lr_scheduler
from torch.optim._multi_tensor import SGD


def optimizer_factory(params, hparams):
    if hparams.opt == "adam":
        return optim.Adam(
            params,
            lr=hparams.lr,
            weight_decay=hparams.wd,
        )
    if hparams.opt == "adamw":
        return transformers.AdamW(
            params,
            lr=hparams.lr,
            betas=(0.9, 0.999),
            eps=1e-06,
            weight_decay=hparams.wd,
            correct_bias=True,
        )
    if hparams.opt == "sam":
        return SAM(
            params,
            lr=hparams.lr,
            momentum=hparams.mom,
            weight_decay=hparams.wd,
        )
    else:
        raise ValueError("Optimizer not supported yet.")


def lr_scheduler_factory(optimizer, hparams, data_loader):
    steps_per_epoch = len(data_loader)
    if hparams.sched == "plateau":
        return lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            patience=2,
            threshold=0.01,
            factor=0.1,
            verbose=True,
        )
    elif hparams.sched == "onecycle":
        return lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=hparams.lr,
            cycle_momentum=True,
            pct_start=0.1,  # hparams.warmup_epochs / hparams.epochs,
            div_factor=25.0,
            final_div_factor=100000.0,
            steps_per_epoch=steps_per_epoch,
            epochs=hparams.epochs,
        )
    elif hparams.sched == "cosine":
        return transformers.get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=steps_per_epoch * hparams.warmup_epochs,
            num_training_steps=steps_per_epoch * hparams.epochs,
        )
    elif hparams.sched == "cosine_with_restart":
        return lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=20,
            eta_min=1e-4,
        )
    else:
        raise ValueError("Learning rate scheduler not supported yet.")


# copied from https://github.com/moskomule/sam.pytorch/blob/main/sam.py
class SAM(SGD):
    """SGD wrapped with Sharp-Aware Minimization
    Args:
        params: tensors to be optimized
        lr: learning rate
        momentum: momentum factor
        dampening: damping factor
        weight_decay: weight decay factor
        nesterov: enables Nesterov momentum
        rho: neighborhood size
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False,
        rho: float = 0.05,
    ):
        if rho <= 0:
            raise ValueError(f"Invalid neighborhood size: {rho}")
        super().__init__(
            params, lr, momentum, dampening, weight_decay, nesterov
        )
        # todo: generalize this
        if len(self.param_groups) > 1:
            raise ValueError("Not supported")
        self.param_groups[0]["rho"] = rho

    @torch.no_grad()
    def step(self, closure=None) -> torch.Tensor:
        """
        Args:
            closure: A closure that reevaluates the model and returns the loss.
        Returns: the loss value evaluated on the original point
        """
        loss = None
        if closure is not None:
            closure = torch.enable_grad()(closure)
            loss = closure().detach()

        for group in self.param_groups:
            grads = []
            params_with_grads = []

            rho = group["rho"]
            # update internal_optim's learning rate

            for p in group["params"]:
                if p.grad is not None:
                    # without clone().detach(), p.grad will be zeroed by closure()
                    grads.append(p.grad.clone().detach())
                    params_with_grads.append(p)
            device = grads[0].device

            # compute \hat{\epsilon}=\rho/\norm{g}\|g\|
            grad_norm = torch.stack(
                [g.detach().norm(2).to(device) for g in grads]
            ).norm(2)
            epsilon = grads  # alias for readability
            torch._foreach_mul_(epsilon, rho / grad_norm)

            # virtual step toward \epsilon
            torch._foreach_add_(params_with_grads, epsilon)
            # compute g=\nabla_w L_B(w)|_{w+\hat{\epsilon}}
            if closure is not None:
                closure()
            # virtual step back to the original point
            torch._foreach_sub_(params_with_grads, epsilon)

        super().step()
        return loss
