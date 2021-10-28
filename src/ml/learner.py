from typing import Any, List, Optional

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from timm.models import create_model
from torch._C import Value

from .loss import loss_factory
from .metrics import metric_factory
from .optim import lr_scheduler_factory, optimizer_factory


class ImageClassifier(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        cfg: OmegaConf,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(cfg)

        self.model = create_model(
            model_name=self.hparams.arch,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=in_channels,
            drop_rate=self.hparams.dropout,
        )

        self.train_metric = metric_factory(name=cfg.metric)
        self.val_metric = metric_factory(name=cfg.metric)
        self.best_train_metric = None
        self.best_val_metric = None

    def forward(self, x):
        """Contain only tensor operations with your model."""
        x = x.float()
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        """Encapsulate forward() logic with logging, metrics, and loss
        computation.
        """
        loss, target, preds = self.step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.log(
            "train_metric",
            self.train_metric(preds, target),
            on_step=False,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, target, preds = self.step(batch)
        self.log("val_loss", loss, on_step=True, on_epoch=False)
        self.log(
            "val_metric",
            self.val_metric(preds, target),
            on_step=False,
            on_epoch=True,
        )
        return loss

    def validation_epoch_end(self, outputs: List):
        self.register_best_train_and_val_metrics()
        # BUG: the metrics for the very last epoch are not printed
        # but are nonetheless logged in neptune
        self.print_metrics_to_console()

    def step(self, batch):
        x, target = batch

        # TODO: this should be done outside of the LightningModule
        target = target.float()

        preds = self.forward(x)
        # TODO: handle logit vs. no logit case for both loss and preds
        loss = self.compute_loss(preds=preds, target=target)
        return loss, target, preds.sigmoid()

    def configure_optimizers(self):
        optimizer = optimizer_factory(
            params=self.parameters(), hparams=self.hparams
        )

        scheduler = lr_scheduler_factory(
            optimizer=optimizer,
            hparams=self.hparams,
            data_loader=self.train_dataloader(),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_metric",
        }

    def compute_loss(self, preds, target):
        # apply label smoothing
        # TODO: extract function
        y_ = (
            target * (1 - self.hparams.label_smoothing)
            + 0.5 * self.hparams.label_smoothing
        )

        loss_fn = loss_factory(name=self.hparams.loss)
        loss = loss_fn(preds, target.float())
        return loss

    def compute_metric(self, preds, target):
        metric_fn = metric_factory(name=self.hparams.metric)
        return metric_fn(preds, target)

    def register_best_train_and_val_metrics(self):
        try:
            train_metric = self.trainer.callback_metrics["train_metric"]
            val_metric = self.trainer.callback_metrics["val_metric"]
            if self.best_val_metric is None or self.is_metric_better(
                val_metric
            ):
                self.best_val_metric = val_metric
                self.best_train_metric = train_metric
        except (KeyError, AttributeError):
            # these errors occurs when in "tuning" mode (find optimal lr)
            pass

    def is_metric_better(self, new_metric):
        if self.hparams.metric_mode == "max":
            return new_metric > self.best_val_metric
        elif self.hparams.metric_mode == "min":
            return new_metric < self.best_val_metric
        else:
            raise ValueError("metric_mode can only be min or max")

    def print_metrics_to_console(self):
        try:
            train_metric = self.trainer.callback_metrics["train_metric"]
            val_metric = self.trainer.callback_metrics["val_metric"]
            self.trainer.progress_bar_callback.main_progress_bar.write(
                f"Epoch {self.current_epoch} // "
                f"train metric: {train_metric:.4f}, valid metric: {val_metric:.4f}"
            )
        except (KeyError, AttributeError):
            # these errors occurs when in "tuning" mode (find optimal lr)
            pass

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> Any:
        """Encapsulate forward() with any necessary preprocess or postprocess
        functions.
        """
        preds = self.forward(batch)
        # TODO: we don't always need a sigmoid. Handle that case.
        outs = preds.sigmoid()
        return outs.detach().cpu().numpy()

    # def predict_proba(self, dl):
    #     self.eval()
    #     self.to("cuda")

    #     for batch in dl():
    #         x = batch.float()
    #         x = x.to("cuda")
    #         with torch.no_grad():
    #             preds = self.model(x)
    #             outs = preds.sigmoid()
    #             yield outs.detach().cpu().numpy()
