from typing import Any, List, Optional

import pytorch_lightning as pl
import timm
from omegaconf import OmegaConf
from torch import nn

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

        self.backbone = timm.create_model(
            model_name=self.hparams.arch,
            pretrained=pretrained,
            num_classes=0,
            in_chans=in_channels,
            drop_rate=self.hparams.dropout,
        )
        self.head = nn.Sequential(
            nn.LazyLinear(128),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.Linear(64, num_classes),
        )

        self.train_metric = metric_factory(cfg=cfg)
        self.val_metric = metric_factory(cfg=cfg)
        self.best_train_metric = None
        self.best_val_metric = None

    def forward(self, x):
        """Contain only tensor operations with your model."""
        x = self.backbone(x)
        x = self.head(x)
        return x

    def training_step(self, batch, batch_idx):
        """Encapsulate forward() logic with logging, metrics, and loss
        computation.
        """
        x, target = batch
        loss, target, preds = self._step(x, target)

        if target.shape[1] == 3:  # if using MixUp, compute target
            lam = target[:, 2]
            target = (1 - lam) * target[:, 0] + lam * target[:, 1]
            target = target.reshape(-1, 1)

        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.train_metric.update(preds=preds, target=target)
        return loss

    def training_epoch_end(self, outputs) -> None:
        self.log("train_metric", self.train_metric.compute())

    def validation_step(self, batch, batch_idx):
        x, target = batch
        loss, target, preds = self._step(x, target)

        self.log("val_loss", loss, on_step=True, on_epoch=False)
        self.val_metric.update(preds=preds, target=target)
        return loss

    def validation_epoch_end(self, outputs: List):
        self.log("val_metric", self.val_metric.compute())
        self._register_best_train_and_val_metrics()
        # BUG: the metrics for the very last epoch are not printed
        # but are nonetheless logged in neptune
        self._print_metrics_to_console()

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> Any:
        """Encapsulate forward() with any necessary preprocess or postprocess
        functions.
        """
        x = batch

        preds = self.forward(x)
        # TODO: we don't always need a sigmoid. Handle that case.
        outs = preds.sigmoid()
        return outs.detach().cpu().float().numpy()

    def _step(self, x, target):
        preds = self.forward(x)
        # TODO: handle logit vs. no logit case for both loss and preds
        loss = self._compute_loss(preds=preds, target=target)
        return loss, target, preds.sigmoid()

    def configure_optimizers(self):
        optimizer = optimizer_factory(
            params=self.parameters(), hparams=self.hparams
        )

        scheduler = lr_scheduler_factory(
            optimizer=optimizer,
            hparams=self.hparams,
            data_loader=self.trainer.datamodule.train_dataloader(),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "monitor": "val_metric",
                "strict": True,
                "name": "lr",
            },
        }

    def _compute_loss(self, preds, target):
        if self.hparams.label_smoothing > 0.0:
            target = (
                target * (1 - self.hparams.label_smoothing)
                + 0.5 * self.hparams.label_smoothing
            )

        loss_fn = loss_factory(name=self.hparams.loss)
        loss = loss_fn(preds, target)
        return loss

    def _register_best_train_and_val_metrics(self):
        try:
            train_metric = self.trainer.callback_metrics["train_metric"]
            val_metric = self.trainer.callback_metrics["val_metric"]
            if self.best_val_metric is None or self._is_metric_better(
                val_metric
            ):
                self.best_val_metric = val_metric
                self.best_train_metric = train_metric
        except (KeyError, AttributeError):
            # these errors occurs when in "tuning" mode (find optimal lr)
            pass

    def _is_metric_better(self, new_metric):
        if self.hparams.metric_mode == "max":
            return new_metric > self.best_val_metric
        elif self.hparams.metric_mode == "min":
            return new_metric < self.best_val_metric
        else:
            raise ValueError("metric_mode can only be min or max")

    def _print_metrics_to_console(self):
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
