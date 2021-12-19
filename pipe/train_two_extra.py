import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import omegaconf
import pandas as pd
import pytorch_lightning as pl
import torch
from ml import learner
from ml.params import load_cfg
from ml.vision import data
from omegaconf import OmegaConf
from omegaconf.omegaconf import OmegaConf
from pytorch_lightning import callbacks
from pytorch_lightning.callbacks import lr_monitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import seed
from timm.data import transforms_factory
from torchmetrics import MeanSquaredError

import constants
import utils


def train(cfg: OmegaConf):
    print(OmegaConf.to_yaml(cfg))

    seed.seed_everything(seed=cfg.seed, workers=False)

    wandb_logger = WandbLogger(
        project="paw",
    )

    is_crossvalidation = True if cfg.fold == -1 else False
    if is_crossvalidation:
        y_true = []
        y_pred = []
        valid_scores = []
        train_scores = []

        for current_fold in range(cfg.n_folds):
            cfg.fold = current_fold
            train_score, valid_score, target, preds = train_one_fold(
                cfg=cfg, logger=wandb_logger
            )
            y_true.extend(target)
            y_pred.extend(preds)
            valid_scores.append(valid_score)
            train_scores.append(train_score)

        # needed for final ensemble
        save_predictions(cfg=cfg, preds=y_pred)

        train_metric = np.mean(train_scores)
        val_metric = np.mean(valid_scores)

        rmse = MeanSquaredError(squared=False)
        y_pred = torch.tensor(y_pred)
        y_true = torch.tensor(y_true)
        oof_metric = rmse(y_pred, y_true)
    else:
        train_metric, val_metric = train_one_fold(cfg=cfg, logger=wandb_logger)

    if is_crossvalidation:
        fpath = constants.metrics_path / f"model_{cfg.name}.json"
        save_metrics(
            fpath=fpath,
            metric=cfg.metric,
            train_metric=train_metric,
            cv_metric=val_metric,
            oof_metric=oof_metric,
        )
        wandb_logger.log_metrics({"cv_train_metric": train_metric})
        wandb_logger.log_metrics({"cv_val_metric": val_metric})
        wandb_logger.log_metrics({"oof_val_metric": oof_metric})


def train_one_fold(cfg: omegaconf.DictConfig, logger) -> Tuple:

    print()
    print(f"#####################")
    print(f"# FOLD {cfg.fold}")
    print(f"#####################")

    # get image paths and targets
    df = pd.read_csv(utils.train_folds_fpath[cfg.n_folds])
    df_train = df[df.kfold != cfg.fold].reset_index()
    df_val = df[df.kfold == cfg.fold].reset_index()

    train_image_paths, train_targets = utils.get_image_paths_and_targets(
        df=df_train, cfg=cfg, include_extra=cfg.use_extra_images
    )
    val_image_paths, val_targets = utils.get_image_paths_and_targets(
        df=df_val, cfg=cfg, include_extra=False
    )

    # define augmentations
    train_aug = transforms_factory.create_transform(
        input_size=cfg.sz,
        is_training=True,
        auto_augment=f"rand-n{cfg.n_tfms}-m{cfg.magn}",
    )
    val_aug = transforms_factory.create_transform(
        input_size=cfg.sz,
        is_training=False,
    )

    # create datamodule
    dm = data.ImageDataModule(
        task="classification",
        batch_size=cfg.bs,
        # train
        train_image_paths=train_image_paths,
        train_targets=train_targets,
        train_augmentations=train_aug,
        # valid
        val_image_paths=val_image_paths,
        val_targets=val_targets,
        val_augmentations=val_aug,
        # test
        test_image_paths=val_image_paths,
        test_augmentations=val_aug,
    )

    model = learner.ImageClassifier(
        in_channels=3,
        num_classes=1,
        pretrained=cfg.pretrained,
        cfg=cfg,
    )

    checkpoint_callback = callbacks.ModelCheckpoint(
        monitor="val_metric",
        mode=cfg.metric_mode,
        dirpath=constants.ckpts_path,
        filename=f"model_{cfg.name}_fold{cfg.fold}",
        save_weights_only=True,
    )
    lr_callback = lr_monitor.LearningRateMonitor(
        logging_interval="step", log_momentum=True
    )

    trainer = pl.Trainer(
        gpus=1,
        precision=cfg.precision,
        auto_lr_find=cfg.auto_lr,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        auto_scale_batch_size=cfg.auto_batch_size,
        max_epochs=cfg.epochs,
        logger=logger,
        callbacks=[checkpoint_callback, lr_callback],
        # limit_train_batches=1,
        # limit_val_batches=1,
    )

    if cfg.auto_lr or cfg.auto_batch_size:
        trainer.tune(model, dm)

    trainer.fit(model, dm)
    targets_list = df_val.loc[:, "Pawpularity"].values.tolist()
    preds = trainer.predict(model, dm.test_dataloader(), ckpt_path="best")
    preds_list = [p[0] * 100 for b in preds for p in b]

    print_metrics(cfg.metric, model.best_train_metric, model.best_val_metric)
    return (
        model.best_train_metric.detach().cpu().numpy(),
        model.best_val_metric.detach().cpu().numpy(),
        targets_list,
        preds_list,
    )


def save_predictions(cfg: OmegaConf, preds: List):
    preds = np.array(preds)
    with open(f"preds/model_{cfg.name}_oof.npy", "wb") as f:
        np.save(f, preds)


def print_metrics(metric: str, train_metric: float, valid_metric: float):
    print(
        f"\nBest {metric}: Train {train_metric:.4f}, Valid: {valid_metric:.4f}"
    )


def save_metrics(
    fpath: Path,
    metric: str,
    train_metric: float,
    cv_metric: float,
    oof_metric: float,
):
    data = {}
    data[f"train {metric}"] = round(float(train_metric), 4)
    data[f"cv {metric}"] = round(float(cv_metric), 4)
    data[f"oof {metric}"] = round(float(oof_metric), 4)
    with open(fpath, "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    cfg = load_cfg(fpath=str(constants.cfg_fpath), cfg_name=f"train_two_extra")
    train(cfg)
