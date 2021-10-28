import json
import os
from pathlib import Path
from typing import Tuple

import albumentations
import numpy as np
import omegaconf
import pandas as pd
import pytorch_lightning as pl
from albumentations.pytorch import transforms
from loguru import logger
from ml import learner
from ml.params import load_cfg
from ml.vision import data
from numpy.typing import ArrayLike
from omegaconf import OmegaConf
from omegaconf.omegaconf import OmegaConf
from pytorch_lightning import callbacks
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.utilities import seed

import constants
import utils


def main():
    cfg = load_cfg(fpath=str(constants.cfg_fpath), cfg_name="train_one")

    seed.seed_everything(seed=cfg.seed, workers=False)

    neptune_logger = NeptuneLogger(
        api_key=os.environ["NEPTUNE_API_TOKEN"],
        project_name="gr.gianlucarossi/paw",
        experiment_name="logs",
        close_after_fit=False,
    )

    is_crossvalidation = True if cfg.fold == -1 else False
    if is_crossvalidation:  # run 5-fold CV
        valid_scores = []
        train_scores = []

        for current_fold in range(5):
            cfg.fold = current_fold
            train_score, valid_score = train_one_fold(
                cfg=cfg, logger=neptune_logger
            )
            valid_scores.append(valid_score)
            train_scores.append(train_score)

        train_metric = np.mean(train_scores)
        val_metric = np.mean(valid_scores)
    else:  # run one fold
        train_metric, val_metric = train_one_fold(
            cfg=cfg, logger=neptune_logger
        )

    if is_crossvalidation:
        fpath = constants.metrics_path / f"model_one.json"
        write_metrics(
            fpath=fpath,
            metric=cfg.metric,
            train_metric=train_metric,
            cv_metric=val_metric,
        )
        neptune_logger.experiment.log_metric("cv_train_metric", train_metric)
        neptune_logger.experiment.log_metric("cv_val_metric", val_metric)

    neptune_logger.experiment.stop()


def train_one_fold(
    cfg: omegaconf.DictConfig, logger
) -> Tuple[ArrayLike, ArrayLike]:
    print(OmegaConf.to_yaml(cfg))

    # get image paths and targets
    df = pd.read_csv(constants.train_folds_fpath)
    df_train = df[df.kfold != cfg.fold].reset_index()
    df_val = df[df.kfold == cfg.fold].reset_index()

    train_image_paths, train_targets = utils.get_image_paths_and_targets(
        df=df_train
    )
    val_image_paths, val_targets = utils.get_image_paths_and_targets(df=df_val)
    train_aug, val_aug, test_aug = get_augmentations(cfg=cfg)

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
    )

    model = learner.ImageClassifier(
        in_channels=3,
        num_classes=1,
        pretrained=True,
        cfg=cfg,
    )

    checkpoint_callback = callbacks.ModelCheckpoint(
        monitor="val_metric",
        mode=cfg.metric_mode,
        dirpath=constants.ckpts_path,
        filename=f"arch={cfg.arch}_sz={cfg.sz}_fold={cfg.fold}",
    )

    trainer = pl.Trainer(
        gpus=1,
        precision=cfg.precision,
        auto_lr_find=cfg.auto_lr,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        auto_scale_batch_size=cfg.auto_batch_size,
        max_epochs=cfg.epochs,
        logger=logger,
        callbacks=[checkpoint_callback],
        # limit_train_batches=1,
        # limit_val_batches=1,
    )

    if cfg.auto_lr or cfg.auto_batch_size:
        logger.info("\nTuning...")
        trainer.tune(model, dm)

    trainer.fit(model, dm)

    print_metrics(cfg.metric, model.best_train_metric, model.best_val_metric)
    return (
        model.best_train_metric.detach().cpu().numpy(),
        model.best_val_metric.detach().cpu().numpy(),
    )


def print_metrics(metric: str, train_metric: float, valid_metric: float):
    logger.info(
        f"\nBest {metric}: Train {train_metric:.4f}, Valid: {valid_metric:.4f}"
    )


def write_metrics(
    fpath: Path, metric: str, train_metric: float, cv_metric: float
):
    data = {}
    data[f"train {metric}"] = f"{train_metric:.4f}"
    data[f"cv {metric}"] = f"{cv_metric:.4f}"
    with open(fpath, "w") as f:
        json.dump(data, f)


def get_augmentations(cfg: OmegaConf):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if cfg.aug == "none":
        train_aug = albumentations.Compose(
            [
                albumentations.Resize(cfg.sz, cfg.sz),
                transforms.ToTensorV2(),
            ]
        )
        val_aug = train_aug
        test_aug = train_aug
    return train_aug, val_aug, test_aug


if __name__ == "__main__":
    main()