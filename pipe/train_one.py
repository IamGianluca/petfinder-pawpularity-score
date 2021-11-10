import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import omegaconf
import pandas as pd
import pytorch_lightning as pl
from ml import learner
from ml.params import load_cfg
from ml.vision import data
from omegaconf import OmegaConf
from omegaconf.omegaconf import OmegaConf
from pytorch_lightning import callbacks
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.utilities import seed

import constants
import utils


def main():
    cfg = load_cfg(fpath=str(constants.cfg_fpath), cfg_name="train_one")
    print(OmegaConf.to_yaml(cfg))

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
        save_metrics(
            fpath=fpath,
            metric=cfg.metric,
            train_metric=train_metric,
            cv_metric=val_metric,
        )
        neptune_logger.experiment.log_metric("cv_train_metric", train_metric)
        neptune_logger.experiment.log_metric("cv_val_metric", val_metric)

    neptune_logger.experiment.stop()


def train_one_fold(cfg: omegaconf.DictConfig, logger) -> Tuple:

    print()
    print(f"#####################")
    print(f"# FOLD {cfg.fold}")
    print(f"#####################")

    # get image paths and targets
    df = pd.read_csv(constants.train_folds_fpath)
    df_train = df[df.kfold != cfg.fold].reset_index()
    df_val = df[df.kfold == cfg.fold].reset_index()

    train_image_paths, train_targets = utils.get_image_paths_and_targets(
        df=df_train, cfg=cfg
    )
    val_image_paths, val_targets = utils.get_image_paths_and_targets(
        df=df_val, cfg=cfg
    )

    # create datamodule
    dm = data.ImageDataModule(
        task="classification",
        batch_size=cfg.bs,
        # train
        train_image_paths=train_image_paths,
        train_targets=train_targets,
        # valid
        val_image_paths=val_image_paths,
        val_targets=val_targets,
        # test
        test_image_paths=val_image_paths,
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
        filename=f"model_one_fold{cfg.fold}",
        save_weights_only=True,
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
        trainer.tune(model, dm)

    trainer.fit(model, dm)
    preds = trainer.predict(model, dm.test_dataloader())

    save_predictions(cfg, preds)

    print_metrics(cfg.metric, model.best_train_metric, model.best_val_metric)
    return (
        model.best_train_metric.detach().cpu().numpy(),
        model.best_val_metric.detach().cpu().numpy(),
    )


def save_predictions(cfg: OmegaConf, preds: List[List]):
    preds = np.vstack([i for sl in preds for i in sl])
    with open(f"preds/model_one_fold{cfg.fold}.npy", "wb") as f:
        np.save(f, preds)


def print_metrics(metric: str, train_metric: float, valid_metric: float):
    print(
        f"\nBest {metric}: Train {train_metric:.4f}, Valid: {valid_metric:.4f}"
    )


def save_metrics(
    fpath: Path, metric: str, train_metric: float, cv_metric: float
):
    data = {}
    data[f"train {metric}"] = f"{train_metric:.4f}"
    data[f"cv {metric}"] = f"{cv_metric:.4f}"
    with open(fpath, "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    main()
