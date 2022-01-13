import json
from pathlib import Path
from typing import List, Tuple

import joblib
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
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import seed
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from timm.data import transforms_factory
from torch import nn
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
        fpath = constants.metrics_path / f"model_five.json"
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
    df = pd.read_csv(constants.train_folds_fpath)
    df_train = df[df.kfold != cfg.fold].reset_index()
    df_val = df[df.kfold == cfg.fold].reset_index()

    train_image_paths, train_targets = utils.get_image_paths_and_targets(
        df=df_train,
        cfg=cfg,
        ignore=True,  # TODO: False, use L1 model instead of extra2
    )
    val_image_paths, val_targets = utils.get_image_paths_and_targets(
        df=df_val,
        cfg=cfg,
        ignore=False,
    )

    # define augmentations
    val_aug = transforms_factory.create_transform(
        input_size=cfg.sz,
        is_training=False,
    )

    model = learner.ImageClassifier(
        in_channels=3,
        num_classes=1,
        pretrained=cfg.pretrained,
        cfg=cfg,
    )

    # load pre-trained weights
    ckpt_fpath = f"ckpts/model_{cfg.name}_fold{cfg.fold}.ckpt"
    ckpt = torch.load(ckpt_fpath)
    model.load_state_dict(ckpt["state_dict"])

    # swap model's head as we are interested in extracting the embeddings
    model.head = nn.Identity()

    trainer = pl.Trainer(
        gpus=1,
        precision=cfg.precision,
    )

    # create datamodule
    dm = data.ImageDataModule(
        task="classification",
        batch_size=int(cfg.bs),
        # test
        test_image_paths=train_image_paths,
        test_augmentations=val_aug,
    )
    dm.setup()

    train_emb = trainer.predict(model, dm.test_dataloader())
    train_emb = np.vstack(train_emb)
    tar = np.array(train_targets)

    lr = make_pipeline(StandardScaler(), SVR(C=0.05))
    lr.fit(X=train_emb, y=tar.ravel())
    joblib.dump(lr, f"ckpts/model_five_fold{cfg.fold}.ckpt")

    preds = np.clip(lr.predict(X=train_emb), a_min=0, a_max=1)

    train_metric = mean_squared_error(tar, preds, squared=False)

    # create datamodule
    dm = data.ImageDataModule(
        task="classification",
        batch_size=int(cfg.bs),
        # test
        test_image_paths=val_image_paths,
        test_augmentations=val_aug,
    )
    dm.setup()

    test_emb = trainer.predict(model, dm.test_dataloader())
    test_emb = np.vstack(test_emb)

    preds = np.clip(lr.predict(X=test_emb), a_min=0, a_max=1)

    val_metric = mean_squared_error(val_targets, preds, squared=False)
    print_metrics(cfg.metric, train_metric, val_metric)

    preds_list = (preds * 100).tolist()
    targets_list = df_val.loc[:, "Pawpularity"].values.tolist()
    return (
        train_metric,
        val_metric,
        targets_list,
        preds_list,
    )


def save_predictions(cfg: OmegaConf, preds: List):
    preds = np.array(preds)
    with open(f"preds/model_five_oof.npy", "wb") as f:
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
    cfg = load_cfg(
        fpath=str(constants.cfg_fpath), cfg_name=f"train_one_extra2"
    )
    train(cfg)
