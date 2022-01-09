from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from ml.learner import ImageClassifier
from ml.params import load_cfg
from ml.vision.data import ImageDataModule
from omegaconf import OmegaConf
from timm.data import transforms_factory

import constants


def main():
    df = pd.read_csv(constants.train_folds_all_fpath)

    test_image_fpaths = [
        Path(f"data/train/{i}.jpg") for i in df["Id"].values.tolist()
    ]
    test_targets = [[t] for t in df.Pawpularity.values.tolist()]
    val_aug = transforms_factory.create_transform(
        input_size=384, is_training=False
    )

    cfg = load_cfg(constants.cfg_fpath, cfg_name=f"train_zero")
    cfg = OmegaConf.create(cfg)

    n_folds = int(df.kfold.max() + 1)
    ckpt_fpaths = [
        f"ckpts/model_{cfg.name}_fold{i}.ckpt" for i in range(n_folds)
    ]
    model = ImageClassifier(cfg=cfg, in_channels=3, num_classes=1)

    print(f"Generating predictions using model {cfg.name}...")
    tmp = df.loc[:, ["Id", "Pawpularity"]]
    for idx, ckpt_fpath in enumerate(ckpt_fpaths):
        ckpt = torch.load(ckpt_fpath)
        model.load_state_dict(ckpt["state_dict"])

        dm = ImageDataModule(
            task="classification",
            batch_size=cfg.bs * 6,
            test_image_paths=test_image_fpaths,
            test_augmentations=val_aug,
        )
        dm.setup()

        trainer = pl.Trainer(gpus=1)
        preds = trainer.predict(model, dm.test_dataloader())

        tmp[f"preds_model_{cfg.name}_fold{idx}"] = np.vstack(preds) * 100

    preds_cols = [f"preds_model_{cfg.name}_fold{i}" for i in range(n_folds)]

    cfg = load_cfg(constants.cfg_fpath, cfg_name=f"remove_hardest_samples")
    cfg = OmegaConf.create(cfg)

    print(
        f"Identifying top {1 - cfg.pct_to_keep:.0%} hardest to classify samples..."
    )
    tmp["y_pred"] = tmp.loc[:, preds_cols].mean(axis=1)
    tmp["y_true"] = np.array(test_targets)
    tmp["y_diff"] = np.abs(tmp.y_true - tmp.y_pred)
    order = np.argsort(tmp.y_diff)
    tmp["hard_index"] = order.argsort()

    n_keep = tmp.shape[0] * cfg.pct_to_keep
    print(
        f"We are going to keep {n_keep} samples and ignore the rest for future model training..."
    )

    df["ignore"] = np.where(tmp["hard_index"] > n_keep, 1, 0)

    # drop columns
    df.to_csv(constants.train_folds_fpath)


if __name__ == "__main__":
    main()
