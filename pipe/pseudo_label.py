from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from ml.learner import ImageClassifier
from ml.params import load_cfg
from ml.vision import data
from omegaconf import OmegaConf
from timm.data import transforms_factory

import constants


def pseudo_label():
    cfg = load_cfg(
        constants.cfg_fpath,
        cfg_name=f"pseudo_labeling",
    )
    cfg = OmegaConf.create(cfg)

    extra_image_paths = list(Path(constants.extra_images_path).glob("*.jpg"))

    df = pd.DataFrame()
    df["Id"] = np.array([p.name.split(".")[0] for p in extra_image_paths])

    # for each model, set up ImageDataModule
    for model_name in cfg.models:
        print()
        print(f"#####################")
        print(f"# Generating predictions for model {model_name}")
        print(f"#####################")
        model_cfg = load_cfg(
            constants.cfg_fpath, cfg_name=f"train_{model_name}"
        )
        test_aug = transforms_factory.create_transform(
            input_size=model_cfg.sz,
            is_training=False,
        )
        dm = data.ImageDataModule(
            task="classification",
            batch_size=model_cfg.bs * 6,
            test_image_paths=extra_image_paths,
            test_augmentations=test_aug,
        )
        dm.setup()
        model = ImageClassifier(cfg=model_cfg, in_channels=3, num_classes=1)

        # for each fold, load weights from all relevant model
        # and predict what would be the label
        ckpt_fpaths = [
            f"ckpts/model_{model_name}_fold{i}.ckpt"
            for i in range(model_cfg.n_folds)
        ]
        for idx, ckpt_fpath in enumerate(ckpt_fpaths):
            ckpt = torch.load(ckpt_fpath)
            model.load_state_dict(ckpt["state_dict"])

            trainer = pl.Trainer(gpus=1)
            preds = trainer.predict(model, dm.test_dataloader())
            df[f"preds_{model_name}_fold{idx}"] = np.vstack(preds) * 100

    # write pseudo labels are the average prediction from
    # the different L1 models
    # NOTE: later on, we will replace the ensemble method
    # with something more powerful
    print()
    print(f"#####################")
    print(f"Creating pseudo labels")
    print(f"#####################")
    for fold_number in range(cfg.n_folds):
        preds_cols = [
            f"preds_{model_name}_fold{fold_number}"
            for model_name in cfg.models
        ]
        df[f"pseudo_label_fold{fold_number}"] = df.loc[:, preds_cols].mean(
            axis=1
        )

    # we should have 5 pseudo labels for each record, one
    # for each fold
    df.head()
    df.to_csv("data/extra.csv", index=False)


if __name__ == "__main__":
    pseudo_label()
