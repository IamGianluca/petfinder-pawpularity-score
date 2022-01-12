import json

import joblib
import numpy as np
import pandas as pd
from ml.params import load_cfg
from omegaconf import OmegaConf
from pytorch_lightning.utilities import seed
from scipy.stats import skew
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import PredefinedSplit

import constants


def ensemble(cfg):
    seed.seed_everything(seed=cfg.seed, workers=False)

    # reconstruct target
    df = pd.read_csv(constants.train_folds_fpath)
    y_true = []
    for i in range(cfg.n_folds):
        y_true.extend(df[df.kfold == i].Pawpularity.values.tolist())
    y_true = np.array(y_true)

    # create training set for meta-learner by using OOF predictions from
    # L1 models
    x_train = pd.DataFrame()

    for model in cfg.models:
        x_train[f"preds_{model}"] = np.load(f"preds/model_{model}_oof.npy")

    # add statistical properties of OOF predictions
    preds_cols = x_train.columns
    x_train["preds_min"] = x_train[preds_cols].min(axis=1)
    x_train["preds_max"] = x_train[preds_cols].max(axis=1)
    x_train["preds_range"] = x_train["preds_max"] - x_train["preds_min"]
    x_train["preds_mean"] = x_train[preds_cols].mean(axis=1)
    x_train["preds_median"] = np.median(x_train[preds_cols], axis=1)
    x_train["preds_std"] = x_train[preds_cols].std(axis=1)
    x_train["preds_skew"] = skew(x_train[preds_cols], axis=1)

    kfolds = []
    meta_data = []
    for i in range(cfg.n_folds):
        kfolds.extend(df[df.kfold == i].kfold.values.tolist())
        meta_data.extend(
            df[df.kfold == i][
                [
                    "Subject Focus",
                    "Eyes",
                    "Face",
                    "Near",
                    "Action",
                    "Accessory",
                    "Group",
                    "Collage",
                    "Human",
                    "Occlusion",
                    "Info",
                    "Blur",
                ]
            ].values
        )
    meta_data = np.array(meta_data)
    x_train.loc[
        :,
        [
            "Subject Focus",
            "Eyes",
            "Face",
            "Near",
            "Action",
            "Accessory",
            "Group",
            "Collage",
            "Human",
            "Occlusion",
            "Info",
            "Blur",
        ],
    ] = meta_data

    # train meta-learner
    cv = PredefinedSplit(kfolds)
    lr = LassoCV(
        fit_intercept=True,
        normalize=True,
        cv=cv,
        random_state=cfg.seed,
    )
    lr.fit(X=x_train, y=y_true)
    joblib.dump(lr, f"ckpts/model_{cfg.name}.joblib")

    y_pred = lr.predict(X=x_train)

    # compute ensemble score
    ensemble_metric = mean_squared_error(y_true, y_pred, squared=False)
    print(f"rmse test: {ensemble_metric}")
    print()
    for idx, col in enumerate(x_train.columns):
        print(f"{col}: {lr.coef_[idx]}")

    data = {}
    data["test rmse"] = round(float(ensemble_metric), 4)
    with open(f"metrics/model_{cfg.name}.json", "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    cfg = load_cfg(
        constants.cfg_fpath,
        cfg_name=f"ensemble_final",
    )
    cfg = OmegaConf.create(cfg)
    ensemble(cfg)
