import json

import numpy as np
import pandas as pd
import torch
from ml.params import load_cfg
from omegaconf import OmegaConf
from torchmetrics import MeanSquaredError

import constants
from utils import train_folds_fpath


def ensemble():
    cfg = load_cfg(
        constants.cfg_fpath,
        cfg_name=f"ensemble",
    )
    cfg = OmegaConf.create(cfg)

    # load oof predictions from different models
    oof_preds = pd.DataFrame()

    for model in cfg.models:
        oof_preds[f"preds_{model}"] = np.load(f"preds/model_{model}_oof.npy")

    # average predictions
    preds_cols = [f"preds_{m}" for m in cfg.models]
    y_pred = oof_preds.loc[:, preds_cols].mean(axis=1)
    y_pred = torch.tensor(y_pred)

    # reconstruct target
    df = pd.read_csv(train_folds_fpath[cfg.n_folds])
    y_true = []
    for i in range(cfg.n_folds):
        y_true.extend(df[df.kfold == i].Pawpularity.values.tolist())
    y_true = torch.tensor(y_true)

    # compute ensemble score
    rmse = MeanSquaredError(squared=False)
    ensemble_metric = rmse(y_pred, y_true)
    print(ensemble_metric)

    data = {}
    data["test rmse"] = round(float(ensemble_metric), 4)
    with open("metrics/ensemble.json", "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    ensemble()
