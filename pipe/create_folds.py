import numpy as np
import pandas as pd
from ml.params import load_cfg
from sklearn.model_selection import StratifiedKFold

import constants
import utils


def create_folds(cfg):
    for n_folds in cfg.k:
        df = pd.read_csv(constants.train_deduped_fpath)

        # create bins for target variable
        num_bins = int(np.floor(1 + np.log2(len(df))))
        df.loc[:, "bins"] = pd.cut(
            df["Pawpularity"], bins=num_bins, labels=False
        )

        # assign records to folds
        df["kfold"] = -1
        skf = StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=cfg.seed,
        )
        for fold_number, (_, val_idx) in enumerate(
            skf.split(X=df, y=df.bins.values)
        ):
            df.loc[val_idx, "kfold"] = fold_number

        # remove unused column
        df = df.drop(["bins"], axis=1)

        print(df.groupby("kfold").mean())
        df.to_csv(utils.train_folds_fpath[n_folds], index=False)


if __name__ == "__main__":
    cfg = load_cfg(fpath=str(constants.cfg_fpath), cfg_name="create_folds")
    create_folds(cfg=cfg)
