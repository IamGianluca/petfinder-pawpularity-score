import pandas as pd
import yaml
from ml.params import load_cfg
from sklearn.model_selection import StratifiedKFold

import constants


def split():
    cfg = load_cfg(fpath=str(constants.cfg_fpath), cfg_name="create_folds")
    df = pd.read_csv(constants.train_labels_fpath)

    df["kfold"] = -1
    df = df.sample(frac=1).reset_index(drop=True)

    skf = StratifiedKFold(n_splits=cfg.k, shuffle=True, random_state=cfg.seed)
    for fold_number, (train_idx, val_idx) in enumerate(
        skf.split(X=df, y=df.loc[:, constants.target_col])
    ):
        df.loc[val_idx, "kfold"] = fold_number

    print(df.groupby("kfold").mean())

    df.to_csv(constants.train_folds_fpath, index=False)


if __name__ == "__main__":
    split()
