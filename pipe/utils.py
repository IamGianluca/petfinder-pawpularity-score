from pathlib import Path
from typing import List, Union

import pandas as pd
from omegaconf.omegaconf import OmegaConf

import constants

train_folds_fpath = {
    5: constants.train_5folds_fpath,
    10: constants.train_10folds_fpath,
}


def get_image_paths_and_targets(
    df: pd.DataFrame, cfg: OmegaConf, include_extra: bool = False
) -> Union[List[Path], List[List[int]]]:

    # add image fpaths
    df["fpath"] = f"./data/train_{cfg.sz}/" + df.Id + ".jpg"

    image_paths = df.fpath.tolist()
    targets = [[t / 100.0] for t in df.Pawpularity.tolist()]

    if include_extra:
        extra_df = pd.read_csv(constants.extra_labels_fpath)
        extra_df["fpath"] = f"./data/extra_{cfg.sz}/" + extra_df.Id + ".jpg"
        extra_image_paths = extra_df.fpath.tolist()
        image_paths += extra_image_paths

        extra_targets = [
            [t / 100.0]
            for t in extra_df[f"pseudo_label_fold{cfg.fold}"].tolist()
        ]
        targets += extra_targets

    return image_paths, targets
