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
    df: pd.DataFrame, cfg: OmegaConf
) -> Union[List[Path], List[List[int]]]:

    # add image fpaths
    df["fpath"] = f"./data/train_{cfg.sz}/" + df.Id + ".jpg"

    image_paths = df.fpath.tolist()
    targets = [[t / 100.0] for t in df.Pawpularity.tolist()]
    return image_paths, targets
