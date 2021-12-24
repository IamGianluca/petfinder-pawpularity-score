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
    df: pd.DataFrame,
    cfg: OmegaConf,
    include_extra: bool = False,
    target_format: str = "binary_classification",
) -> Union[List[Path], List[List[int]]]:

    # add image fpaths
    df["fpath"] = f"./{cfg.train_data}/" + df.Id + ".jpg"

    image_paths = df.fpath.tolist()
    targets = df.Pawpularity.tolist()

    if include_extra:
        extra_df = pd.read_csv(constants.extra_labels_fpath)
        extra_df["fpath"] = f"./data/extra_{cfg.sz}/" + extra_df.Id + ".jpg"
        extra_image_paths = extra_df.fpath.tolist()
        image_paths += extra_image_paths

        extra_targets = extra_df[f"pseudo_label_fold{cfg.fold}"].tolist()
        targets += extra_targets

    if target_format == "binary_classification":
        targets = [[t / 100.0] for t in targets]
    elif target_format == "multiclass_classification":
        targets = pd.cut(targets, bins=10, retbins=True, labels=False)[
            0
        ].tolist()
    elif target_format == "regression":
        pass
    else:
        raise ValueError(f"target_format {target_format} not supported yet.")

    return image_paths, targets
