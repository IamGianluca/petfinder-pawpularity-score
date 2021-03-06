from pathlib import Path
from typing import List, Union

import pandas as pd
from omegaconf.omegaconf import OmegaConf

import constants


def get_image_paths_and_targets(
    df: pd.DataFrame,
    cfg: OmegaConf,
    include_extra: int = 0,
    ignore=False,
    target_format: str = "binary_classification",
) -> Union[List[Path], List[List[int]]]:

    # add image fpaths
    df["fpath"] = f"./{cfg.train_data}/" + df.Id + ".jpg"

    if ignore:
        df = df[df.ignore == 0]

    image_paths = df.fpath.tolist()
    targets = df.Pawpularity.tolist()

    if include_extra == 0:
        pass
    elif include_extra == 1:
        extra_df = pd.read_csv(constants.adoption_labels_fpath)
        extra_df["fpath"] = f"./data/extra/" + extra_df.Id + ".jpg"
        extra_image_paths = extra_df.fpath.tolist()
        image_paths += extra_image_paths

        extra_targets = extra_df[f"pseudo_label_fold{cfg.fold}"].tolist()
        targets += extra_targets
    elif include_extra == 2:
        extra_df = pd.read_csv(constants.dogsvscats_labels_fpath)
        extra_df["fpath"] = f"./data/extra2/" + extra_df.Id + ".jpg"
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
    elif target_format == "under_20":
        targets = [[1] if t <= 20 else [0] for t in targets]
    else:
        raise ValueError(f"target_format {target_format} not supported yet.")

    return image_paths, targets
