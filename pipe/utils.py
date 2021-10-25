from pathlib import Path
from typing import List, Union

import pandas as pd


def get_image_paths_and_targets(
    df: pd.DataFrame,
) -> Union[List[Path], List[List[int]]]:
    image_paths = df.fpath.tolist()
    targets = [[t / 100.0] for t in df.Pawpularity.tolist()]
    return image_paths, targets
