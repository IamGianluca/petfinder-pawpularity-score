from typing import List

import imagehash
import pandas as pd
from pandas.io.pytables import duplicate_doc
from PIL import Image

import constants


def find_similar_images(userpaths, hashfunc=imagehash.average_hash):
    image_filenames = userpaths.glob("*.jpg")

    images = {}
    for img in sorted(image_filenames):
        try:
            hash = hashfunc(Image.open(img))
        except Exception as e:
            print("Problem:", e, "with", img.stem)
            continue
        if hash in images:
            print(img.stem, "  already exists as", " ".join(images[hash]))
        images[hash] = images.get(hash, []) + [img.stem]

    duplicates = []
    for k, v in images.items():
        if len(v) > 1:
            duplicates.append(v)

    return duplicates


def flag_duplicates_and_impute_mean(duplicates: List[List], df: pd.DataFrame):
    """For every duplicate pair/triplet in the training dataset, keep one image
    and replace the target value to the average of the original images.
    """
    df["keep"] = 1
    print(f"There are {len(duplicates)} duplicate images")

    for pair in duplicates:
        pawpularity = df[df.Id.isin(pair)].Pawpularity.mean()
        df.loc[df.Id.isin(pair), "Pawpularity"] = int(pawpularity)
        df.loc[df.Id.isin(pair[1:]), "keep"] = 0
    return df.loc[df["keep"] == 1, :]


if __name__ == "__main__":
    df = pd.read_csv(constants.train_labels_fpath)
    print(f"Original DataFrame shape: {df.shape}")
    duplicates = find_similar_images(
        constants.train_images_path, imagehash.phash
    )
    df = flag_duplicates_and_impute_mean(duplicates=duplicates, df=df)
    print(f"New DataFrame shape: {df.shape}")
    df.to_csv(constants.train_deduped_fpath)
