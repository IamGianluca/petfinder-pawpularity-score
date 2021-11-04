from pathlib import Path

import ml.vision.transform as transform
import yaml

import constants

with open("params.yaml", "r") as fd:
    params = yaml.safe_load(fd)

desired_img_sizes = params["resize"]["sz"]


def main():
    for img_size in desired_img_sizes:
        paths = [
            constants.train_images_path,
            constants.test_images_path,
        ]
        for in_path in paths:
            out_path = Path(f"{in_path}_{img_size}")

            if in_path == out_path:
                raise ValueError("in_path and out_path cannot be the same.")

            if not out_path.exists():
                out_path.mkdir(parents=True)

            transform.resize_images_from_folder(
                in_path=in_path,
                out_path=out_path,
                sz=img_size,
            )


if __name__ == "__main__":
    main()
