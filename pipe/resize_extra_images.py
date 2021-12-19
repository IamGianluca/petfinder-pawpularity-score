from pathlib import Path

import ml.vision.transform as transform
from ml.params import load_cfg
from omegaconf import OmegaConf

import constants


def main():
    cfg = load_cfg(
        constants.cfg_fpath,
        cfg_name=f"resize_extra_images",
    )
    cfg = OmegaConf.create(cfg)

    for img_size in cfg.sz:
        for in_path in [constants.extra_images_path]:
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
