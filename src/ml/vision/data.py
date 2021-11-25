import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image, ImageFile
from scipy.ndimage import zoom
from torch.utils.data import DataLoader, Dataset

# sometimes, you will have images without an ending bit; this
# takes care of those kind of (corrupt) images
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageClassificationDataset(Dataset):
    def __init__(
        self, image_paths: List[Path], targets: List = None, augmentations=None
    ) -> None:
        self.image_paths = image_paths
        self.targets = targets
        self.augmentations = augmentations
        self.length = len(image_paths)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        image = np.array(image) / 255.0

        # set channel first --> from [H, W, C] to [C, H, W]
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        if self.targets is not None:  # train/val dataset
            return image, torch.tensor(self.targets[index])
        else:  # test dataset
            return image


class Image3DClassificationDataset(Dataset):
    def __init__(
        self, image_paths: List[Path], targets: List = None, augmentations=None
    ) -> None:
        self.image_paths = image_paths
        self.targets = targets
        self.augmentations = augmentations
        self.length = len(image_paths)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index):
        # for 3D images we load each individual frame as a numpy array,
        # then using Spline Interpolated Zoom to standardize the output
        # volume's shape
        fpaths = sorted(
            list(Path(self.image_paths[index]).iterdir()),
            key=lambda x: int(re.findall(r"\d+", x.name)[0]),
        )
        images = [np.array(Image.open(fpath)) for fpath in fpaths]
        image = np.stack(images, axis=0)
        image = spline_interpolated_zoom(image, desired_depth=15)
        image = torch.tensor(image).float()

        # if self.augmentations:
        #     image = self.augmentations(image=image)["image"]
        # if image.ndim == 2:  # add channel axis to grayscale images
        #     image = image[None, ...]

        if self.targets is not None:  # train/val dataset
            return image, torch.tensor(self.targets[index])
        else:  # test dataset
            return image


def spline_interpolated_zoom(img, desired_depth: int = 3):
    """Spline Interpolated Zoom
    ref: https://github.com/hasibzunair/uniformizing-3D/blob/master/1_data_process_clef19.ipynb
    """
    current_depth = img.shape[0]
    depth = current_depth / desired_depth
    depth_factor = 1 / depth
    img_new = zoom(img, (depth_factor, 1, 1), mode="nearest")
    return img_new


class ObjectDetectionDataset(Dataset):
    def __init__(
        self,
        image_paths: List[Path],
        targets: Optional[List[Dict[str, Any]]] = None,
        augmentations=None,
    ) -> None:
        self.image_paths = image_paths
        self.targets = targets
        self.augmentations = augmentations
        self.length = len(image_paths)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index):
        image = np.array(Image.open(self.image_paths[index]).convert("RGB"))

        if self.targets:
            # TODO: move this 'multiplication to utils.get_targets()
            boxes = self.targets[index]["boxes"]
            labels = [self.targets[index]["labels"]] * boxes.shape[0]
        else:
            boxes, labels = None, None

        if self.augmentations:
            transformed = self.augmentations(
                image=image, bboxes=boxes, labels=labels
            )
            image = transformed["image"]
            boxes = transformed["bboxes"]

        # after applying data augmentation, boxes for only background
        # cases should be brought back to [[0, 0, 1, 1]]
        if labels == [0]:
            boxes = [[0, 0, 1, 1]]

        image = torch.tensor(image).float().permute(2, 0, 1) / 255.0

        if self.targets:  # train/val dataset
            target = {
                "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(
                    -1, 4
                ),
                "labels": torch.tensor(labels, dtype=torch.long),
            }
            return image, target
        else:  # test dataset
            return image


class ImageDataModule(pl.LightningDataModule):
    def __init__(
        self,
        task: str,
        batch_size: int,
        train_image_paths: List[Path] = None,
        val_image_paths: List[Path] = None,
        test_image_paths: List[Path] = None,
        train_targets: np.ndarray = None,
        val_targets: np.ndarray = None,
        train_augmentations=None,
        val_augmentations=None,
        test_augmentations=None,
    ):
        super().__init__()

        self.task = task
        self.train_image_paths = train_image_paths
        self.val_image_paths = val_image_paths
        self.test_image_paths = test_image_paths

        self.train_targets = train_targets
        self.val_targets = val_targets

        self.train_augmentations = train_augmentations
        self.val_augmentations = val_augmentations
        self.test_augmentations = test_augmentations

        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None) -> None:
        if stage:
            print(stage)
        if self.train_image_paths:
            self.train_ds = get_dataset[self.task](
                image_paths=self.train_image_paths,
                targets=self.train_targets,
                augmentations=self.train_augmentations,
            )
        if self.val_image_paths:
            self.val_ds = get_dataset[self.task](
                image_paths=self.val_image_paths,
                targets=self.val_targets,
                augmentations=self.val_augmentations,
            )
        if self.test_image_paths:
            self.test_ds = get_dataset[self.task](
                image_paths=self.test_image_paths,
                targets=None,
                augmentations=self.test_augmentations,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=12,
            pin_memory=True,
            drop_last=True,
            collate_fn=self.collate_fn if self.task == "detection" else None,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=12,
            drop_last=False,
            collate_fn=self.collate_fn if self.task == "detection" else None,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=12,
            drop_last=False,
        )

    def collate_fn(self, batch):
        return tuple(zip(*batch))


get_dataset = {
    "classification": ImageClassificationDataset,
    "3Dclassification": Image3DClassificationDataset,
    "detection": ObjectDetectionDataset,
}
