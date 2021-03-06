# credit: https://github.com/adam-mehdi/MuarAugment/blob/17964bf9cd6c8b96b8c7a306a309a0fc0c512395/muar/augmentations.py


from random import random
from typing import Union

import kornia.augmentation as K
import numpy as np
import torch

# from muar.loss import MixUpCrossEntropy
# from muar.transform_lists import albumentations_list, kornia_list
from torch import nn


def kornia_list(magn: int = 4):
    """
    Returns standard list of kornia transforms, each with magnitude `magn`.

    Args:
        magn (int): Magnitude of each transform in the returned list.
    """
    transform_list = [
        # spatial
        K.RandomHorizontalFlip(p=1),
        K.RandomRotation(degrees=90.0, p=1),
        K.RandomAffine(
            degrees=magn * 5.0, shear=magn / 5, translate=magn / 20, p=1
        ),
        # K.RandomPerspective(distortion_scale=magn / 25, p=1),
        # # pixel-level
        K.ColorJitter(brightness=magn / 30, p=1),  # brightness
        K.ColorJitter(saturation=magn / 30, p=1),  # saturation
        K.ColorJitter(contrast=magn / 30, p=1),  # contrast
        K.ColorJitter(hue=magn / 30, p=1),  # hue
        K.ColorJitter(p=0),  # identity
        # K.RandomMotionBlur(
        #     kernel_size=2 * (magn // 3) + 1, angle=magn, direction=1.0, p=1
        # ),
        K.RandomErasing(
            scale=(magn / 100, magn / 50), ratio=(magn / 20, magn), p=1
        ),
    ]
    return transform_list


class BatchRandAugment(nn.Module):
    """
    Image augmentation pipeline that applies a composition of `n_tfms` transforms
    each of magnitude `magn` sampled uniformly at random from `transform_list` with
    optional batch resizing and label mixing transforms.

    Args:
        n_tfms (int): Number of transformations sampled for each composition,
            excluding resize or label mixing transforms. N in paper.
        magn (int): Magnitude of augmentation applied. Ranges from [0, 10] with
            10 being the max magnitude. M in paper.
        mean (tuple, torch.Tensor): Mean of images after normalized in range [0,1]
        std (tuple, torch.Tensor): Mean of images after normalized in range [0,1]
        transform_list (list): List of transforms to sample from. Default list
            provided if not specified.
        use_resize (int): Batch-wise resize transform to apply. Options:
            None: Don't use.
            0: RandomResizedCrop
            1: RandomCrop
            2: CenterCrop
            3: Randomly select a resize transform per batch.
        image_size (tuple): Final size after applying batch-wise resize transforms.
        use_mix (int): Label mixing transform to apply. Options:
            None: No label mixing transforms.
            0: CutMix
            1: MixUp
        mix_p (float): probability of applying the mix transform on a batch
            given `use_mix` is not None.
    """

    def __init__(
        self,
        n_tfms: int,
        magn: int,
        mean: Union[tuple, list, torch.tensor],
        std: Union[tuple, list, torch.tensor],
        transform_list: list = None,
        use_normalize: bool = True,
        use_resize: int = None,
        image_size: tuple = None,
        use_mix: int = None,
        mix_p: float = 0.5,
    ):
        super().__init__()

        self.n_tfms, self.magn = n_tfms, magn
        self.use_mix, self.mix_p = use_mix, mix_p
        self.image_size = image_size

        if not isinstance(mean, torch.Tensor):
            mean = torch.Tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.Tensor(std)

        if self.use_mix is not None:
            self.mix_list = [
                K.RandomCutMix(self.image_size[0], self.image_size[1], p=1),
                K.RandomMixUp(p=1),
            ]

        self.use_resize = use_resize
        if use_resize is not None:
            assert (
                len(image_size) == 2
            ), "Invalid `image_size`. Must be a tuple of form (h, w)"
            self.resize_list = [
                K.RandomResizedCrop(image_size),
                K.RandomCrop(image_size),
                K.CenterCrop(image_size),
            ]
            if self.use_resize < 3:
                self.resize = self.resize_list[use_resize]

        self.use_normalize = use_normalize
        if use_normalize:
            self.normalize = K.Normalize(mean, std)

        self.transform_list = transform_list
        if transform_list is None:
            self.transform_list = kornia_list(magn)

    def setup(self):
        if self.use_resize == 3:
            self.resize = np.random.choice(self.resize_list)

        if self.use_mix is not None and random() < self.mix_p:
            self.mix = self.mix_list[self.use_mix]
        else:
            self.mix = None

        sampled_tfms = list(
            np.random.choice(self.transform_list, self.n_tfms, replace=False)
        )

        if self.use_normalize:
            sampled_tfms += [self.normalize]

        self.transform = nn.Sequential(*sampled_tfms)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, y: torch.Tensor = None):
        """
        Applies transforms on the batch.

        If `use_mix` is not `None`, `y` is required. Else, it is optional.

        If a label-mixing transform is applied on the batch, `y` is returned
        in shape `(batch_size, 3)`, in which case use the special loss function
        provided in `muar.utils` if.

        Args:
            x (torch.Tensor): Batch of input images.
            y (torch.Tensor): Batch of labels.
        """
        if self.use_resize is not None:
            x = self.resize(x)

        if self.mix is not None:
            x, y = self.mix(x, y)
            return self.transform(x), y
        elif y is None:
            return self.transform(x)
        else:
            return self.transform(x), y
