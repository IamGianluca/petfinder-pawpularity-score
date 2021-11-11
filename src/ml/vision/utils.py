import copy
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils as utils
from PIL.ImageDraw import Draw
from torch.utils.data import DataLoader
from torchvision import transforms


def plot_batches(dl: DataLoader, n_batches: int = 1):
    """Plot batches from classification tasks. Dataloader should
    return lists of images (tensors) and labels (tensors).
    """
    for batch_number, batch in enumerate(dl):
        plt.figure(figsize=(20, 10))
        _show_images_in_batch(batch=batch, verbose=False)
        plt.axis("off")
        plt.ioff()
        plt.show()
        if batch_number == (n_batches - 1):
            break


def _show_images_in_batch(batch: Tuple[List, List], verbose: bool = False):
    try:
        images, targets = batch
    except ValueError:
        images = batch
        targets = None

    if verbose:
        print(images.shape)
        print(targets)

    grid = utils.make_grid(images)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))


def plot_batches_detection(dl: DataLoader, n_batches: int = 1, preds=None):
    """Plot batches from detection tasks. Dataloader should
    return lists of images (tensors) and targets (dict containing
    'boxes' and 'labels').
    """
    images_with_bboxes = []
    titles = []
    true_labels = []
    pred_labels = []

    for n_batch, batch in enumerate(dl):
        images, targets = batch

        for image, target in zip(images, targets):
            boxes = target["boxes"]
            image_with_boxes = _add_bbox_to_image_tensor(
                image, boxes, color="red"
            )
            images_with_bboxes.append(image_with_boxes)
            true_labels.append(target["labels"][0])

        if n_batch == n_batches - 1:
            break

    if preds:
        # unnest predictions for list of batches of predictions
        # to list of predictions
        preds = [p for b in preds for p in b]

        original_images_with_bboxes = copy.deepcopy(images_with_bboxes)
        images_with_bboxes = []
        for image, pred in zip(original_images_with_bboxes, preds):
            pred_box = pred["boxes"]

            image = _add_bbox_to_image_tensor(
                image.permute(2, 0, 1), pred_box[:5], color="green"
            )
            images_with_bboxes.append(image)
            pred_labels.append(pred["labels"][0])

    titles = []
    if preds:
        for true_label, pred_label in zip(true_labels, pred_labels):
            titles.append(f"cls // true: {true_label}, pred: {pred_label}")
    else:
        for true_label in true_labels:
            titles.append(f"cls // true: {true_label}")

    cols = 8
    rows = int((len(images_with_bboxes) / cols) + 1)
    fig, axs = plt.subplots(rows, cols, figsize=(20, 20))

    for i, ax in enumerate(axs.ravel()):
        if i >= len(images_with_bboxes):
            fig.delaxes(ax)
        else:
            ax.set_box_aspect(aspect=1)
            ax.set_title(titles[i])
            ax.imshow(images_with_bboxes[i], aspect=1)

    plt.tight_layout()
    plt.show()


def _add_bbox_to_image_tensor(
    img: torch.Tensor, boxes: torch.Tensor, color: str = "red"
):
    """NOTE: boxes should use pascalvoc format."""
    img = transforms.ToPILImage()(img)
    draw = Draw(img)
    for box in boxes:
        draw.rectangle(box.tolist(), outline=color, width=2)
    return torch.tensor(np.array(img))
