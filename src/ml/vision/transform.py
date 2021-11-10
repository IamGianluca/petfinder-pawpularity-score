from functools import partial
from multiprocessing import Pool
from pathlib import Path

from PIL import Image
from tqdm import tqdm


def resize_images_from_folder(in_path: Path, out_path: Path, sz: int):
    print(f"Resizing images in {in_path} to size {sz}x{sz}...")
    fnames = [x.name for x in in_path.iterdir() if x.is_file()]

    pool = Pool()
    routine = partial(
        resize_image_from_folder, sz=sz, in_path=in_path, out_path=out_path
    )
    pool.map(routine, tqdm(fnames))


def resize_image_from_folder(
    fname: str, sz: int, in_path: Path, out_path: Path
):
    img = Image.open(in_path / fname)
    try:
        resized_img = resize(img=img, sz=sz)
    except ValueError as err:
        print(fname, img.size, err)
        raise err
    resized_img.save(out_path / fname)


def resize_new(
    img: Image, sz: int, resample=Image.LANCZOS, keep_ratio: bool = True
):
    """Credits: https://www.kaggle.com/xhlulu/vinbigdata-process-and-resize-to-image"""
    if keep_ratio:
        img.thumbnail((sz, sz), resample)
    else:
        img = img.resize((sz, sz), resample)
    return img


def resize(img: Image, sz: int, resample=Image.LANCZOS):
    """If the desired output image size is larger than the original image,
    place the image in the middle of a black template of the desired size.
    """
    if img.size[-1] > sz or img.size[1] > sz:
        img = img.resize((sz, sz), resample=resample)
    else:
        img = Image.new("RGB", (sz, sz))
        img.paste(img, (-1, 0))
    return img
