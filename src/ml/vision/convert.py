from pathlib import Path

import numpy as np
import pydicom
from PIL import Image
from pydicom.pixel_data_handlers.util import apply_voi_lut
from tqdm import tqdm


def convert_dicom2jpg(in_path, out_path):
    # TODO: extend to support different input/output type
    print(f"Converting images in {in_path} to JPG...")

    img_paths = in_path.rglob("*.dcm")
    for img_path in tqdm(list(img_paths)):
        img_array = dicom_to_numpy(
            img_path=img_path, voi_lut=True, fix_monochrome=True
        )
        img = Image.fromarray(img_array)
        img.save(out_path / f"{img_path.stem}.jpg")


def dicom_to_numpy(
    img_path: Path, voi_lut: bool = True, fix_monochrome: bool = True
):
    # credits: https://www.kaggle.com/raddar/convert-dicom-to-np-array-the-correct-way
    dicom = pydicom.read_file(img_path)

    # VOI LUT (if available by DICOM device) is used to transform
    # raw DICOM data to "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array

    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data

    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)

    return data
