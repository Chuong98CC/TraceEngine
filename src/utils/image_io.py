# -*- coding: utf-8 -*-
"""
Shared image-input handling for model runtimes.

All runtimes accept images in several forms — a path to an image file, a PIL
image, a numpy array or a torch tensor. ``to_image_tensor`` normalizes these
into a single CHW tensor so each model deals with one input type; see also
``video_io`` for video-frame helpers.
"""

import os
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2

# Image formats accepted by model runtimes: a path to an image file, a PIL
# image, an RGB numpy array (HWC, uint8) or a CHW tensor (uint8 [0, 255] or
# float [0, 1]).
ImageInput = Union[str, os.PathLike, Image.Image, np.ndarray, torch.Tensor]


def to_image_tensor(image: ImageInput, mode: Optional[str] = "RGB") -> torch.Tensor:
    """Convert any ImageInput to a CHW torch.Tensor.

    PIL images and numpy arrays become uint8 [0, 255] (via
    ``v2.functional.to_image``); torch tensors pass through untouched, so both
    uint8 [0, 255] and float [0, 1] CHW tensors are accepted. ``mode`` is
    applied with ``PIL.Image.convert`` when the source is a PIL image (default
    "RGB"; pass None to keep the source's own mode, e.g. for depth maps).
    """
    if isinstance(image, (str, os.PathLike)):
        image = Image.open(image)
    if isinstance(image, Image.Image):
        if mode is not None:
            image = image.convert(mode)
        return v2.functional.to_image(image)
    elif isinstance(image, np.ndarray):
        if not image.flags.writeable:
            # np.asarray on a decoded PIL image can yield a read-only view;
            # torch.from_numpy (inside to_image) warns on non-writable input
            image = image.copy()
        return v2.functional.to_image(image)
    elif isinstance(image, torch.Tensor):
        if image.ndim != 3:
            raise ValueError(f"image must be CHW, got shape {tuple(image.shape)}")
        return image
    raise TypeError(
        "image must be a path, PIL.Image, numpy.ndarray (HWC) or "
        f"torch.Tensor (CHW); got {type(image).__name__}"
    )
