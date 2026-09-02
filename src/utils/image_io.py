# -*- coding: utf-8 -*-
"""
Shared image-input handling for model runtimes.

All runtimes accept images in several forms — a path to an image file, a PIL
image, a numpy array or a torch tensor. ``to_image_tensor`` normalizes these
into a single CHW tensor so each model deals with one input type; see also
``video_io`` for video-frame helpers. ``letterbox`` / ``imagenet_normalize`` /
``to_pixel_uint8`` provide the shared pixel-space geometry for the model
runtimes.
"""

import math
import os
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
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


def to_pixel_uint8(x: torch.Tensor) -> torch.Tensor:
    """Rescale a CHW float [0,1] tensor into uint8 pixel space [0,255].

    ``to_image_tensor`` passes float [0,1] CHW tensors through untouched; this
    is the uniform "decoded image" representation the runtimes consume. uint8
    input passes through unchanged (same object).
    """
    if x.dtype == torch.uint8:
        return x
    return (x.clamp(0.0, 1.0) * 255.0).round().byte()


def letterbox(
    x: torch.Tensor,
    target_h: int,
    target_w: int,
    *,
    scale_mode: str = "trunc2",
    resize: v2.InterpolationMode = v2.InterpolationMode.BILINEAR,
    antialias: bool = False,
) -> tuple[torch.Tensor, dict]:
    """Aspect-preserving center-pad of a CHW image to ``(target_h, target_w)``.

    ``x`` is a CHW 3-channel pixel-space image (uint8, or float [0,1] —
    rescaled to uint8 on entry via :func:`to_pixel_uint8`).  Resize runs on
    uint8 (integer quantization preserved, like the legacy cv2/PIL letterboxes);
    the zero pad is centered.  Returns ``(padded uint8 CHW, meta)`` with meta
    keys ``orig_h/orig_w/scale_factor/tile_h/tile_w/pad_top/pad_left`` computed
    in float64 — bit-identical to the legacy math.

    ``scale_mode``: ``"trunc2"`` truncates the uniform scale to 2 decimals
    (DA3 / WAFT convention); ``"round"`` keeps the raw scale and rounds the
    tile size (VGGT convention).  ``resize``/``antialias`` select the
    torchvision resampling.
    """
    x = to_pixel_uint8(x)
    if x.ndim != 3 or x.shape[0] != 3:
        raise ValueError(
            f"letterbox expects a CHW 3-channel image, got shape {tuple(x.shape)}"
        )
    orig_h, orig_w = int(x.shape[1]), int(x.shape[2])
    raw_scale = min(target_w / orig_w, target_h / orig_h)
    if scale_mode == "trunc2":
        scale = math.floor(raw_scale * 100.0) / 100.0
        if scale <= 0:
            scale = raw_scale
        tile_w, tile_h = int(orig_w * scale), int(orig_h * scale)
    elif scale_mode == "round":
        scale = raw_scale
        tile_w, tile_h = round(orig_w * scale), round(orig_h * scale)
    else:
        raise ValueError(f"unknown scale_mode {scale_mode!r} (trunc2 | round)")

    pad_w, pad_h = target_w - tile_w, target_h - tile_h
    pad_left, pad_top = pad_w // 2, pad_h // 2
    resized = v2.functional.resize(
        x, (tile_h, tile_w), interpolation=resize, antialias=antialias
    )
    padded = F.pad(resized, (pad_left, pad_w - pad_left, pad_top, pad_h - pad_top))
    meta = {
        "orig_h": orig_h,
        "orig_w": orig_w,
        "scale_factor": float(scale),
        "tile_h": tile_h,
        "tile_w": tile_w,
        "pad_top": int(pad_top),
        "pad_left": int(pad_left),
    }
    return padded, meta


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def imagenet_normalize(x: torch.Tensor) -> torch.Tensor:
    """ImageNet-normalize a uint8 pixel-space CHW tensor: ``(x/255-mu)/std`` fp32."""
    f = x.float().div(255.0)
    mean = torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    return (f - mean.to(f.device)) / std.to(f.device)
