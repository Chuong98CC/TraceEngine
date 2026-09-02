"""Image I/O and geometric pre-processing for PT2-based WAFT inference.

The letterbox logic mirrors ``model/waft_base.py`` (used by the ONNX /
TensorRT backends) so that all WAFT inference backends share identical
pre-processing: aspect-preserving resize + centre-pad to the model's
``(target_h, target_w)`` input geometry.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import torch


def load_image(img) -> np.ndarray:
    """Return a uint8 H×W×3 image from a path or an existing array.

    Images read from disk are BGR (OpenCV convention); ``bgr_to_rgb`` must
    be applied before feeding the model unless ``bgr_input=False``.
    """
    if isinstance(img, str):
        if not os.path.isfile(img):
            raise FileNotFoundError(f"Image not found: {img}")
        arr = cv2.imread(img)
        if arr is None:
            raise FileNotFoundError(
                f"Could not read image (check format / permissions): {img}"
            )
        return arr
    if isinstance(img, np.ndarray):
        return img
    raise TypeError(
        f"Expected str (path) or np.ndarray, got {type(img).__name__}"
    )


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    """Convert a BGR H×W×3 uint8 image to RGB."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def letterbox(img: np.ndarray, target_h: int, target_w: int) -> tuple[np.ndarray, dict]:
    """Aspect-preserving resize + centre-pad to ``(target_h, target_w)``.

    Mirrors ``WAFTBase._letterbox`` / ``TRTModel.resize_img`` so the metadata
    dict can be consumed by :func:`infer_pt2.utils.flow_utils.postprocess_flow`
    regardless of backend.

    Args:
        img: uint8 H×W×3 (RGB) image.
        target_h, target_w: Model input geometry.

    Returns:
        tuple: (img_padded, meta) where ``img_padded`` has shape
        (target_h, target_w, 3) and ``meta`` holds the inverse mapping
        (orig size, scale factor, tile size, pad offsets).
    """
    orig_h, orig_w = img.shape[:2]

    # Scale so the largest dimension fits ; floor to 2 decimals.
    raw_scale = min(target_w / orig_w, target_h / orig_h)
    scale_factor = np.floor(raw_scale * 100.0) / 100.0
    if scale_factor <= 0:
        scale_factor = raw_scale

    new_w = int(orig_w * scale_factor)
    new_h = int(orig_h * scale_factor)
    img_resized = cv2.resize(img, (new_w, new_h))

    pad_w = target_w - new_w
    pad_h = target_h - new_h
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    img_padded = cv2.copyMakeBorder(
        img_resized,
        pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )

    meta = {
        "orig_h": orig_h,
        "orig_w": orig_w,
        "scale_factor": float(scale_factor),
        "tile_h": new_h,
        "tile_w": new_w,
        "pad_top": int(pad_top),
        "pad_left": int(pad_left),
    }
    return img_padded, meta


def to_model_input(img: np.ndarray, dtype: torch.dtype, device: str) -> torch.Tensor:
    """Convert an H×W×3 image to the backend's (1, 3, H, W) tensor layout.

    Args:
        img: uint8 (or float) H×W×3 array in [0, 255] (RGB).
        dtype: Output tensor dtype (bfloat16 for the pt2 artifact).
        device: Output tensor device, e.g. ``"cuda"`` or ``"cpu"``.

    Returns:
        torch.Tensor: ``[1, 3, H, W]`` on ``device`` with the given dtype.
    """
    arr = np.ascontiguousarray(img.transpose(2, 0, 1))[np.newaxis]
    return torch.from_numpy(arr).to(device=device, dtype=dtype)
