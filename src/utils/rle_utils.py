"""COCO-style run-length-encoded (RLE) mask helpers.

Masks are stored in the official COCO JSON convention (pycocotools):
``{"size": [H, W], "counts": <base64-str>}`` — compact, JSON-portable, and
directly loadable with ``pycocotools.mask.frPyObjects`` / ``decode``, so the
masks saved by the pipeline (e.g. SAM3 output in Step 3b) can be reused
across models and downstream tools.

Requires pycocotools (a main-env dependency).
"""

from __future__ import annotations

import base64

import numpy as np
import pycocotools.mask as mask_util


def encode_rle(mask: np.ndarray) -> dict:
    """Encode a boolean (H, W) mask as a COCO RLE dict with base64 counts.

    Args:
        mask: (H, W) bool array (or uint8 0/1).

    Returns:
        ``{"size": [H, W], "counts": <base64 str>}`` — json.dump-able.
    """
    mask = np.ascontiguousarray(mask).astype(np.uint8)
    rle = mask_util.encode(np.asfortranarray(mask))
    return {
        "size": [int(mask.shape[0]), int(mask.shape[1])],
        "counts": base64.b64encode(rle["counts"]).decode("ascii"),
    }


def decode_rle(rle: dict) -> np.ndarray:
    """Decode a COCO RLE dict (base64 counts) back to a (H, W) bool mask."""
    rle = dict(rle)
    rle["counts"] = base64.b64decode(rle["counts"])
    return mask_util.decode(rle).astype(bool)
