"""WAFT optical flow inference — ONNX and TensorRT backends.

Each concrete class inherits :class:`WAFTBase` (image geometry) plus a
backend-specific base (:class:`ONNXModel` or :class:`TRTModel`).  Tensor
format conversion (HWC uint8 → NCHW float32) is owned by each subclass's
:meth:`run` so the base stays backend-agnostic.
"""

from __future__ import annotations

import numpy as np

from base.base_onnx import ONNXModel
from base.base_trt import TRTModel
from flow_models.waft.base_waft import WAFTBase

# ---------------------------------------------------------------------------
# Shared tensor conversion
# ---------------------------------------------------------------------------


def _to_nchw_f32(img: np.ndarray) -> np.ndarray:
    """HWC uint8 → (1, 3, H, W) float32 contiguous array."""
    return np.ascontiguousarray(img.transpose(2, 0, 1))[np.newaxis].astype(np.float32)


# ---------------------------------------------------------------------------
# ONNX backend
# ---------------------------------------------------------------------------


class WAFTOnnx(WAFTBase, ONNXModel):
    """WAFT optical flow inference backed by ONNX Runtime.

    Parameters
    ----------
    onnx_path : str
        Path to the ``.onnx`` model file.
    device : str
        ``"cuda"`` (CUDA EP with CPU fallback) or ``"cpu"``.
    bgr_input : bool
        If ``True`` (default), input images are BGR and will be converted
        to RGB internally.
    """

    def __init__(
        self, onnx_path: str, device: str = "cuda", bgr_input: bool = True
    ) -> None:
        ONNXModel.__init__(self, onnx_path, device)
        WAFTBase.__init__(self, bgr_input=bgr_input)

    def run(self, img1: np.ndarray, img2: np.ndarray) -> dict[str, np.ndarray]:
        """Convert padded HWC images → NCHW float32 feed → ONNX inference."""
        feed = {"image1": _to_nchw_f32(img1), "image2": _to_nchw_f32(img2)}
        return ONNXModel.run(self, feed)


# ---------------------------------------------------------------------------
# TensorRT backend
# ---------------------------------------------------------------------------


class WAFT(WAFTBase, TRTModel):
    """WAFT optical flow inference backed by TensorRT.

    Parameters
    ----------
    engine_path : str
        Path to the ``.engine`` file.
    bgr_input : bool
        If ``True`` (default), input images are BGR and will be converted
        to RGB internally.
    """

    def __init__(self, engine_path: str, bgr_input: bool = True) -> None:
        TRTModel.__init__(self, engine_path)
        WAFTBase.__init__(self, bgr_input=bgr_input)

    def run(self, img1: np.ndarray, img2: np.ndarray) -> dict[str, np.ndarray]:
        """Convert padded HWC images → NCHW float32 feed → TRT inference."""
        feed = {"image1": _to_nchw_f32(img1), "image2": _to_nchw_f32(img2)}
        return self._run(feed, np_output=True)