"""Input preprocessing for the exported MoGe v3 graph.

The exported graph is compiled for a fixed image size (default 640x480,
width x height) and fp16 inputs, so all preprocessing is deterministic:
BGR/uint8 (OpenCV) -> RGB float in [0, 1] -> resize to the fixed size
(INTER_AREA, non-preserving if the aspect ratio differs) -> fp16 CUDA
tensor in (B, 3, H, W).
"""

import cv2
import torch


def load_image_bgr(path) -> "numpy.ndarray":
    """Read an image as BGR uint8 (same convention as moge/scripts/infer.py)."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f'Failed to read image: {path}')
    return image


def resize_to_fixed(image_bgr, width: int, height: int) -> "numpy.ndarray":
    """Resize a BGR image to exactly (width, height) using INTER_AREA."""
    if image_bgr.shape[1] != width or image_bgr.shape[0] != height:
        return cv2.resize(image_bgr, (width, height), interpolation=cv2.INTER_AREA)
    return image_bgr


def to_input_tensor(image_bgr, width: int, height: int, device: torch.device) -> torch.Tensor:
    """BGR uint8 (H, W, 3) -> fp16 CUDA tensor (B, 3, H, W) in [0, 1]."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.tensor(image_rgb / 255.0, dtype=torch.float32, device=device)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    return tensor.half()
