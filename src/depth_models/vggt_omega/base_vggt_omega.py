"""Backend-agnostic VGGT-Omega pre/post-processing mixin.

Sibling of ``da3.base_da3.BaseDA3Model`` and ``s2m2.base_s2m2.BaseS2M2``: holds
the letterbox preprocessing, input-feed assembly, and pose/depth post-processing
shared by every VGGT-Omega inference wrapper.  It is a pure mixin — it defines no
``__init__`` and relies on ``self.target_h`` / ``self.target_w`` and the
name-keyed ``self.inputs`` metadata supplied by the backend base (``TRTModel`` or
``ONNXModel``).

Each backend supplies ``_forward(feed) -> {name: np.ndarray}``:
``TRTModel`` via ``_run`` (numpy → engine-dtype CUDA tensors internally),
``ONNXModel`` via ``run`` (numpy cast to the declared input dtypes).  Both consume
the same float32 RGB [0,1] ``(1, N, 3, H, W)`` input this mixin produces.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def quat_to_mat(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Quaternion Order: XYZW or say ijkr, scalar-last

    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part last,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    i, j, k, r = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def encoding_to_camera(pose_encoding, image_size_hw, build_intrinsics=True):
    """Decode VGGT-Omega pose encoding into extrinsics and intrinsics."""
    T = pose_encoding[..., :3]
    quat = pose_encoding[..., 3:7]
    fov_h = pose_encoding[..., 7]
    fov_w = pose_encoding[..., 8]

    R = quat_to_mat(quat)
    extrinsics = torch.cat([R, T[..., None]], dim=-1)

    intrinsics = None
    if build_intrinsics:
        H, W = image_size_hw
        fy = (H / 2.0) / torch.tan(fov_h / 2.0)
        fx = (W / 2.0) / torch.tan(fov_w / 2.0)

        intrinsics = torch.zeros(pose_encoding.shape[:-1] + (3, 3), device=pose_encoding.device)
        intrinsics[..., 0, 0] = fx
        intrinsics[..., 1, 1] = fy
        intrinsics[..., 0, 2] = W / 2
        intrinsics[..., 1, 2] = H / 2
        intrinsics[..., 2, 2] = 1.0

    return extrinsics, intrinsics


def _letterbox_preprocess(
    image: np.ndarray, target_w: int, target_h: int
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Letterbox uint8 RGB (H0,W0,3) into a zero-padded float32 [0,1] canvas.

    Scale s = min(target_w/W0, target_h/H0) preserves aspect (upscales small
    images too), content is pasted centered; returns (canvas, (ox, oy, h, w)).
    """
    h0, w0 = image.shape[:2]
    scale = min(target_w / w0, target_h / h0)
    w, h = round(w0 * scale), round(h0 * scale)
    if (w, h) == (w0, h0):
        resized = image
    else:
        resized = np.asarray(
            Image.fromarray(image, mode="RGB").resize((w, h), Image.Resampling.BICUBIC)
        )
    canvas = np.zeros((target_h, target_w, 3), dtype=np.float32)
    ox, oy = (target_w - w) // 2, (target_h - h) // 2
    canvas[oy : oy + h, ox : ox + w] = resized.astype(np.float32) / 255.0
    return canvas, (ox, oy, h, w)


def _adjust_intrinsics(intrinsics: np.ndarray, crops: np.ndarray) -> np.ndarray:
    """Shift the principal point by each frame's letterbox offset.

    intrinsics: (N,3,3); crops: (N,4) int [ox, oy, h', w'].
    """
    adjusted = intrinsics.copy()
    adjusted[:, 0, 2] -= crops[:, 0].astype(np.float64)
    adjusted[:, 1, 2] -= crops[:, 1].astype(np.float64)
    return adjusted


def _crop_spatial(
    output: np.ndarray, crops: np.ndarray
) -> np.ndarray | list[np.ndarray]:
    """Crop (N,H,W) per frame; stacked ndarray when crops are uniform, else list."""
    ox, oy, h, w = (int(v) for v in crops[0])
    if (crops == crops[:1]).all():
        return output[:, oy : oy + h, ox : ox + w]
    return [
        output[i, oy_i : oy_i + h_i, ox_i : ox_i + w_i]
        for i, (ox_i, oy_i, h_i, w_i) in enumerate(crops)
    ]


def _load_rgb_uint8(image: str | Path | np.ndarray) -> np.ndarray:
    if isinstance(image, (str, Path)):
        with Image.open(image) as im:
            return np.asarray(im.convert("RGB"), dtype=np.uint8)
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3 or arr.dtype != np.uint8:
        raise ValueError(
            f"Expected uint8 (H, W, 3) RGB array, got shape {arr.shape}, dtype {arr.dtype}"
        )
    return arr


class BaseVGGTOmega:
    """Shared VGGT-Omega preprocessing + pose/depth post-processing.  Requires
    ``self.target_h/target_w``, ``self.inputs`` and ``self._forward`` from the
    concrete wrapper (backend base + this mixin)."""

    @property
    def width(self) -> int | None:
        return self.target_w

    @property
    def height(self) -> int | None:
        return self.target_h

    def infer(self, images: list[str | Path] | list[np.ndarray]) -> dict[str, np.ndarray]:
        """Run inference on image paths or uint8 RGB arrays.

        The static image size comes from the backend base's resolved geometry;
        images are letterboxed to it and spatial outputs are cropped back to the
        content region.  Mean/std normalization happens inside the model, so
        inputs only need to be float32 RGB in [0,1] (done here).

        Returns ``{pose_enc, extrinsic, intrinsic, depth, depth_conf, crop}``.
        """
        if not images:
            raise ValueError("At least one image is required")
        frames = [_load_rgb_uint8(image) for image in images]
        canvases, crops = zip(
            *(_letterbox_preprocess(frame, self.target_w, self.target_h) for frame in frames)
        )
        crops_np = np.asarray(crops, dtype=np.int64)  # (N, 4) [ox, oy, h', w']
        outputs = self._forward(self.build_feed(canvases))
        return self.postprocess(outputs, crops_np)

    # ---- input-feed assembly ------------------------------------------------

    def build_feed(self, canvases: list[np.ndarray]) -> dict:
        """Stack the letterboxed canvases into the model's ``(1, N, 3, H, W)``
        float32 input, keyed by the model's own input name.  Each backend casts
        to the model's expected dtype (fp16 engine / declared ONNX dtype)."""
        batch = np.stack(canvases).transpose(0, 3, 1, 2)[None]
        return {self.inputs[0]["name"]: batch}

    # ---- pose/depth post-processing -----------------------------------------

    def postprocess(
        self, outputs: dict[str, np.ndarray], crops_np: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Decode the pose encoding into extrinsics/intrinsics, shift the
        principal point by each frame's letterbox offset, and crop the spatial
        outputs back to the content region."""
        pose_enc = outputs["pose_enc"][0].astype(np.float32)
        depth = outputs["depth"][0].astype(np.float32)
        if depth.ndim == 4 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        depth_conf = outputs["depth_conf"][0].astype(np.float32)
        if depth_conf.ndim == 4 and depth_conf.shape[-1] == 1:
            depth_conf = depth_conf[..., 0]

        extrinsics, intrinsics = encoding_to_camera(
            torch.from_numpy(pose_enc), (self.target_h, self.target_w)
        )
        extrinsics = extrinsics.detach().float().cpu().numpy()
        intrinsics = _adjust_intrinsics(
            intrinsics.detach().float().cpu().numpy(), crops_np
        )

        return {
            "pose_enc": pose_enc,
            "extrinsic": extrinsics,
            "intrinsic": intrinsics,
            "depth": _crop_spatial(depth, crops_np),
            "depth_conf": _crop_spatial(depth_conf, crops_np),
            "crop": crops_np,
        }
