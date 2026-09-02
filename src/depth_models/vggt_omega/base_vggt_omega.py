"""Backend-agnostic VGGT-Omega pre/post-processing mixin.

Holds the letterbox preprocessing, input-feed assembly, and pose/depth
post-processing shared by the VGGT-Omega inference wrapper.  It is a pure
mixin — it defines no ``__init__`` and relies on ``self.target_h`` /
``self.target_w`` and the name-keyed ``self.inputs`` metadata supplied by
the backend base.  Images are decoded through the shared ``utils.image_io``
helpers (PIL / torchvision).

The concrete wrapper supplies ``_forward(feed) -> {name: np.ndarray}``: it
receives the CPU float32 RGB [0,1] ``(1, N, 3, H, W)`` tensor feed
``build_feed`` assembles, keyed by the model's own input name, and casts it
to the model's expected dtype (the torch.export wrapper moves it to cuda as
fp16/fp32).  Outputs stay numpy.
"""

from __future__ import annotations

import numpy as np
import torch
from torchvision.transforms import v2

from utils.image_io import (
    ImageInput,
    letterbox,
    to_image_tensor,
    to_pixel_uint8,
)


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

    def _load_image(self, image: ImageInput) -> torch.Tensor:
        """Decode any ImageInput into a CHW uint8 RGB tensor (CPU)."""
        return to_pixel_uint8(to_image_tensor(image))

    def infer(self, images: list[ImageInput]) -> dict[str, np.ndarray]:
        """Run inference on image paths, RGB arrays or tensors.

        The static image size comes from the backend base's resolved geometry;
        images are letterboxed (VGGT convention: raw-scale ``round`` tiles,
        bicubic) to it and spatial outputs are cropped back to the content
        region.  Mean/std normalization happens inside the model, so inputs
        only need to be float32 RGB in [0,1] (done here).

        Returns ``{pose_enc, extrinsic, intrinsic, depth, depth_conf, crop}``.
        """
        if not images:
            raise ValueError("At least one image is required")
        canvases, metas = zip(
            *(
                letterbox(
                    self._load_image(image),
                    self.target_h,
                    self.target_w,
                    scale_mode="round",
                    resize=v2.InterpolationMode.BICUBIC,
                    antialias=True,  # matches the legacy PIL bicubic within 1 uint8 level
                )
                for image in images
            )
        )
        # VGGT pads into float [0,1] canvases; the letterbox meta maps 1:1 onto
        # the old (ox, oy, h, w) crop tuple (pad_left, pad_top, tile_h, tile_w).
        canvases = [c.float().div(255.0) for c in canvases]
        crops_np = np.asarray(
            [
                [m["pad_left"], m["pad_top"], m["tile_h"], m["tile_w"]]
                for m in metas
            ],
            dtype=np.int64,
        )
        outputs = self._forward(self.build_feed(canvases))
        return self.postprocess(outputs, crops_np)

    def build_feed(self, canvases: list[torch.Tensor]) -> dict:
        """Stack the letterboxed CHW canvases into the model's ``(1, N, 3, H, W)``
        fp32 CPU tensor, keyed by the model's own input name.  Each backend
        casts to the model's expected dtype (fp16 export / declared ONNX
        dtype)."""
        batch = torch.stack(canvases)[None]
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
