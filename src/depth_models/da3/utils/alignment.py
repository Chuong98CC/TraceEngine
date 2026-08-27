# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Alignment utilities for depth estimation and metric scaling.
"""

from typing import Tuple
import torch
import numpy as np
from .pose_align import align_poses_umeyama  # noqa: PLC0415


def least_squares_scale_scalar(
    a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    """
    Compute least squares scale factor s such that a ≈ s * b.

    Args:
        a: First tensor
        b: Second tensor
        eps: Small epsilon for numerical stability

    Returns:
        Scalar tensor containing the scale factor

    Raises:
        ValueError: If tensors have mismatched shapes or devices
        TypeError: If tensors are not floating point
    """
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    if a.device != b.device:
        raise ValueError(f"Device mismatch: {a.device} vs {b.device}")
    if not a.is_floating_point() or not b.is_floating_point():
        raise TypeError("Tensors must be floating point type")

    # Compute dot products for least squares solution
    num = torch.dot(a.reshape(-1), b.reshape(-1))
    den = torch.dot(b.reshape(-1), b.reshape(-1)).clamp_min(eps)
    return num / den


def compute_sky_mask(sky_prediction: torch.Tensor, threshold: float = 0.3) -> torch.Tensor:
    """
    Compute non-sky mask from sky prediction.

    Args:
        sky_prediction: Sky prediction tensor
        threshold: Threshold for sky classification

    Returns:
        Boolean mask where True indicates non-sky regions
    """
    return sky_prediction < threshold


def compute_alignment_mask(
    depth_conf: torch.Tensor,
    non_sky_mask: torch.Tensor,
    depth: torch.Tensor,
    metric_depth: torch.Tensor,
    median_conf: torch.Tensor,
    min_depth_threshold: float = 1e-3,
    min_metric_depth_threshold: float = 1e-2,
) -> torch.Tensor:
    """
    Compute mask for depth alignment based on confidence and depth thresholds.

    Args:
        depth_conf: Depth confidence tensor
        non_sky_mask: Non-sky region mask
        depth: Predicted depth tensor
        metric_depth: Metric depth tensor
        median_conf: Median confidence threshold
        min_depth_threshold: Minimum depth threshold
        min_metric_depth_threshold: Minimum metric depth threshold

    Returns:
        Boolean mask for valid alignment regions
    """
    return (
        (depth_conf >= median_conf)
        & non_sky_mask
        & (metric_depth > min_metric_depth_threshold)
        & (depth > min_depth_threshold)
    )


def sample_tensor_for_quantile(tensor: torch.Tensor, max_samples: int = 100000) -> torch.Tensor:
    """
    Sample tensor elements for quantile computation to reduce memory usage.

    Args:
        tensor: Input tensor to sample
        max_samples: Maximum number of samples to take

    Returns:
        Sampled tensor
    """
    if tensor.numel() <= max_samples:
        return tensor

    idx = torch.randperm(tensor.numel(), device=tensor.device)[:max_samples]
    return tensor.flatten()[idx]


def apply_metric_scaling(
    depth: torch.Tensor, intrinsics: torch.Tensor, scale_factor: float = 300.0
) -> torch.Tensor:
    """
    Apply metric scaling to depth based on camera intrinsics.

    Args:
        depth: Input depth tensor
        intrinsics: Camera intrinsics tensor
        scale_factor: Scaling factor for metric conversion

    Returns:
        Scaled depth tensor
    """
    focal_length = (intrinsics[:, :, 0, 0] + intrinsics[:, :, 1, 1]) / 2
    return depth * (focal_length[:, :, None, None] / scale_factor)


def set_sky_regions_to_max_depth(
    depth: torch.Tensor,
    depth_conf: torch.Tensor,
    non_sky_mask: torch.Tensor,
    max_depth: float = 200.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Set sky regions to maximum depth and high confidence.

    Args:
        depth: Depth tensor
        depth_conf: Depth confidence tensor
        non_sky_mask: Non-sky region mask
        max_depth: Maximum depth value for sky regions

    Returns:
        Tuple of (updated_depth, updated_depth_conf)
    """
    depth = depth.clone()

    # Set sky regions to max depth and high confidence
    depth[~non_sky_mask] = max_depth
    if depth_conf is not None:
        depth_conf = depth_conf.clone()
        depth_conf[~non_sky_mask] = 1.0
        return depth, depth_conf
    else:
        return depth, None


#---------------------------------------------------------------
#additional functions for alignment and scaling can be added here as needed
def align_anyview_with_metric(
    anyview_depth: torch.Tensor,
    anyview_conf: torch.Tensor,
    anyview_extrinsics: torch.Tensor,
    anyview_intrinsics: torch.Tensor,
    metric_depth: torch.Tensor,
    metric_sky: torch.Tensor,
    apply_metric_scaling_step: bool = True,
) -> dict[str, torch.Tensor]:
    """Replicate ``NestedDepthAnything3Net`` alignment logic in standalone Python.

    Mirrors the three post-processing steps so they run **outside** the ONNX
    or TRT graph:

    1. Metric scaling via predicted intrinsics
    2. Least-squares depth alignment
    3. Sky-region handling

    Parameters
    ----------
    anyview_depth : ``(B, N, H, W)`` or ``(N, H, W)``
        Raw any-view depth prediction.
    anyview_conf : ``(B, N, H, W)`` or ``(N, H, W)``
        Any-view depth confidence.
    anyview_extrinsics : ``(B, N, 3, 4)`` or ``(N, 3, 4)``
        Any-view predicted camera extrinsics.
    anyview_intrinsics : ``(B, N, 3, 3)`` or ``(N, 3, 3)``
        Any-view predicted camera intrinsics.
    metric_depth : ``(B, N, H, W)`` or ``(N, H, W)``
        Raw metric-model depth (monocular, unscaled).
    metric_sky : ``(B, N, H, W)`` or ``(N, H, W)``
        Metric-model sky logits (0 = non-sky, 1 = sky).
    apply_metric_scaling_step : bool, default ``True``
        When False, step 1 is skipped: the metric depth is already metric
        metres (e.g. Any2Full's affine-fit output) and the ``focal / 300``
        rescale would corrupt its scale.

    Returns
    -------
    dict[str, torch.Tensor]
        ``depth``, ``depth_conf``, ``extrinsics``, ``intrinsics``.
    """


    # ---- step 1: metric scaling --------------------------------------------
    if apply_metric_scaling_step:
        metric_depth = apply_metric_scaling(metric_depth, anyview_intrinsics)

    # ---- step 2: least-squares scale alignment -----------------------------
    non_sky_mask = compute_sky_mask(metric_sky, threshold=0.3)
    if non_sky_mask.sum() <= 10:
        raise RuntimeError("Insufficient non-sky pixels for alignment")

    depth_conf_ns = anyview_conf[non_sky_mask]
    depth_conf_sampled = sample_tensor_for_quantile(depth_conf_ns, max_samples=100_000)
    median_conf = torch.quantile(depth_conf_sampled, 0.5)

    align_mask = compute_alignment_mask(
        anyview_conf, non_sky_mask, anyview_depth, metric_depth, median_conf,
    )

    scale_factor = least_squares_scale_scalar(
        metric_depth[align_mask], anyview_depth[align_mask],
    )

    anyview_depth = anyview_depth * scale_factor
    anyview_extrinsics = anyview_extrinsics.clone()
    anyview_extrinsics[..., :3, 3] *= scale_factor

    # ---- step 3: sky-region handling ---------------------------------------
    non_sky_depth = anyview_depth[non_sky_mask]
    if non_sky_depth.numel() > 100_000:
        idx = torch.randint(
            0, non_sky_depth.numel(), (100_000,), device=non_sky_depth.device,
        )
        sampled_depth = non_sky_depth[idx]
    else:
        sampled_depth = non_sky_depth
    non_sky_max = min(float(torch.quantile(sampled_depth, 0.99)), 200.0)

    anyview_depth, anyview_conf = set_sky_regions_to_max_depth(
        anyview_depth, anyview_conf, non_sky_mask, max_depth=non_sky_max,
    )

    return {
        "depth": anyview_depth,
        "depth_conf": anyview_conf,
        "extrinsics": anyview_extrinsics,
        "intrinsics": anyview_intrinsics,
    }


def align_to_input_ext_scale(
    pred_depth: np.ndarray,
    pred_extrinsics: np.ndarray,
    input_extrinsics: np.ndarray,
    input_intrinsics: np.ndarray,
    align_scale: bool = True,
    ransac_view_thresh: int = 10,
) -> dict[str, np.ndarray]:
    """Align a prediction to the input camera poses (numpy post-processing).

    Standalone replica of ``DepthAnything3._align_to_input_extrinsics_intrinsics``
    (``api.py``) so it can run **outside** the ONNX / TRT graph, on the outputs of
    :func:`align_anyview_with_metric`.  The Umeyama Sim(3) scale it needs (3x3 SVD,
    optional RANSAC via ``evo``) is not ONNX-exportable, hence the Python helper.

    Parameters
    ----------
    pred_depth : ``(N, H, W)``
        Predicted (metric-scaled) depth from the nested pipeline.
    pred_extrinsics : ``(N, 3, 4)`` or ``(N, 4, 4)``
        Predicted camera extrinsics (world-to-camera), in the model's frame.
    input_extrinsics : ``(N, 4, 4)``
        Original **un-normalised** input extrinsics (world-to-camera).
    input_intrinsics : ``(N, 3, 3)``
        Input intrinsics, scaled to the processing resolution.  Passed straight
        through to the output (mirrors the PyTorch behaviour).
    align_scale : bool, default ``True``
        If ``True``: output extrinsics are the input extrinsics and ``depth`` is
        divided by the Umeyama scale.  If ``False``: output extrinsics are the
        predicted poses aligned into the input frame, and depth is unchanged.
    ransac_view_thresh : int, default 10
        Use RANSAC alignment when the number of views is ``>=`` this threshold.

    Returns
    -------
    dict[str, np.ndarray]
        ``depth``, ``extrinsics`` ``(N, 3, 4)``, ``intrinsics``.
    """


    pred_extrinsics = np.asarray(pred_extrinsics, dtype=np.float64)
    input_extrinsics = np.asarray(input_extrinsics, dtype=np.float64)

    _, _, scale, aligned_extrinsics = align_poses_umeyama(
        pred_extrinsics,
        input_extrinsics,
        ransac=len(input_extrinsics) >= ransac_view_thresh,
        return_aligned=True,
        random_state=42,
    )

    out_depth = np.asarray(pred_depth).copy()
    if align_scale:
        out_extrinsics = input_extrinsics[..., :3, :].copy()
        out_depth = out_depth / scale
    else:
        out_extrinsics = aligned_extrinsics[..., :3, :]

    return {
        "depth": out_depth.astype(np.float32),
        "extrinsics": out_extrinsics.astype(np.float32),
        "intrinsics": np.asarray(input_intrinsics, dtype=np.float32).copy(),
    }