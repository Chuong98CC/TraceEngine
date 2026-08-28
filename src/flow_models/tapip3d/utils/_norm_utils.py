# Vendored from <repo>/export_onnx/correlation_utils.py @ 32aeb94 (verbatim subset:
# EPS, nanvar, nanstd, compute_normalization_stats, normalize_coords,
# denormalize_coords, build_projector; lines 14-83 of the source). Import
# rewiring only: the export-time-only helpers (interpolate_time_embed,
# ModelParams, derive_model_params, assemble_updater_input) and their imports
# (torch.nn.functional, dataclass, posenc/bilinear_sampler re-exports) are not
# vendored. The vendored functions use no torch.nn.functional.
# Copyright (c) TAPIP3D team(https://tapip3d.github.io/)
"""Python-side math helpers for the ONNX inference wrapper.

These mirror the non-exportable parts of PointTracker3D / KNNCorrFeature4D_Optimized
exactly, so ONNX results are numerically equivalent to the PyTorch pipeline.
"""
from typing import Callable, Tuple
import torch

EPS = 1e-6

def nanvar(tensor, dim=None, keepdim=False):
    # Verbatim copy of the shim in models/point_tracker_3d.py (torch.nanstd does
    # not exist; this is the exact math the pipeline uses for normalization).
    tensor_mean = tensor.nanmean(dim=dim, keepdim=True)
    output = (tensor - tensor_mean).square().nanmean(dim=dim, keepdim=keepdim)
    return output

def nanstd(tensor, dim=None, keepdim=False):
    output = nanvar(tensor, dim=dim, keepdim=keepdim)
    output = output.sqrt()
    return output

def compute_normalization_stats(pcds: torch.Tensor, norm_mode: str = "isotropic") -> Tuple[torch.Tensor, torch.Tensor]:
    """Mirror PointTracker3D._wrapped_forward_window (lines 355-380).

    pcds: (B, T_w, 3, H, W) with invalid entries already set to NaN.
    Returns mean_coords (B, 3), std_coords (B, 3).
    """
    assert norm_mode in ["anisotropy", "isotropic"], norm_mode
    if norm_mode == "anisotropy":
        mean_coords = torch.nanmean(pcds, dim=(1, 3, 4), keepdim=True)  # (B, 1, 3, 1, 1)
        std_coords = nanstd(pcds, dim=(1, 3, 4), keepdim=True)
    else:
        mean_coords = torch.nanmean(pcds, dim=(1, 3, 4), keepdim=True)
        std_coords = nanstd(pcds - mean_coords, dim=(1, 2, 3, 4), keepdim=True).expand(-1, -1, 3, -1, -1)
    return mean_coords.reshape(-1, 3), std_coords.reshape(-1, 3)

def normalize_coords(coords: torch.Tensor, mean_coords: torch.Tensor, std_coords: torch.Tensor, norm_scale: float) -> torch.Tensor:
    return (coords - mean_coords[:, None, :]) / std_coords[:, None, :] * norm_scale

def denormalize_coords(coords: torch.Tensor, mean_coords: torch.Tensor, std_coords: torch.Tensor, norm_scale: float) -> torch.Tensor:
    return (coords / norm_scale * std_coords[:, None, :]) + mean_coords[:, None, :]

def build_projector(
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    mean_coords: torch.Tensor,
    std_coords: torch.Tensor,
    norm_scale: float,
    image_size: Tuple[int, int],
    norm_mode: str = "isotropic",
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a projector mirroring PointTracker3D._project.

    Input points: (B, N, 4) = (t, x, y, z) in normalized space.
    Output: (B, N, 3) = (clamped_x, clamped_y, camera_z).
    """
    def projector(points: torch.Tensor) -> torch.Tensor:
        B, N, _ = points.shape
        time_index = points[..., 0].long()
        coords = points[..., 1:]
        batch_index = torch.arange(B, device=points.device)[:, None].expand(-1, N)
        K = intrinsics[batch_index, time_index]   # B, N, 3, 3
        E = extrinsics[batch_index, time_index]   # B, N, 4, 4

        if norm_mode in ["anisotropy", "isotropic"]:
            coords = (coords / norm_scale * std_coords[:, None, :]) + mean_coords[:, None, :]

        coords_homo = torch.cat([coords, torch.ones_like(coords[..., :1])], dim=-1)
        coords_local_homo = torch.einsum("bnij,bnj->bni", E, coords_homo)
        coords_local = coords_local_homo[..., :3] / torch.clamp(coords_local_homo[..., 3:], min=EPS)
        coords_pixel = torch.einsum("bnij,bnj->bni", K, coords_local)
        coords_pixel = coords_pixel[..., :2] / torch.clamp(coords_pixel[..., 2:3], min=EPS)

        clamped_x = torch.clamp(coords_pixel[..., 0], min=-image_size[1] * 2, max=image_size[1] * 2)
        clamped_y = torch.clamp(coords_pixel[..., 1], min=-image_size[0] * 2, max=image_size[0] * 2)
        return torch.cat([clamped_x[..., None], clamped_y[..., None], coords_local[..., 2:]], dim=-1)
    return projector
