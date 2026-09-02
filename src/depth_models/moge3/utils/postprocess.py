"""Eager post-processing around the exported MoGe v3 dense graph.

The sparse 3D refiner (flex_gemm) cannot live inside the exported graph, so
it runs eagerly on the graph's intermediate outputs, then the affine
factorized point map is converted to camera space exactly as
`moge/model/v3.py` `infer()` does (focal/shift recovery, intrinsics,
depth, metric scale, masking).
"""

import torch
import torch.nn.functional as F

try:
    import utils3d_moge as utils3d
except ImportError:
    import utils3d

from .geometry_torch import recover_focal_shift


def voxelize(point_coord: torch.Tensor, depth_resolution: float):
    """Mirror `moge/model/v3.py` MoGeModel._voxelize.

    point_coord: (B, H, W, 3) at (x/z, y/z, logz). Returns
    (feats, coords, shape, logz) in fp32 with data-dependent z extent.
    """
    point_coord = point_coord.float()
    bsz, height, width, _ = point_coord.shape
    device = point_coord.device

    logz = point_coord[..., 2]
    zq = torch.round(logz * depth_resolution).long()
    z_offset = zq.amin(dim=(1, 2), keepdim=True)
    z_idx = zq - z_offset
    z_extent = z_idx.amax().item() + 1

    i = torch.arange(height, device=device, dtype=torch.long).view(1, height, 1).expand(bsz, height, width)
    j = torch.arange(width, device=device, dtype=torch.long).view(1, 1, width).expand(bsz, height, width)
    batch = torch.arange(bsz, device=device, dtype=torch.long).view(bsz, 1, 1).expand(bsz, height, width)
    coords = torch.stack([batch, i, j, z_idx], dim=-1).reshape(-1, 4).to(torch.int32)

    feats = point_coord.reshape(-1, 3)
    shape = torch.Size([bsz, height, width, z_extent, feats.shape[-1]])
    return feats, coords, shape, logz


def refine_step(refiner, point_coord: torch.Tensor, encoder_feature: torch.Tensor, depth_resolution: float) -> torch.Tensor:
    """Mirror `moge/model/v3.py` MoGeModel._refine_logz.

    Runs the sparse UNet on the voxelized point map and returns the updated
    (B, H, W) logz in fp32. Must be called under fp16 autocast (the refiner
    weights are fp16 while the voxelized feats are fp32).
    """
    feats, coords, shape, logz = voxelize(point_coord, depth_resolution)
    out = refiner(feats, coords, shape, encoder_feature)
    out_logz = out.float().squeeze(-1).reshape(*point_coord.shape[:3])
    return logz + out_logz


def refine_logz(refiner, point_coord: torch.Tensor, encoder_feature: torch.Tensor, depth_resolution: float, steps: int, device: torch.device) -> torch.Tensor:
    """Run `steps` sparse refinement updates on the (B, H, W, 3) affine point map.

    Mirrors the refine loop in `moge/model/v3.py` forward() (with
    `refiner_detach_backbone=True`, i.e. the encoder feature is detached).
    Returns the refined (B, H, W, 3) point map at (x/z, y/z, logz).
    """
    current = point_coord
    with torch.autocast(device_type=device.type, dtype=torch.float16):
        for _ in range(steps):
            refined_logz = refine_step(refiner, current, encoder_feature, depth_resolution)
            current = torch.cat([current[..., :2], refined_logz.unsqueeze(-1)], dim=-1)
    return current


def remap_points_exp(points: torch.Tensor) -> torch.Tensor:
    """Mirror MoGeModel._remap_points with remap_output='exp':
    (x/z, y/z, logz) -> (x/z * z, y/z * z, z) with z = exp(logz)."""
    xy, z = points.split([2, 1], dim=-1)
    z = torch.exp(z)
    return torch.cat([xy * z, z], dim=-1)


def resize_channel_last(x: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    """Bilinear resize of a (B, H, W, C) tensor to (B, out_h, out_w, C)."""
    return F.interpolate(
        x.movedim(-1, -3),
        (out_h, out_w),
        mode='bilinear',
        align_corners=False,
        antialias=False,
    ).movedim(-3, -1)


def affine_to_camera(
    affine_points: torch.Tensor,   # (B, H, W, 3) fp32 at (x/z*z, y/z*z, z) pre-metric-scale
    mask: torch.Tensor,            # (B, H, W) fp32 in [0, 1]
    metric_scale: torch.Tensor,    # (B,)
    aspect_ratio: float,
    fov_x: float = None,
    force_projection: bool = True,
    apply_mask: bool = True,
) -> dict:
    """Convert an affine point map to camera space, mirroring v3.infer().

    Returns {points, depth, mask, intrinsics, normal-less} as fp32 tensors.
    """
    device = affine_points.device
    mask_binary = mask > 0.5

    if fov_x is None:
        focal, shift = recover_focal_shift(affine_points, mask_binary)
    else:
        focal = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5 / torch.tan(torch.deg2rad(torch.as_tensor(fov_x, device=device, dtype=torch.float32) / 2))
        if focal.ndim == 0:
            focal = focal[None].expand(affine_points.shape[0])
        _, shift = recover_focal_shift(affine_points, mask_binary, focal=focal)
    fx = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 / aspect_ratio
    fy = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5
    intrinsics = utils3d.pt.intrinsics_from_focal_center(
        fx, fy,
        torch.tensor(0.5, device=device, dtype=torch.float32),
        torch.tensor(0.5, device=device, dtype=torch.float32),
    )

    points = affine_points.clone()
    points[..., 2] += shift[..., None, None]
    mask_binary = mask_binary & (points[..., 2] > 0) if mask_binary is not None else None
    depth = points[..., 2].clone()

    if force_projection:
        points = utils3d.pt.depth_map_to_point_map(depth, intrinsics=intrinsics)

    if metric_scale is not None:
        points = points * metric_scale[:, None, None, None]
        depth = depth * metric_scale[:, None, None]

    if apply_mask and mask_binary is not None:
        points = torch.where(mask_binary[..., None], points, torch.inf)
        depth = torch.where(mask_binary, depth, torch.inf)

    return {'points': points, 'depth': depth, 'mask': mask_binary, 'intrinsics': intrinsics}
