"""Shared helpers for image-folder-based inference pipelines."""

import glob
import os
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
from torchvision.transforms import v2 as _v2

from utils.depth_pose_io import DepthPoseReader
from utils.file_io.image_io import to_image_tensor

F_resize = _v2.functional.resize

def resize_depth_bilinear(depth: np.ndarray, new_shape: Tuple[int, int]) -> np.ndarray:
    is_valid = (depth > 0).astype(np.float32)
    depth_resized = cv2.resize(depth, new_shape, interpolation=cv2.INTER_LINEAR)
    is_valid_resized = cv2.resize(is_valid, new_shape, interpolation=cv2.INTER_LINEAR)
    depth_resized = depth_resized / (is_valid_resized + 1e-6)
    depth_resized[is_valid_resized <= 1e-6] = 0.0
    return depth_resized


def scan_image_folder(image_dir, start_frame=0, fps=1, max_frames=None):
    """Scan frame_{idx}.jpg / .png files, return metadata without loading images.

    Returns:
        file_list: list of (frame_index, file_path) sorted by index
        frame_H, frame_W: dimensions from first frame
    """
    files = []
    for ext in ("jpg", "png"):
        files.extend(glob.glob(os.path.join(image_dir, f"frame_*.{ext}")))
    if not files:
        raise FileNotFoundError(f"No frame_*.jpg / .png files in {image_dir}")

    all_indices = []
    for f in files:
        basename = os.path.splitext(os.path.basename(f))[0]
        idx = int(basename.split("_")[-1])
        all_indices.append((idx, f))
    all_indices.sort(key=lambda x: x[0])

    # Filter by frame index value (not list position)
    sampled = [(idx, f) for idx, f in all_indices if idx >= start_frame][::fps]
    if max_frames is not None:
        sampled = sampled[:max_frames]
    if not sampled:
        raise RuntimeError(
            f"No frame images with index >= {start_frame} in {image_dir}"
        )

    # Probe first frame for dimensions
    first = cv2.imread(sampled[0][1])
    if first is None:
        raise RuntimeError(f"Cannot read first frame: {sampled[0][1]}")
    frame_H, frame_W = first.shape[:2]

    return sampled, frame_H, frame_W


def load_stream_data(depth_dir, frame_index: int):
    """Load ``(depth, extrinsics, intrinsics)`` of one camera at one frame.

    ``depth_dir`` is a ``depth_pose/<camera>`` folder (``depth.lz4`` +
    ``poses.npz``); ``frame_index`` is the absolute dataset frame index.
    Depth is (H, W) float32 metres.
    """
    with DepthPoseReader(depth_dir) as reader:
        i = reader.index_of(frame_index)
        return reader.depth(i), reader.extrinsics[i], reader.intrinsics[i]


def load_pair(
    frame_index: int,
    image_dirs: list[str],
    depth_dirs: list[str],
):
    """Load depth, extrinsics, intrinsics and images for one time step.

    ``image_dirs`` / ``depth_dirs`` are parallel per-camera lists (the
    ``frames/<camera>`` and ``depth_pose/<camera>`` folders).

    Returns:
        depth:       (N, H, W) float32
        extrinsics:  (N, 3, 4) float32  world-to-camera
        intrinsics:  (N, 3, 3) float32
        images_u8:   (N, H, W, 3) uint8 RGB  (resized to match depth)
    """
    if len(image_dirs) != len(depth_dirs):
        raise ValueError(
            f"image_dirs ({len(image_dirs)} folders) and depth_dirs "
            f"({len(depth_dirs)} folders) must be parallel per-camera lists"
        )
    depths, extrinsics_list, intrinsics_list, images_u8_list = [], [], [], []

    for img_dir, depth_dir in zip(image_dirs, depth_dirs, strict=True):
        depth, ext, intr = load_stream_data(depth_dir, frame_index)
        depths.append(depth)
        extrinsics_list.append(ext)
        intrinsics_list.append(intr)

        stem = f"frame_{int(frame_index):06d}"
        img_path = None
        for suffix in (".jpg", ".jpeg", ".png"):
            candidate = Path(img_dir) / f"{stem}{suffix}"
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            raise FileNotFoundError(f"Image not found: {Path(img_dir) / stem}.*")
        H, W = depth.shape
        rgb = (
            _v2.functional.resize(
                to_image_tensor(img_path), (H, W),
                interpolation=_v2.InterpolationMode.BILINEAR, antialias=False,
            )
            .permute(1, 2, 0)
            .numpy()
        )
        images_u8_list.append(rgb)

    return (
        np.stack(depths, axis=0),
        np.stack(extrinsics_list, axis=0),
        np.stack(intrinsics_list, axis=0),
        np.stack(images_u8_list, axis=0),
    )


def load_batch_frames(file_list, start, end):
    """Load a slice of frames into a (T, 3, H, W) uint8 CHW CPU tensor."""
    return torch.stack(
        [to_image_tensor(fpath) for _, fpath in file_list[start:end]]
    )


def load_npz_batch(depth_dir, frame_indexes: list[int], start: int, end: int):
    """Load geometry for a slice of a depth_pose camera folder.

    ``frame_indexes`` are absolute dataset frame indices (the order is the
    caller's, normally ascending); frames [start, end) are loaded.
    Returns dict with keys: depth (T,H,W), extrs (T,4,4), intrs (T,3,3).
    """
    depths, extrs, intrs = [], [], []
    reader = DepthPoseReader(depth_dir)
    try:
        for frame_index in frame_indexes[start:end]:
            i = reader.index_of(frame_index)
            depths.append(reader.depth(i))
            extr = reader.extrinsics[i]
            if extr.shape == (3, 4):
                extr = np.vstack([extr, [0, 0, 0, 1]])
            extrs.append(extr)
            intrs.append(reader.intrinsics[i])
    finally:
        reader.close()
    return {
        "depth": np.stack(depths, axis=0).astype(np.float32),
        "extrs": np.stack(extrs, axis=0).astype(np.float32),
        "intrs": np.stack(intrs, axis=0).astype(np.float32),
    }


def sample_grid_in_bbox(x0, y0, x1, y1, grid_x, grid_y, frame_H, frame_W, device="cpu"):
    """Return query points (1, N, 2) on a regular grid inside a bbox."""
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(frame_W - 1, int(x1)), min(frame_H - 1, int(y1))
    gy = torch.linspace(y0, y1, grid_y, device=device)
    gx = torch.linspace(x0, x1, grid_x, device=device)
    gy_m, gx_m = torch.meshgrid(gy, gx, indexing="ij")
    return torch.stack([gx_m.flatten(), gy_m.flatten()], dim=-1).unsqueeze(0)


def mask_points_by_frame(points_2d, frame_H, frame_W, margin=10):
    """Boolean mask: True for points inside the frame (with margin)."""
    x, y = points_2d[..., 0], points_2d[..., 1]
    return (x >= -margin) & (x < frame_W + margin) & (y >= -margin) & (y < frame_H + margin)


def unproject_xy_queries(xy, depth0, intr0, extr0, device="cpu"):
    """Unproject a set of 2D pixel coordinates to 3D world-coordinate queries.

    Args:
        xy: (N, 2) pixel coordinates (x, y) at the query frame (numpy or
            torch; int or float — rounded to pixels for the depth lookup)
        depth0: (H, W) depth map at the query frame (numpy or torch)
        intr0: (3, 3) intrinsics at the query frame
        extr0: (4, 4) extrinsics (world→camera) at the query frame

    Returns:
        queries: (M, 4) tensor with (frame_idx=0, x, y, z) in world coords,
                 or None if no valid points (points with depth == 0 are dropped)
    """
    if isinstance(xy, np.ndarray):
        xy = torch.from_numpy(xy).float()
    else:
        xy = xy.float()
    if isinstance(depth0, np.ndarray):
        depth0 = torch.from_numpy(depth0).float().to(device)
    if isinstance(intr0, np.ndarray):
        intr0 = torch.from_numpy(intr0).float().to(device)
    if isinstance(extr0, np.ndarray):
        extr0 = torch.from_numpy(extr0).float().to(device)

    # Look up depth at each point (float coords rounded to pixels)
    ji = torch.round(xy).to(torch.int32)
    ji[:, 0] = ji[:, 0].clamp(0, depth0.shape[1] - 1)
    ji[:, 1] = ji[:, 1].clamp(0, depth0.shape[0] - 1)
    d = depth0[ji[:, 1], ji[:, 0]]  # (N,)

    # Filter out points with invalid depth
    mask = d > 0
    if not mask.any():
        return None
    xy = xy[mask]
    d = d[mask]

    # Unproject to world coordinates
    inv_intr0 = torch.linalg.inv(intr0)
    inv_extr0 = torch.linalg.inv(extr0)

    xy_homo = torch.cat([xy, torch.ones_like(xy[..., :1])], dim=-1)  # (N, 3)
    # pixel coords -> camera-frame rays (K^-1), matching get_grid_queries
    xy_homo = torch.einsum('ij,nj->ni', inv_intr0, xy_homo)
    local_coords = xy_homo * d.unsqueeze(-1)  # (N, 3)
    local_coords_homo = torch.cat(
        [local_coords, torch.ones_like(local_coords[..., :1])], dim=-1)  # (N, 4)
    world_coords = torch.einsum('ij,nj->ni', inv_extr0, local_coords_homo)
    world_coords = world_coords[..., :3]  # (N, 3)

    queries = torch.cat(
        [torch.zeros_like(xy[:, :1]), world_coords], dim=-1)  # (N, 4)
    return queries


def unproject_bbox_queries(x0, y0, x1, y1, grid_x, grid_y, depth0, intr0, extr0,
                           frame_H, frame_W, device="cpu"):
    """Sample a grid inside a bbox and unproject to 3D world-coordinate queries.

    Args:
        x0, y0, x1, y1: bbox in pixel coordinates
        grid_x, grid_y: number of grid points in each dimension
        depth0: (H, W) depth map at the query frame (numpy or torch)
        intr0: (3, 3) intrinsics at the query frame
        extr0: (4, 4) extrinsics (world→camera) at the query frame
        frame_H, frame_W: image dimensions

    Returns:
        queries: (N, 4) tensor with (frame_idx=0, x, y, z) in world coords,
                 or None if no valid points
    """
    # Sample 2D grid in bbox
    xy = sample_grid_in_bbox(x0, y0, x1, y1, grid_x, grid_y,
                              frame_H, frame_W, device=device)  # (1, N, 2)
    xy = xy.squeeze(0)  # (N, 2)
    return unproject_xy_queries(xy, depth0, intr0, extr0, device=device)


def resize_batch_to_inference(video_u8: torch.Tensor, geo: dict,
                              inference_h: int, inference_w: int):
    """Resize an already-loaded batch to the inference resolution.

    The resize/batch math of ``load_resized_batch``, split out so callers
    that decode their frames online (no image files on disk, e.g. the
    Astribot step-4 tracker) can reuse it: same bilinear frame resize to
    (inference_h, inference_w), same valid-depth-preserving depth resize,
    same (inf - 1)/(orig - 1) intrinsics scaling.

    Args:
        video_u8: (T, 3, H0, W0) uint8 CPU frames (torchvision tensor-space).
        geo: load_npz_batch-style dict with ``depth`` (T, H0, W0),
            ``intrs`` (T, 3, 3) and ``extrs`` (T, 4, 4) at the frame scale.
        inference_h, inference_w: target resolution.

    Returns CPU tensors:
      video: (T, 3, H, W) float32 in [0, 1]
      depths: (T, H, W) float32
      intrs: (T, 3, 3) float32, fx/fy/cx/cy scaled to the inference resolution
      extrs: (T, 4, 4) float32
    """
    orig_h, orig_w = video_u8.shape[2:4]  # video_u8 is (T, 3, H0, W0) CHW
    video_rs = torch.stack([
        F_resize(video_u8[t], (inference_h, inference_w),
                 interpolation=_v2.InterpolationMode.BILINEAR, antialias=False)
        for t in range(video_u8.shape[0])])
    depths_rs = np.stack([
        resize_depth_bilinear(geo["depth"][t], (inference_w, inference_h))
        for t in range(geo["depth"].shape[0])])

    scale_y = (inference_h - 1) / (orig_h - 1)
    scale_x = (inference_w - 1) / (orig_w - 1)
    intrs = geo["intrs"].copy()
    intrs[:, 0, :] *= scale_x
    intrs[:, 1, :] *= scale_y

    video_t = video_rs.float() / 255.0
    depths_t = torch.from_numpy(depths_rs).float()
    intrs_t = torch.from_numpy(intrs).float()
    extrs_t = torch.from_numpy(geo["extrs"]).float()
    return video_t, depths_t, intrs_t, extrs_t


def load_resized_batch(file_list, depth_dir, start, end, inference_h, inference_w):
    """Load frames [start, end), resize to the inference resolution, scale intrinsics.

    ``file_list`` is the ``scan_image_folder`` output (``[(idx, path), ...]``,
    the RGB images); geometry comes from the ``depth_pose`` camera folder.

    Returns CPU tensors:
      video: (T, 3, H, W) float32 in [0, 1]
      depths: (T, H, W) float32
      intrs: (T, 3, 3) float32, fx/fy/cx/cy scaled to the inference resolution
      extrs: (T, 4, 4) float32
    """
    frame_indexes = [int(idx) for idx, _ in file_list]
    video = load_batch_frames(file_list, start, end)          # (T,3,H0,W0) uint8
    geo = load_npz_batch(depth_dir, frame_indexes, start, end)
    return resize_batch_to_inference(video, geo, inference_h, inference_w)


def compute_global_depth_roi(depth_dir, frame_indexes, inference_h, inference_w):
    """Pre-scan all frames' resized depths; return the global IQR depth ROI.

    ``frame_indexes`` are absolute dataset frame indices of the ``depth_pose``
    camera folder (see load_npz_batch).
    Matches utils.inference_utils.inference(): roi = [1e-7, q75 + 1.5*iqr]
    computed over the resized depth maps of every frame.
    """
    all_d = []
    reader = DepthPoseReader(depth_dir)
    try:
        for frame_index in frame_indexes:
            depth = reader.depth_at(frame_index)
            d = resize_depth_bilinear(depth, (inference_w, inference_h))
            all_d.append(d[d > 0])
    finally:
        reader.close()
    d = torch.from_numpy(np.concatenate(all_d)).float()
    if len(d) < 4:
        return torch.tensor([1e-7, 1e7], dtype=torch.float32)
    q25 = torch.kthvalue(d, int(0.25 * len(d))).values
    q75 = torch.kthvalue(d, int(0.75 * len(d))).values
    iqr = q75 - q25
    return torch.tensor([1e-7, (q75 + 1.5 * iqr).item()], dtype=torch.float32)
