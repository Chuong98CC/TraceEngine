"""Shared helpers for image-folder-based inference pipelines."""

import glob
import os
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
from torchvision.transforms import v2 as _v2

from utils.depth_utils import load_depth_lz4
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


def load_stream_data(img_dir: str, result_dir: str, stem: str):
    """Load ``(depth, extrinsics, intrinsics)`` from one view's results.

    Depth is raw uint16 mm in ``<stem>.lz4`` (see load_depth_lz4) and is
    returned as float32 metres; pose lives in ``<stem>.npz``
    (``extrinsics``, ``intrinsics``, ``shape`` — the depth shape the lz4
    buffer must be reshaped to).
    """
    pose_path = Path(result_dir) / f"depth_{Path(img_dir).name}/{stem}.npz"
    if not pose_path.exists():
        raise FileNotFoundError(f"Result file not found: {pose_path}")
    with np.load(pose_path) as data:
        depth = load_depth_lz4(pose_path.with_suffix(".lz4"),
                               tuple(int(v) for v in data["shape"]))
        return (
            (depth / 1000.0).astype(np.float32),  # (H, W) uint16 mm -> metres
            data["extrinsics"].astype(np.float32),  # (3, 4)
            data["intrinsics"].astype(np.float32),  # (3, 3)
        )


def load_pair(
    stem: str,
    input_dirs: list[str],
    result_dir: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load depth, extrinsics, intrinsics, and images for one time step.

    Returns:
        depth:       (N, H, W) float32
        extrinsics:  (N, 3, 4) float32  world-to-camera
        intrinsics:  (N, 3, 3) float32
        images_u8:   (N, H, W, 3) uint8 RGB  (resized to match depth)
    """
    depths = []
    extrinsics_list = []
    intrinsics_list = []
    images_u8_list = []

    for img_dir in input_dirs:
        depth, ext, intr = load_stream_data(img_dir, result_dir, stem)
        depths.append(depth)
        extrinsics_list.append(ext)
        intrinsics_list.append(intr)

        # Load source image (any supported extension)
        img_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = Path(img_dir) / f"{stem}{ext}"
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


def load_npz_batch(npz_dir, file_list, start, end):
    """Load geometry for a slice of frames by original frame index.

    Depth is raw uint16 mm in ``frame_{idx:06d}.lz4`` (returned as float32
    metres); pose lives in ``frame_{idx:06d}.npz`` (``extrinsic``,
    ``intrinsic``, ``shape`` — the depth shape the lz4 buffer must be
    reshaped to).
    Returns dict with keys: depth (T,H,W), extrs (T,4,4), intrs (T,3,3).
    """
    depths, extrs, intrs = [], [], []
    for idx, _ in file_list[start:end]:
        lz4_path = os.path.join(npz_dir, f"frame_{idx:06d}.lz4")
        pose_path = os.path.join(npz_dir, f"frame_{idx:06d}.npz")
        if not os.path.exists(lz4_path) or not os.path.exists(pose_path):
            raise FileNotFoundError(f"Missing depth/pose: {lz4_path} / {pose_path}")
        with np.load(pose_path) as data:
            depth = load_depth_lz4(lz4_path, tuple(int(v) for v in data["shape"]))
            depths.append((depth / 1000.0).astype(np.float32))
            # repo pose-npz keys are plural (extrinsics 3x4 w2c / intrinsics);
            # accept the legacy singular 4x4 form too and pad 3x4 to 4x4
            extr = data["extrinsics"] if "extrinsics" in data else data["extrinsic"]
            if extr.shape == (3, 4):
                extr = np.vstack([extr, [0, 0, 0, 1]])
            extrs.append(extr)
            intrs.append(data["intrinsics"] if "intrinsics" in data else data["intrinsic"])

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


def load_resized_batch(file_list, npz_dir, start, end, inference_h, inference_w):
    """Load frames [start, end), resize to the inference resolution, scale intrinsics.

    Returns CPU tensors:
      video: (T, 3, H, W) float32 in [0, 1]
      depths: (T, H, W) float32
      intrs: (T, 3, 3) float32, fx/fy/cx/cy scaled to the inference resolution
      extrs: (T, 4, 4) float32
    """
    video = load_batch_frames(file_list, start, end)          # (T,3,H0,W0) uint8
    geo = load_npz_batch(npz_dir, file_list, start, end)
    orig_h, orig_w = video.shape[1:3]

    video_rs = torch.stack([
        F_resize(video[t], (inference_h, inference_w),
                 interpolation=_v2.InterpolationMode.BILINEAR, antialias=False)
        for t in range(video.shape[0])])
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


def compute_global_depth_roi(npz_dir, file_list, inference_h, inference_w):
    """Pre-scan all frames' resized depths; return the global IQR depth ROI.

    Depths are raw uint16 mm ``frame_{idx:06d}.lz4`` files with the depth
    shape recorded in the companion ``frame_{idx:06d}.npz`` (see
    load_npz_batch).
    Matches utils.inference_utils.inference(): roi = [1e-7, q75 + 1.5*iqr]
    computed over the resized depth maps of every frame.
    """
    all_d = []
    for idx, _ in file_list:
        lz4_path = os.path.join(npz_dir, f"frame_{idx:06d}.lz4")
        pose_path = os.path.join(npz_dir, f"frame_{idx:06d}.npz")
        if not os.path.exists(lz4_path):
            raise FileNotFoundError(f"Missing depth lz4: {lz4_path}")
        with np.load(pose_path) as data:
            depth = load_depth_lz4(lz4_path, tuple(int(v) for v in data["shape"]))
        d = resize_depth_bilinear((depth / 1000.0).astype(np.float32),
                                  (inference_w, inference_h))
        all_d.append(d[d > 0])
    d = torch.from_numpy(np.concatenate(all_d)).float()
    if len(d) < 4:
        return torch.tensor([1e-7, 1e7], dtype=torch.float32)
    q25 = torch.kthvalue(d, int(0.25 * len(d))).values
    q75 = torch.kthvalue(d, int(0.75 * len(d))).values
    iqr = q75 - q25
    return torch.tensor([1e-7, (q75 + 1.5 * iqr).item()], dtype=torch.float32)
