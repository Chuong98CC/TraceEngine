"""Depth + point-cloud visualization for the ONNX/TRT inference wrappers.

Adapted from the PyTorch model's exporters
(``depth_anything_3.utils.export.glb`` / ``depth_vis``) so it operates directly
on the wrappers' plain numpy outputs (``depth``/``conf``/``intrinsics``/
``extrinsics`` + processed images) instead of a ``Prediction`` object.

- ``save_depth_vis`` — colour-coded depth maps saved alongside the original
  images (side-by-side JPEGs).
- ``export_glb`` — back-projects the depth maps into a coloured world point
  cloud (+ optional camera frustums) and writes a ``.glb``.
"""

from __future__ import annotations

from pathlib import Path

import imageio
import numpy as np
import trimesh

import matplotlib
import numpy as np
import torch
from einops import rearrange



def visualize_depth(
    depth: np.ndarray,
    depth_min=None,
    depth_max=None,
    percentile=2,
    ret_minmax=False,
    ret_type=np.uint8,
    cmap="Spectral",
):
    """
    Visualize a depth map using a colormap.

    Args:
        depth: Input depth map array
        depth_min: Minimum depth value for normalization. If None, uses percentile
        depth_max: Maximum depth value for normalization. If None, uses percentile
        percentile: Percentile for min/max computation if not provided
        ret_minmax: Whether to return min/max depth values
        ret_type: Return array type (uint8 or float)
        cmap: Matplotlib colormap name to use

    Returns:
        Colored depth visualization as numpy array
        If ret_minmax=True, also returns depth_min and depth_max
    """
    depth = depth.copy()
    depth.copy()
    valid_mask = depth > 0
    depth[valid_mask] = 1 / depth[valid_mask]
    if depth_min is None:
        if valid_mask.sum() <= 10:
            depth_min = 0
        else:
            depth_min = np.percentile(depth[valid_mask], percentile)
    if depth_max is None:
        if valid_mask.sum() <= 10:
            depth_max = 0
        else:
            depth_max = np.percentile(depth[valid_mask], 100 - percentile)
    if depth_min == depth_max:
        depth_min = depth_min - 1e-6
        depth_max = depth_max + 1e-6
    cm = matplotlib.colormaps[cmap]
    depth = ((depth - depth_min) / (depth_max - depth_min)).clip(0, 1)
    depth = 1 - depth
    img_colored_np = cm(depth[None], bytes=False)[:, :, :, 0:3]  # value from 0 to 1
    if ret_type == np.uint8:
        img_colored_np = (img_colored_np[0] * 255.0).astype(np.uint8)
    elif ret_type == np.float32 or ret_type == np.float64:
        img_colored_np = img_colored_np[0]
    else:
        raise ValueError(f"Invalid return type: {ret_type}")
    if ret_minmax:
        return img_colored_np, depth_min, depth_max
    else:
        return img_colored_np


# GS video rendering visulization function, since it operates in Tensor space...


def vis_depth_map_tensor(
    result: torch.Tensor,  # "*batch height width"
    color_map: str = "Spectral",
) -> torch.Tensor:  # "*batch 3 height with"
    """
    Color-map the depth map.
    """
    far = result.reshape(-1)[:16_000_000].float().quantile(0.99).log().to(result)
    try:
        near = result[result > 0][:16_000_000].float().quantile(0.01).log().to(result)
    except (RuntimeError, ValueError) as e:
        print(f"No valid depth values found. Reason: {e}")
        near = torch.zeros_like(far)
    result = result.log()
    result = (result - near) / (far - near)
    return apply_color_map_to_image(result, color_map)


def apply_color_map(
    x: torch.Tensor,  # " *batch"
    color_map: str = "inferno",
) -> torch.Tensor:  # "*batch 3"
    cmap = matplotlib.cm.get_cmap(color_map)

    # Convert to NumPy so that Matplotlib color maps can be used.
    mapped = cmap(x.float().detach().clip(min=0, max=1).cpu().numpy())[..., :3]

    # Convert back to the original format.
    return torch.tensor(mapped, device=x.device, dtype=torch.float32)


def apply_color_map_to_image(
    image: torch.Tensor,  # "*batch height width"
    color_map: str = "inferno",
) -> torch.Tensor:  # "*batch 3 height with"
    image = apply_color_map(image, color_map)
    return rearrange(image, "... h w c -> ... c h w")

# ---------------------------------------------------------------------------
# Depth visualization (colour-coded depth alongside the original image)
# ---------------------------------------------------------------------------


def save_depth_vis(
    images_u8: np.ndarray,
    depth: np.ndarray,
    out_dir: str | Path,
    filenames: list[str] | None = None,
) -> Path:
    """Save ``[original | colour-coded depth]`` JPEGs, one per view.

    ``images_u8`` is ``(N, H, W, 3)`` uint8 RGB; ``depth`` is ``(N, H, W)``.
    Mirrors ``depth_anything_3.utils.export.depth_vis.export_to_depth_vis``.
    """
    vis_dir = Path(out_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(depth.shape[0]):
        depth_vis = visualize_depth(depth[idx]).astype(np.uint8)  # (H, W, 3) RGB
        image_vis = images_u8[idx].astype(np.uint8)
        vis_image = np.concatenate([image_vis, depth_vis], axis=1)
        filename = filenames[idx] if filenames is not None and idx < len(filenames) else f"{idx:04d}.jpg"
        imageio.imwrite((vis_dir / filename).as_posix(), vis_image, quality=95)
    return vis_dir


# ---------------------------------------------------------------------------
# GLB point-cloud export
# ---------------------------------------------------------------------------


def export_glb(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    images_u8: np.ndarray,
    conf: np.ndarray | None = None,
    out_path: str | Path = "scene.glb",
    num_max_points: int = 1_000_000,
    conf_thresh: float = 1.05,
    conf_thresh_percentile: float = 40.0,
    ensure_thresh_percentile: float = 90.0,
    show_cameras: bool = True,
    camera_size: float = 0.03,
) -> str:
    """Back-project ``depth`` into a coloured world point cloud and write a ``.glb``.

    ``depth`` ``(N, H, W)``; ``intrinsics`` ``(N, 3, 3)``; ``extrinsics`` ``(N, 3, 4)``
    or ``(N, 4, 4)`` world-to-camera; ``images_u8`` ``(N, H, W, 3)`` uint8 RGB;
    ``conf`` optional ``(N, H, W)`` used to filter low-confidence points.
    """
    depth = np.asarray(depth)
    intrinsics = np.asarray(intrinsics)
    extrinsics = np.asarray(extrinsics)
    images_u8 = np.asarray(images_u8)

    if depth.ndim == 2:
        depth = depth[None]
    if images_u8.ndim == 3:
        images_u8 = images_u8[None]

    depth_for_points = depth
    intrinsics_for_points = intrinsics
    extrinsics_for_points = extrinsics
    images_for_points = images_u8
    conf_for_points = conf
    if depth.shape[0] == 1 and intrinsics.shape[0] > 1:
        depth_for_points = depth[:1]
        intrinsics_for_points = intrinsics[:1]
        extrinsics_for_points = extrinsics[:1]
        images_for_points = images_u8[:1]
        if conf is not None:
            conf_for_points = conf[:1]

    if depth_for_points.shape[0] != intrinsics_for_points.shape[0] or depth_for_points.shape[0] != extrinsics_for_points.shape[0]:
        raise ValueError(
            f"Depth count {depth_for_points.shape[0]} must match intrinsics/extrinsics counts "
            f"{intrinsics_for_points.shape[0]}/{extrinsics_for_points.shape[0]}"
        )

    # Adaptive confidence threshold (clamped into the [p_lo, p_hi] band).
    if conf_for_points is not None:
        conf_for_points = np.asarray(conf_for_points)
        lower = np.percentile(conf_for_points, conf_thresh_percentile)
        upper = np.percentile(conf_for_points, ensure_thresh_percentile)
        conf_thr = float(min(max(conf_thresh, lower), upper))
    else:
        conf_thr = conf_thresh

    # Back-project to world coordinates and pull per-point colours.
    points, colors = _depths_to_world_points_with_colors(
        depth_for_points, intrinsics_for_points, extrinsics_for_points, images_for_points, conf_for_points, conf_thr
    )

    # Align to the first camera in glTF axes, centred on the point cloud.
    A = _compute_alignment_transform_first_cam_glTF_center_by_points(extrinsics[0], points)
    if points.shape[0] > 0:
        points = trimesh.transform_points(points, A)

    points, colors = _filter_and_downsample(points, colors, num_max_points)

    scene = trimesh.Scene()
    scene.metadata = {"hf_alignment": A}
    if points.shape[0] > 0:
        scene.add_geometry(trimesh.points.PointCloud(vertices=points, colors=colors))

    if show_cameras:
        scene_scale = _estimate_scene_scale(points, fallback=1.0)
        H, W = depth.shape[1:3]
        _add_cameras_to_scene(
            scene=scene,
            K=intrinsics,
            ext_w2c=extrinsics,
            image_sizes=[(H, W)] * intrinsics.shape[0],
            scale=scene_scale * camera_size,
        )

    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    scene.export(out_path)
    return out_path


# =========================
# utilities (copied from depth_anything_3.utils.export.glb)
# =========================


def _as_homogeneous44(ext: np.ndarray) -> np.ndarray:
    """Accept ``(4, 4)`` or ``(3, 4)`` extrinsics, return a ``(4, 4)`` matrix."""
    if ext.shape == (4, 4):
        return ext
    if ext.shape == (3, 4):
        h = np.eye(4, dtype=ext.dtype)
        h[:3, :4] = ext
        return h
    raise ValueError(f"extrinsic must be (4,4) or (3,4), got {ext.shape}")


def _depths_to_world_points_with_colors(
    depth: np.ndarray,
    K: np.ndarray,
    ext_w2c: np.ndarray,
    images_u8: np.ndarray,
    conf: np.ndarray | None,
    conf_thr: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Unproject each frame's pixels to world points, gathering their colours."""
    n, h, w = depth.shape
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    ones = np.ones_like(us)
    pix = np.stack([us, vs, ones], axis=-1).reshape(-1, 3)  # (H*W, 3)

    pts_all, col_all = [], []
    for i in range(n):
        d = depth[i]
        valid = np.isfinite(d) & (d > 0)
        if conf is not None:
            valid &= conf[i] >= conf_thr
        if not np.any(valid):
            continue

        d_flat = d.reshape(-1)
        vidx = np.flatnonzero(valid.reshape(-1))

        k_inv = np.linalg.inv(K[i])
        c2w = np.linalg.inv(_as_homogeneous44(ext_w2c[i]))

        rays = k_inv @ pix[vidx].T  # (3, M)
        xc = rays * d_flat[vidx][None, :]
        xc_h = np.vstack([xc, np.ones((1, xc.shape[1]))])
        xw = (c2w @ xc_h)[:3].T.astype(np.float32)  # (M, 3)

        cols = images_u8[i].reshape(-1, 3)[vidx].astype(np.uint8)
        pts_all.append(xw)
        col_all.append(cols)

    if len(pts_all) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)
    return np.concatenate(pts_all, 0), np.concatenate(col_all, 0)


def _filter_and_downsample(points: np.ndarray, colors: np.ndarray, num_max: int):
    if points.shape[0] == 0:
        return points, colors
    finite = np.isfinite(points).all(axis=1)
    points, colors = points[finite], colors[finite]
    if points.shape[0] > num_max:
        idx = np.random.choice(points.shape[0], num_max, replace=False)
        points, colors = points[idx], colors[idx]
    return points, colors


def _estimate_scene_scale(points: np.ndarray, fallback: float = 1.0) -> float:
    if points.shape[0] < 2:
        return fallback
    lo = np.percentile(points, 5, axis=0)
    hi = np.percentile(points, 95, axis=0)
    diag = np.linalg.norm(hi - lo)
    return float(diag if np.isfinite(diag) and diag > 0 else fallback)


def _compute_alignment_transform_first_cam_glTF_center_by_points(
    ext_w2c0: np.ndarray,
    points_world: np.ndarray,
) -> np.ndarray:
    """Align to the first camera view, convert CV→glTF axes, centre on the points."""
    w2c0 = _as_homogeneous44(ext_w2c0).astype(np.float64)

    m = np.eye(4, dtype=np.float64)
    m[1, 1] = -1.0  # flip Y
    m[2, 2] = -1.0  # flip Z

    a_no_center = m @ w2c0
    if points_world.shape[0] > 0:
        pts_tmp = trimesh.transform_points(points_world, a_no_center)
        center = np.median(pts_tmp, axis=0)
    else:
        center = np.zeros(3, dtype=np.float64)

    t_center = np.eye(4, dtype=np.float64)
    t_center[:3, 3] = -center
    return t_center @ a_no_center


def _add_cameras_to_scene(
    scene: trimesh.Scene,
    K: np.ndarray,
    ext_w2c: np.ndarray,
    image_sizes: list[tuple[int, int]],
    scale: float,
) -> None:
    """Draw each camera as a wireframe frustum, aligned via ``hf_alignment``."""
    n = K.shape[0]
    if n == 0:
        return

    a = None
    try:
        a = scene.metadata.get("hf_alignment", None) if scene.metadata else None
    except Exception:
        a = None
    if a is None:
        a = np.eye(4, dtype=np.float64)

    for i in range(n):
        h, w = image_sizes[i]
        segs = _camera_frustum_lines(K[i], ext_w2c[i], w, h, scale)  # (8, 2, 3)
        segs = trimesh.transform_points(segs.reshape(-1, 3), a).reshape(-1, 2, 3)
        path = trimesh.load_path(segs)
        color = _index_color_rgb(i, n)
        if hasattr(path, "colors"):
            path.colors = np.tile(color, (len(path.entities), 1))
        scene.add_geometry(path)


def _camera_frustum_lines(
    K: np.ndarray, ext_w2c: np.ndarray, w: int, h: int, scale: float
) -> np.ndarray:
    corners = np.array(
        [[0, 0, 1.0], [w - 1, 0, 1.0], [w - 1, h - 1, 1.0], [0, h - 1, 1.0]],
        dtype=float,
    )
    k_inv = np.linalg.inv(K)
    c2w = np.linalg.inv(_as_homogeneous44(ext_w2c))

    center_w = (c2w @ np.array([0, 0, 0, 1.0]))[:3]
    rays = (k_inv @ corners.T).T
    z = rays[:, 2:3]
    z[z == 0] = 1.0
    plane_cam = (rays / z) * scale

    plane_w = []
    for p in plane_cam:
        pw = (c2w @ np.array([p[0], p[1], p[2], 1.0]))[:3]
        plane_w.append(pw)
    plane_w = np.stack(plane_w, 0)

    segs = []
    for k in range(4):
        segs.append(np.stack([center_w, plane_w[k]], 0))
    order = [0, 1, 2, 3, 0]
    for a, b in zip(order[:-1], order[1:]):
        segs.append(np.stack([plane_w[a], plane_w[b]], 0))
    return np.stack(segs, 0)  # (8, 2, 3)


def _index_color_rgb(i: int, n: int) -> np.ndarray:
    h = (i + 0.5) / max(n, 1)
    r, g, b = _hsv_to_rgb(h, 0.85, 0.95)
    return (np.array([r, g, b]) * 255).astype(np.uint8)


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[float, float, float]:
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6
    if i == 0:
        return v, t, p
    if i == 1:
        return q, v, p
    if i == 2:
        return p, v, t
    if i == 3:
        return p, q, v
    if i == 4:
        return t, p, v
    return v, p, q
