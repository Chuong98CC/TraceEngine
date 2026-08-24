#!/usr/bin/env python3
"""
Trajectory mp4 video for the streaming output.

Each video frame shows one time step's coloured point cloud with its
camera frustums, plus the growing camera path from the first frame to the
current one.  The view is fixed for the whole video (aligned to the first
camera, fitted to the union of all frame clouds).

CLI (no ``run_stream.py`` needed — uses the already-saved NPZ outputs):
    python da3_streaming/visualize_stream.py \
        --input-dirs data/astribot_stereo_lrb/extract_frames/stereo_left \
                      data/astribot_stereo_lrb/extract_frames/stereo_right \
        --result-dir output/stream_stereo_pytorch \
        --output output/stream_stereo_pytorch/trajectory.mp4

Also exposed as the ``--video`` flag in ``run_stream.py`` (rendered after
the run) via :func:`render_stream_video`.  Single-step GLB export lives in
``visualize_glb.py``; the shared NPZ/image loaders live in
``stream_utils.py``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import imageio
import numpy as np
import open3d as o3d
import trimesh

from utils.visualize_depth import (  # noqa: E402
    _as_homogeneous44,
    _camera_frustum_lines,
    _compute_alignment_transform_first_cam_glTF_center_by_points,
    _depths_to_world_points_with_colors,
    _index_color_rgb,
)

from utils.stream_utils import load_npz_data, load_pair  # noqa: E402


# ===========================================================================
# Trajectory video rendering
# ===========================================================================
def load_stems(result_dir: str, input_dirs: list[str]) -> list[str]:
    """Sorted NPZ stems of the first camera's output folder (== time order)."""
    npz_dir = Path(result_dir) / f"depth_{Path(input_dirs[0]).name}"
    return sorted(p.stem for p in npz_dir.glob("*.npz"))


def load_trajectory(
    stems: list[str],
    input_dirs: list[str],
    result_dir: str,
) -> np.ndarray:
    """World-space camera centers per time step: ``(T, N, 3)``, in stem order."""
    centers: list[np.ndarray] = []
    for stem in stems:
        cams = []
        for img_dir in input_dirs:
            _, ext, _ = load_npz_data(img_dir, result_dir, stem)
            c2w = np.linalg.inv(_as_homogeneous44(ext))
            cams.append(c2w[:3, 3])
        centers.append(np.stack(cams, axis=0))
    return np.stack(centers, axis=0)


def compute_view_transform(extrinsics0: np.ndarray, center_points: np.ndarray) -> np.ndarray:
    """Fixed ``(4, 4)`` alignment to the first camera in glTF axes.

    Centered on ``center_points`` (typically the union of all frame clouds
    and the trajectory) so the video frame stays stable while per-frame
    content moves inside it.
    """
    return _compute_alignment_transform_first_cam_glTF_center_by_points(
        extrinsics0[0], center_points.reshape(-1, 3)
    )


def build_frame_geometries(
    depth: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    images_u8: np.ndarray,
    traj_upto: np.ndarray,
    alignment: np.ndarray,
    cam_scale: float,
    max_points: int = 1_000_000,
    conf: np.ndarray | None = None,
) -> dict:
    """Build the open3d geometries for one video frame.

    The frame shows the current time step's coloured point cloud (capped at
    ``max_points``), that step's camera frustums, and the trajectory lines
    of all steps up to and including this one (``None`` on the first step).
    All coordinates are in the aligned glTF frame given by ``alignment``.
    """
    # Coloured cloud of the current step, subsampled to the cap.
    pts, cols = _depths_to_world_points_with_colors(
        depth, intrinsics, extrinsics, images_u8, conf, conf_thr=0.0
    )
    finite = np.isfinite(pts).all(axis=1)
    pts, cols = pts[finite], cols[finite]
    if len(pts) > max_points:
        idx = np.random.choice(len(pts), max_points, replace=False)
        pts, cols = pts[idx], cols[idx]
    if len(pts) > 0:
        pts = trimesh.transform_points(pts, alignment)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64) / 255.0)

    N = depth.shape[0]

    # Growing trajectory: one polyline per camera through its centers.
    T = traj_upto.shape[0]
    traj_lines = None
    if T >= 2:
        traj_pts = trimesh.transform_points(traj_upto.reshape(-1, 3), alignment)
        lines, line_colors = [], []
        for v in range(N):
            base = v * T
            for i in range(T - 1):
                lines.append([base + i, base + i + 1])
                line_colors.append(_index_color_rgb(v, N) / 255.0)
        traj_lines = o3d.geometry.LineSet()
        traj_lines.points = o3d.utility.Vector3dVector(traj_pts)
        traj_lines.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
        traj_lines.colors = o3d.utility.Vector3dVector(np.asarray(line_colors))

    # Current-step camera frustums, coloured per camera.
    H, W = depth.shape[1:]
    segs_all, seg_colors = [], []
    for v in range(N):
        segs = _camera_frustum_lines(intrinsics[v], extrinsics[v], W, H, cam_scale)  # (8,2,3)
        segs = trimesh.transform_points(segs.reshape(-1, 3), alignment).reshape(-1, 2, 3)
        segs_all.append(segs)
        seg_colors.append(np.tile(_index_color_rgb(v, N) / 255.0, (len(segs), 1)))
    segs_cat = np.concatenate(segs_all, axis=0)
    frustums = o3d.geometry.LineSet()
    frustums.points = o3d.utility.Vector3dVector(segs_cat.reshape(-1, 3))
    frustums.lines = o3d.utility.Vector2iVector(
        np.arange(len(segs_cat) * 2, dtype=np.int32).reshape(-1, 2)
    )
    frustums.colors = o3d.utility.Vector3dVector(np.concatenate(seg_colors, axis=0))

    return {"pcd": pcd, "trajectory": traj_lines, "frustums": frustums}


def _union_scene_points(
    stems: list[str],
    input_dirs: list[str],
    result_dir: str,
    stride: int,
) -> np.ndarray:
    """Union of all frames' clouds (strided) + the trajectory, for view fitting."""
    all_pts = []
    for stem in stems:
        depth, extrs, intrs, _ = load_pair(stem, input_dirs, result_dir)
        d = depth[:, ::stride, ::stride]
        K = intrs.copy()
        K[:, :2, :] /= stride
        n, h_, w_ = d.shape
        dummy = np.zeros((n, h_, w_, 3), dtype=np.uint8)
        pts, _ = _depths_to_world_points_with_colors(d, K, extrs, dummy, None, 0.0)
        all_pts.append(pts)
    all_pts.append(load_trajectory(stems, input_dirs, result_dir).reshape(-1, 3))
    return np.concatenate(all_pts, axis=0)


def _scene_max_extent(points: np.ndarray) -> float:
    """Longest 5th/95th-percentile side of the point set's bounding box."""
    lo = np.percentile(points, 5, axis=0)
    hi = np.percentile(points, 95, axis=0)
    return float(np.max(hi - lo))


_TONE_LUT_CACHE: dict[str, np.ndarray] = {}


def _build_inverse_tone_curve(renderer, w: int, h: int) -> np.ndarray:
    """LUT mapping rendered pixel values back to source colour levels.

    open3d's renderer applies a filmic tone curve that cannot be disabled
    in this build, so invert it empirically: render a dense point grid
    (full pixel coverage, so no antialiasing mixes the background in) at
    each gray level and read the center pixel back.  The grid matches the
    point-cloud rendering path used by the video frames.
    ``lut[rendered] -> source`` (both 0..255, uint8).

    The curve is a fixed Filament default per open3d build, so the LUT is
    cached per process.
    """
    key = o3d.__version__
    if key in _TONE_LUT_CACHE:
        return _TONE_LUT_CACHE[key]

    # Grid in front of the camera whose spacing is smaller than the probe
    # point size, so the center pixel is always covered by a point —
    # independent of the renderer resolution.  The grid spans beyond the
    # frustum; the camera looks straight at it from the origin.
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = 5.0
    half_h = np.tan(np.deg2rad(30.0))  # vertical half-fov at z=1
    half_w = half_h * (w / h)
    world_per_px = 2.0 * half_h / h
    spacing = mat.point_size * world_per_px * 0.7
    xs = np.arange(-1.5 * half_w, 1.5 * half_w, spacing, dtype=np.float64)
    ys = np.arange(-1.5 * half_h, 1.5 * half_h, spacing, dtype=np.float64)
    grid = np.stack(np.meshgrid(xs, ys, indexing="xy"), axis=-1).reshape(-1, 2)
    pts = np.hstack([grid, np.full((len(grid), 1), 1.0)])
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    renderer.scene.camera.look_at([0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    renderer.scene.set_background([0.0, 0.0, 0.0, 1.0])

    outs = np.empty(256, dtype=np.float64)
    for v in range(256):
        g = v / 255.0
        pcd.colors = o3d.utility.Vector3dVector(np.tile([g, g, g], (len(pts), 1)))
        renderer.scene.clear_geometry()
        renderer.scene.add_geometry("probe", pcd, mat)
        arr = np.asarray(renderer.render_to_image())
        outs[v] = arr[arr.shape[0] // 2, arr.shape[1] // 2, 0]

    outs = np.maximum.accumulate(outs)  # enforce monotonicity
    lut = np.clip(np.searchsorted(outs, np.arange(256)), 0, 255).astype(np.uint8)
    _TONE_LUT_CACHE[key] = lut
    return lut


def render_stream_video(
    stems: list[str],
    input_dirs: list[str],
    result_dir: str,
    out_path: str,
    fps: int = 10,
    size: tuple[int, int] = (960, 540),
    max_points_per_frame: int = 100_000,
    stride: int = 4,
) -> str:
    """Render the streaming trajectory video (current-step cloud, growing path).

    The view is fixed for the whole video: aligned to the first frame's
    first camera (glTF axes) and centered on the full trajectory, with the
    eye distance fitted to the union of all frame clouds.  Frames are
    encoded directly into ``out_path`` (H.264 mp4 via imageio-ffmpeg).
    """
    if not stems:
        raise ValueError("No stems to render.")

    w, h = size
    w -= w % 2
    h -= h % 2  # x264 wants even dimensions

    traj = load_trajectory(stems, input_dirs, result_dir)
    _, extr0, _, _ = load_pair(stems[0], input_dirs, result_dir)
    # The cloud wraps around the camera path (not around the trajectory
    # median), so the view is fit to the union of clouds + trajectory.
    scene_pts = _union_scene_points(stems, input_dirs, result_dir, stride)
    alignment = compute_view_transform(extr0, scene_pts)

    extent = _scene_max_extent(scene_pts)
    if not np.isfinite(extent) or extent <= 0:
        extent = 1.0
    cam_scale = extent * 0.03

    az, el = np.deg2rad(40.0), np.deg2rad(30.0)
    eye = (
        np.array(
            [np.sin(az) * np.cos(el), np.sin(el), np.cos(az) * np.cos(el)],
            dtype=np.float64,
        )
        * extent
    )

    renderer = o3d.visualization.rendering.OffscreenRenderer(w, h)

    # Invert the renderer's filmic tone curve so video colours match the
    # source images.  Built first: the probe sets its own camera, and the
    # fitted scene camera below must be the one used for rendering.
    tone_lut = _build_inverse_tone_curve(renderer, w, h)

    renderer.scene.set_background([0.05, 0.05, 0.05, 1.0])
    renderer.scene.camera.look_at(np.zeros(3), eye, [0.0, 1.0, 0.0])

    pcd_mat = o3d.visualization.rendering.MaterialRecord()
    pcd_mat.shader = "defaultUnlit"
    line_mat = o3d.visualization.rendering.MaterialRecord()
    line_mat.shader = "unlitLine"
    line_mat.line_width = 4.0

    # macro_block_size=1 keeps the exact size (imageio would otherwise pad
    # e.g. 960x540 up to a multiple of 16); even dimensions are enough for
    # libx264/yuv420p, and we enforce them above.
    writer = imageio.get_writer(
        out_path, fps=fps, codec="libx264", quality=8, macro_block_size=1
    )
    try:
        for t, stem in enumerate(stems):
            depth, extrs, intrs, images = load_pair(stem, input_dirs, result_dir)
            geoms = build_frame_geometries(
                depth,
                extrs,
                intrs,
                images,
                traj_upto=traj[: t + 1],
                alignment=alignment,
                cam_scale=cam_scale,
                max_points=max_points_per_frame,
            )
            renderer.scene.clear_geometry()
            renderer.scene.add_geometry("pcd", geoms["pcd"], pcd_mat)
            if geoms["trajectory"] is not None:
                renderer.scene.add_geometry("trajectory", geoms["trajectory"], line_mat)
            renderer.scene.add_geometry("frustums", geoms["frustums"], line_mat)

            arr = np.asarray(renderer.render_to_image())
            if arr.shape[-1] == 4:
                arr = arr[..., :3]
            arr = tone_lut[arr]
            writer.append_data(arr)
            print(f"  rendered {t + 1}/{len(stems)}: {stem}")
    finally:
        writer.close()
    return out_path


# ===========================================================================
# CLI
# ===========================================================================
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render the streaming trajectory video from saved NPZ outputs"
    )
    parser.add_argument(
        "--input-dirs", nargs="+",
        default=[
            "data/astribot_stereo_lrb/extract_frames/stereo_left",
            "data/astribot_stereo_lrb/extract_frames/stereo_right",
        ],
        help="Source image folders (one per camera), same order as inference",
    )
    parser.add_argument("--result-dir", default="output/stream_stereo",
                        help="Streaming output directory")
    parser.add_argument("--output", default=None,
                        help="Video output path (default: <result-dir>/trajectory.mp4)")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--size", default="960x540",
                        help="Video size WxH, e.g. 960x540 (default: 960x540)")
    parser.add_argument(
        "--max-points", type=int, default=100_000,
        help="Max point-cloud points rendered per video frame (default: 100000)",
    )
    args = parser.parse_args(argv)

    stems = load_stems(args.result_dir, args.input_dirs)
    w, h = (int(x) for x in args.size.lower().split("x"))
    out_path = args.output or os.path.join(args.result_dir, "trajectory.mp4")
    print(f"Rendering trajectory video ({len(stems)} frames) -> {out_path}")
    render_stream_video(
        stems,
        args.input_dirs,
        args.result_dir,
        out_path,
        fps=args.fps,
        size=(w, h),
        max_points_per_frame=args.max_points,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
