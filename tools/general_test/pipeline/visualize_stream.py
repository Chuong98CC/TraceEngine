#!/usr/bin/env python3
"""
Trajectory mp4 video for the streaming output.

Each video frame shows one time step's coloured point cloud with its
camera frustums, plus the growing camera path from the first frame to the
current one.  The view is fixed for the whole video (aligned to the first
camera, eye behind it along its optical axis, fitted to the union of all
frame clouds); with ``--views 4`` each frame is a 2x2 grid of viewpoints
(center / down / left / right), all looking at the scene centre.

CLI (no ``run_depth_stream.py`` needed — uses the already-saved NPZ outputs):
    python da3_streaming/visualize_stream.py \
        --input-dirs data/astribot_stereo_lrb/extract_frames/stereo_left \
                      data/astribot_stereo_lrb/extract_frames/stereo_right \
        --result-dir output/stream_stereo_pytorch \
        --output output/stream_stereo_pytorch/trajectory.mp4

Also exposed as the ``--video`` flag in ``run_depth_stream.py`` (rendered after
the run) via :func:`render_stream_video`.  Single-step GLB export lives in
``visualize_glb.py``; the shared NPZ/image loaders live in
``stream_utils.py``.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable
from pathlib import Path

import cv2
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

from utils.streaming_utils import load_stream_data, load_pair  # noqa: E402


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
            _, ext, _ = load_stream_data(img_dir, result_dir, stem)
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
    """Union of all frames' clouds (strided) + the trajectory, for view fitting.

    Geometry only — the per-frame images are discarded anyway, so this works
    without frame folders (e.g. the online visualizer)."""
    all_pts = []
    for stem in stems:
        depths, exts, ints = [], [], []
        for img_dir in input_dirs:
            d, e, i = load_stream_data(img_dir, result_dir, stem)
            depths.append(d)
            exts.append(e)
            ints.append(i)
        depth = np.stack(depths, axis=0)
        extrs = np.stack(exts, axis=0)
        intrs = np.stack(ints, axis=0)
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


def _render_view(
    renderer,
    tone_lut: np.ndarray,
    look_at: np.ndarray,
    eye: np.ndarray,
    fov: float,
    aspect: float,
) -> np.ndarray:
    """Render the current scene from ``eye`` looking at ``look_at`` (up +y).

    The field of view is re-applied after ``look_at`` (which may reset it to
    the renderer default).
    """
    renderer.scene.camera.look_at(look_at, eye, [0.0, 1.0, 0.0])
    renderer.scene.camera.set_projection(
        fov, aspect, 0.01, 1000.0,
        o3d.visualization.rendering.Camera.FovType.Vertical,
    )
    arr = np.asarray(renderer.render_to_image())
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return tone_lut[arr]


def _label_viewport(canvas: np.ndarray, name: str, x: int, y: int) -> None:
    """Draw ``name`` at a viewport corner (black outline + white fill)."""
    cv2.putText(canvas, name, (x, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, name, (x, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)


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
    view_distance: float = 0.3,
    views: int = 4,
    view_angle: float = 45.0,
    view_lower: float = 0.1,
    view_raise: float = 0.1,
    view_back: float = 0.3,
    view_fov: float | None = None,
    frame_loader: Callable[[str], np.ndarray] | None = None,
) -> str:
    """Render the streaming trajectory video (current-step cloud, growing path).

    The view is fixed for the whole video: aligned to the first frame's
    first camera (glTF axes), with the eyes ``view_distance`` scene-extents
    behind it and all looking at the scene centre (the aligned origin —
    the alignment is centred on the union of all frame clouds).  With
    ``views=4`` each frame is a 2x2 grid: center; the down view translated
    ``view_raise`` extents upward and ``view_back`` extents backward (the
    camera pose falls below the fitted viewport when the eye is directly
    above it, so the eye is pulled back to bring the frustums/path into
    view); the left and right views swing ``view_angle`` degrees around the
    scene centre's vertical axis (45 deg by default) and sit ``view_lower``
    extents below the center eye.  ``views=1`` renders only the center view.

    Each viewport's field of view is auto-fitted so the scene fills the
    frame: the union points' p90 angular extent maps to ~85% of the
    viewport (clamped to 20-90 deg); ``view_fov`` overrides the fit in
    vertical degrees.  Frames are encoded directly into ``out_path``
    (H.264 mp4 via imageio-ffmpeg).

    ``frame_loader`` overrides the per-step images (``(N, H, W, 3)`` uint8
    RGB, resized to the depth resolution) when the source frames do not
    live on disk — e.g. ``visualize_subtask_stream.py`` decodes them online
    from a LeRobotDataset.  Geometry (depth/extrinsics/intrinsics) always
    comes from the saved NPZs in ``result_dir``.
    """
    if not stems:
        raise ValueError("No stems to render.")

    w, h = size
    w -= w % 2
    h -= h % 2  # x264 wants even dimensions

    traj = load_trajectory(stems, input_dirs, result_dir)
    exts0 = []
    for img_dir in input_dirs:
        _, e, _ = load_stream_data(img_dir, result_dir, stems[0])
        exts0.append(e)
    extr0 = np.stack(exts0, axis=0)
    # The cloud wraps around the camera path (not around the trajectory
    # median), so the view is fit to the union of clouds + trajectory.
    scene_pts = _union_scene_points(stems, input_dirs, result_dir, stride)
    alignment = compute_view_transform(extr0, scene_pts)

    extent = _scene_max_extent(scene_pts)
    if not np.isfinite(extent) or extent <= 0:
        extent = 1.0
    cam_scale = extent * 0.03

    # Viewpoints, all looking at the scene centre (the aligned origin —
    # the alignment is centred on the union of clouds + trajectory).  The
    # center view sits ``view_distance`` extents behind the first camera
    # (in the aligned glTF frame the camera looks along -z, up = +y, so
    # behind is +z); the down view translates it ``view_raise`` extents
    # upward and backward; the side views keep the center eye's distance
    # from the scene centre but swing ``view_angle`` degrees around its
    # vertical axis (left/right of the center direction — 45 deg by
    # default), and sit ``view_lower`` extents below it.
    c2w0 = np.linalg.inv(_as_homogeneous44(extr0[0]))
    cam_pos = trimesh.transform_points(c2w0[:3, 3][None], alignment)[0]
    eye_center = cam_pos + np.array([0.0, 0.0, view_distance]) * extent
    if views == 1:
        view_eyes = [("center", eye_center)]
    elif views == 4:
        swing = np.clip(np.deg2rad(float(view_angle)), 0.0, np.deg2rad(85.0))
        c, s = np.cos(swing), np.sin(swing)
        side_y = eye_center[1] - view_lower * extent
        # Swing the center eye's own vector around the scene centre's
        # vertical axis so the side eyes keep its distance from the scene:
        # left = -view_angle, right = +view_angle (y-up rotation).
        view_eyes = [
            ("center", eye_center),
            ("down", eye_center + np.array([0.0, view_raise, view_back]) * extent),
            ("left", np.array([eye_center[0] * c - eye_center[2] * s,
                               side_y,
                               eye_center[0] * s + eye_center[2] * c])),
            ("right", np.array([eye_center[0] * c + eye_center[2] * s,
                                side_y,
                                -eye_center[0] * s + eye_center[2] * c])),
        ]
    else:
        raise ValueError(f"views must be 1 or 4, got {views}")
    look_at = np.zeros(3)  # scene centre (aligned origin)

    # 2x2 grid of viewports at half the video size (single view uses the
    # full size).
    vw, vh = (w // 2, h // 2) if views == 4 else (w, h)

    # Fit each viewport's field of view so the scene fills the frame: the
    # union scene points are projected into each eye's camera frame and the
    # p90 angular half-extent per axis is mapped to ~85% of the viewport.
    # ``view_fov`` overrides the fit (vertical degrees, applied to all).
    up = np.array([0.0, 1.0, 0.0])
    pts_aligned = trimesh.transform_points(scene_pts.reshape(-1, 3), alignment)
    pts_aligned = pts_aligned[np.isfinite(pts_aligned).all(axis=1)]
    view_fovs: list[float] = []
    for _, eye in view_eyes:
        if view_fov is not None:
            view_fovs.append(view_fov)
            continue
        fwd = look_at - eye
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, up)
        right /= np.linalg.norm(right)
        up2 = np.cross(right, fwd)
        v = pts_aligned - eye
        z = v @ fwd
        keep = z > 0.01 * np.linalg.norm(eye)  # in front of, and off, the eye
        x = np.arctan2(v[keep] @ right, z[keep])
        y = np.arctan2(v[keep] @ up2, z[keep])
        hx = np.percentile(np.abs(x), 90)
        hy = np.percentile(np.abs(y), 90)
        vfov = max(
            2.0 * np.rad2deg(np.arctan(np.tan(hy) / 0.85)),
            2.0 * np.rad2deg(np.arctan(np.tan(hx) * (vh / vw) / 0.85)),
        )
        view_fovs.append(float(np.clip(vfov, 20.0, 90.0)))

    renderer = o3d.visualization.rendering.OffscreenRenderer(vw, vh)

    # Invert the renderer's filmic tone curve so video colours match the
    # source images.  Built first: the probe sets its own camera, and the
    # fitted scene cameras below must be the ones used for rendering.
    tone_lut = _build_inverse_tone_curve(renderer, vw, vh)

    renderer.scene.set_background([0.05, 0.05, 0.05, 1.0])

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
            if frame_loader is None:
                depth, extrs, intrs, images = load_pair(stem, input_dirs, result_dir)
            else:
                # Online images: geometry from the NPZs, frames from the
                # loader (no frame folders on disk).
                depths, exts, ints = [], [], []
                for img_dir in input_dirs:
                    d, e, i = load_stream_data(img_dir, result_dir, stem)
                    depths.append(d)
                    exts.append(e)
                    ints.append(i)
                depth = np.stack(depths, axis=0)
                extrs = np.stack(exts, axis=0)
                intrs = np.stack(ints, axis=0)
                images = frame_loader(stem)
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

            if views == 1:
                arr = _render_view(renderer, tone_lut, look_at, view_eyes[0][1],
                                   view_fovs[0], vw / vh)
            else:
                canvas = np.zeros((h, w, 3), dtype=np.uint8)
                for k, (name, eye) in enumerate(view_eyes):
                    row, col = divmod(k, 2)
                    y0, x0 = row * vh, col * vw
                    canvas[y0:y0 + vh, x0:x0 + vw] = _render_view(
                        renderer, tone_lut, look_at, eye, view_fovs[k], vw / vh
                    )
                    _label_viewport(canvas, name, x0 + 8, y0 + 6)
                arr = canvas
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
    parser.add_argument(
        "--view-distance", type=float, default=0.3,
        help="Eye distance behind the first camera, in scene-extent units "
             "(default: 0.3)",
    )
    parser.add_argument(
        "--views", type=int, choices=[1, 4], default=4,
        help="Viewpoints per frame: 1 (center only) or 4 (2x2 grid of "
             "center/down/left/right) (default: 4)",
    )
    parser.add_argument(
        "--view-angle", type=float, default=45.0,
        help="Side-view swing for the left/right viewpoints, in degrees off "
             "the center view around the scene's vertical axis (clamped to "
             "85; default: 45)",
    )
    parser.add_argument(
        "--view-lower", type=float, default=0.1,
        help="Downward shift of the left/right viewpoints below the center "
             "eye, in scene-extent units (default: 0.1)",
    )
    parser.add_argument(
        "--view-raise", type=float, default=0.1,
        help="Elevation of the down viewpoint, in scene-extent units "
             "(default: 0.1)",
    )
    parser.add_argument(
        "--view-back", type=float, default=0.3,
        help="Backward pull of the down viewpoint, in scene-extent units — "
             "the camera pose falls outside the auto-fitted viewport when "
             "the eye is straight above it, so the eye is pulled back to "
             "bring the frustums/path into view (default: 0.3)",
    )
    parser.add_argument(
        "--view-fov", type=float, default=None,
        help="Override the auto-fitted vertical field of view in degrees "
             "(default: auto-fit each viewport to the scene)",
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
        view_distance=args.view_distance,
        views=args.views,
        view_angle=args.view_angle,
        view_lower=args.view_lower,
        view_raise=args.view_raise,
        view_back=args.view_back,
        view_fov=args.view_fov,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
