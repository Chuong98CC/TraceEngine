"""
Streaming inference results from infer_stream.py.

Reads the single continuous output (coords.npy, visibs.npy, metadata.json)
and renders trajectories over the original frames.

Usage:
    python visualize_long.py output/torso_tapip3d --fps 10
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
from einops import repeat
from tqdm import tqdm

from flow_models.tapip3d.utils._common import batch_project


def parse_args():
    p = argparse.ArgumentParser(
        description="Visualize streaming TAPIP3D inference results")
    p.add_argument("output_dir", type=str,
                   help="Output directory from infer_stream.py")
    p.add_argument("--fps", type=int, default=10,
                   help="Output video frame rate")
    p.add_argument("--output", "-o", type=str, default=None,
                   help="Output video path (default: <output_dir>/tracks.mp4)")
    p.add_argument("--trail_len", type=int, default=30,
                   help="Number of trailing frames to draw per point")
    p.add_argument("--point_size", type=int, default=2,
                   help="Radius of tracked points")
    p.add_argument("--no_trail", action="store_true",
                   help="Disable drawing trails")
    return p.parse_args()


def draw_tracks(frame, points_2d, visibs, trail_points, trail_visibs,
                point_size=2, trail_alpha=0.4):
    """Draw current points and trailing paths on a frame."""
    H, W = frame.shape[:2]
    overlay = frame.copy()

    # Draw trails
    if trail_points:
        for i in range(points_2d.shape[0]):
            trail = [(p[i] for p in trail_points) if len(trail_points) > 0 else []]
            pts = []
            for tp, tv in zip(trail_points, trail_visibs):
                if tv[i]:
                    x, y = tp[i]
                    if 0 <= x < W and 0 <= y < H:
                        pts.append((int(x), int(y)))
            for j in range(1, len(pts)):
                alpha = trail_alpha * (j / max(len(pts), 1))
                color = (0, int(255 * alpha), int(255 * (1 - alpha)))
                cv2.line(overlay, pts[j - 1], pts[j], color, 1, cv2.LINE_AA)

    # Draw current points
    for i in range(points_2d.shape[0]):
        if visibs[i]:
            x, y = points_2d[i]
            if 0 <= x < W and 0 <= y < H:
                cv2.circle(overlay, (int(x), int(y)), point_size,
                          (0, 255, 0), -1, cv2.LINE_AA)
        else:
            x, y = points_2d[i]
            if 0 <= x < W and 0 <= y < H:
                cv2.circle(overlay, (int(x), int(y)), point_size,
                          (0, 0, 255), -1, cv2.LINE_AA)

    return cv2.addWeighted(frame, 0.3, overlay, 0.7, 0)


def main():
    args = parse_args()
    out_path = render_tracks(
        Path(args.output_dir), fps=args.fps, output=args.output,
        trail_len=args.trail_len, point_size=args.point_size,
        no_trail=args.no_trail)
    print(f"Done! Video saved to {out_path}")


def render_tracks(output_dir, fps=10, output=None, trail_len=30,
                  point_size=2, no_trail=False):
    """Render trajectories from a stream output directory to a video file.

    Args:
        output_dir: path to the infer_stream.py output directory
        fps: output video frame rate
        output: output video path (default: <output_dir>/tracks.mp4)
        trail_len, point_size, no_trail: rendering options

    Returns:
        Path to the rendered video.
    """
    output_dir = Path(output_dir)

    meta_path = output_dir / "metadata.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"metadata.json not found in {output_dir}")

    with open(meta_path) as f:
        meta = json.load(f)

    image_dir = meta["image_dir"]
    # metadata key is "depth_dir" (depth .lz4 + pose .npz folder) since the
    # infer tool rename; accept the legacy "npz_dir" key for older outputs
    depth_dir = meta.get("depth_dir", meta.get("npz_dir"))

    # Resolve relative paths: try as-is, then relative to output dir, then cwd
    def _resolve_dir(path_str, output_dir):
        p = Path(path_str)
        if p.is_dir():
            return str(p)
        # Try relative to output dir
        p2 = output_dir / path_str
        if p2.is_dir():
            return str(p2.resolve())
        # Try relative to cwd
        p3 = Path.cwd() / path_str
        if p3.is_dir():
            return str(p3.resolve())
        return path_str

    image_dir = _resolve_dir(image_dir, output_dir)
    depth_dir = _resolve_dir(depth_dir, output_dir)
    if not Path(image_dir).is_dir():
        print(f"ERROR: image_dir not found: {image_dir}")
        sys.exit(1)
    if not Path(depth_dir).is_dir():
        print(f"ERROR: depth_dir not found: {depth_dir}")
        sys.exit(1)

    inf_h, inf_w = meta["inference_resolution"]

    print(f"Loading {meta['total_frames']} frames...")
    coords = np.load(output_dir / "coords.npy")       # (T, N, 3)
    visibs = np.load(output_dir / "visibs.npy")       # (T, N) bool
    frame_indices = meta["frame_indices"]
    frame_paths = [f["path"] for f in meta["frame_files"]]
    num_queries = meta["num_queries"]
    T_out = len(frame_indices)
    print(f"Loaded: {T_out} frames, {num_queries} points")

    # Probe original frame dimensions from the first frame path
    first_img_path = frame_paths[0]
    if not Path(first_img_path).is_file():
        print(f"ERROR: first frame not found: {first_img_path}")
        sys.exit(1)

    first_img = cv2.imread(first_img_path)
    orig_h, orig_w = first_img.shape[:2]
    print(f"Original resolution: {orig_w}x{orig_h}")

    # Output video
    if output:
        out_path = output
    else:
        out_path = str(output_dir / "tracks.mp4")

    # libx264/yuv420p requires even dimensions (the old mp4v writer accepted
    # odd sizes) — pad odd sizes by 1 with a black bar, like
    # visualize_stream.py does.  H.264 is used instead of mp4v because
    # Chromium-based viewers (e.g. VSCode's preview) cannot decode
    # MPEG-4 Part 2 in an mp4 container.
    out_w, out_h = orig_w - orig_w % 2, orig_h - orig_h % 2
    odd_pad = (out_w, out_h) != (orig_w, orig_h)
    # imageio expects RGB (cv2's writer took BGR); convert per frame.
    writer = imageio.get_writer(
        out_path, fps=fps, codec="libx264", quality=8, macro_block_size=1
    )

    trail_len = trail_len if not no_trail else 0
    trail_coords = []  # list of (N, 2) arrays
    trail_visibs = []  # list of (N,) arrays

    print(f"Rendering {T_out} frames to {out_path}...")

    for t in tqdm(range(T_out)):
        frame_idx = frame_indices[t]

        # Load original frame from stored path
        img_path = frame_paths[t]
        frame_bgr = cv2.imread(img_path)
        if frame_bgr is None:
            continue

        # Load geometry from NPZ
        npz_path = Path(depth_dir) / f"frame_{frame_idx:06d}.npz"
        if not npz_path.is_file():
            continue
        data = dict(np.load(str(npz_path), allow_pickle=True))
        # repo pose-npz keys are plural (extrinsics 3x4 w2c / intrinsics);
        # accept the legacy singular 4x4 form too and pad 3x4 to 4x4
        # (batch_project divides by the 4th homogeneous component)
        extr = data["extrinsics"] if "extrinsics" in data else data["extrinsic"]
        if extr.shape == (3, 4):
            extr = np.vstack([extr, [0, 0, 0, 1]])
        intr = data["intrinsics"].copy() if "intrinsics" in data else data["intrinsic"].copy()
        intr = intr.astype(np.float32)
        extr = extr.astype(np.float32)

        # Scale intrinsics to inference resolution
        scale_y = (inf_h - 1) / (orig_h - 1)
        scale_x = (inf_w - 1) / (orig_w - 1)
        intr[0, :] *= scale_x
        intr[1, :] *= scale_y

        # Project: world coords → camera → 2D pixel (in inference resolution)
        coords_t = torch.from_numpy(coords[t:t + 1]).float()
        intr_t = torch.from_numpy(intr).float()
        extr_t = torch.from_numpy(extr).float()

        points_2d = batch_project(
            coords_t[None],
            repeat(intr_t, 'i j -> 1 1 n i j', n=num_queries),
            repeat(extr_t, 'i j -> 1 1 n i j', n=num_queries),
        )
        points_2d = points_2d.squeeze().numpy()
        vis_t = visibs[t]

        # Scale 2D points from inference resolution to original resolution
        points_2d[:, 0] *= (orig_w - 1) / (inf_w - 1)
        points_2d[:, 1] *= (orig_h - 1) / (inf_h - 1)

        # Draw
        frame_out = draw_tracks(
            frame_bgr, points_2d, vis_t,
            trail_coords[-trail_len:] if trail_len > 0 else [],
            trail_visibs[-trail_len:] if trail_len > 0 else [],
            point_size=point_size,
        )

        if odd_pad:
            canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
            canvas[:orig_h, :orig_w] = frame_out
            frame_out = canvas
        writer.append_data(cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB))

        # Update trails
        trail_coords.append(points_2d)
        trail_visibs.append(vis_t)

    writer.close()
    return out_path


if __name__ == "__main__":
    main()
