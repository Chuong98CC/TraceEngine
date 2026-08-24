"""Streaming long-video inference with the TAPIP3D ONNX models.

Mirror of infer_stream.py: same sliding-window scheduling and IO, but the
encoder/updater run as ONNX sessions (Tapip3D_ONNX) and the window
orchestration as StreamONNX. Image size and query count are auto-detected
from the ONNX graphs.

Exact-N contract: the shipped updater ONNX (weights/tapip3d_updater.onnx) has
a fixed query count (1088). The default grids produce exactly that:
8x8 bbox grid (64) + 32x32 support grid (1024). The script exits with a
clear error otherwise (e.g. depth holes in the bbox drop bbox points, or
different --grid_x/--grid_y/--support_grid_size values).

Usage:
    python infer_stream_onnx.py --image_dir data/.../frames --npz_dir data/.../geometry \
        --bbox 70 357 133 396 --grid_x 8 --grid_y 8 --support_grid_size 32
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from rich import print

from utils.streaming_utils import (scan_image_folder, load_resized_batch,
                         compute_global_depth_roi, unproject_bbox_queries)
from flow_models.tapip3d import Tapip3D_ONNX, Tapip3DStreamONNX
from flow_models.tapip3d.utils.cotracker_utils import get_grid_queries


def parse_args():
    p = argparse.ArgumentParser(description="Streaming TAPIP3D ONNX inference")
    p.add_argument("--image_dir", type=str, required=True)
    p.add_argument("--npz_dir", type=str, required=True)
    p.add_argument("--output_dir", "-o", type=str, default="output/stream_tracks_onnx")
    p.add_argument("--encoder", type=str,
                   default="weights/tapip3d/tapip3d_encoder_480x640.onnx",
                   help="Encoder ONNX path (image size auto-detected from the graph)")
    p.add_argument("--updater", type=str, default="weights/tapip3d/tapip3d_updater.onnx",
                   help="Updater ONNX path (query count auto-detected from the graph)")
    p.add_argument("--corr_forward", type=str,
                   default="weights/tapip3d/tapip3d_corr_forward.onnx",
                   help="Corr forward ONNX path (trained per-iteration corr math)")
    p.add_argument("--fps", type=int, default=1)
    p.add_argument("--start_frame", type=int, default=0)
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--bbox", type=int, nargs=4, default=None,
                   help="Bounding box: x0 y0 x1 y1")
    p.add_argument("--grid_x", type=int, default=8,
                   help="Bbox grid points in x (8x8 default + 32x32 support = "
                        "the updater ONNX's 1088 queries)")
    p.add_argument("--grid_y", type=int, default=8,
                   help="Bbox grid points in y")
    p.add_argument("--support_grid_size", type=int, default=32)
    p.add_argument("--num_iters", type=int, default=6,
                   help="Update iterations inside each window (Tapip3D_ONNX)")
    p.add_argument("--vis_threshold", type=float, default=0.5)
    p.add_argument("--visualize", action="store_true",
                   help="Render and save the visualization video after inference")
    p.add_argument("--video_fps", type=int, default=10,
                   help="Frame rate of the visualization video (with --visualize)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[bold]Scanning {args.image_dir}...[/bold]")
    file_list, frame_H, frame_W = scan_image_folder(
        args.image_dir, args.start_frame, args.fps, args.max_frames)
    total_frames = len(file_list)
    frame_indices = [idx for idx, _ in file_list]
    print(f"  {total_frames} frames, {frame_H}x{frame_W}")

    print("[bold]Loading ONNX models...[/bold]")
    onnx_model = Tapip3D_ONNX(args.encoder, args.updater, args.corr_forward,
                              num_iters=args.num_iters)
    inf_h, inf_w = onnx_model.image_size
    print(f"  Encoder: {args.encoder}")
    print(f"  Updater: {args.updater}")
    print(f"  Corr forward: {args.corr_forward}")
    print(f"  Detected resolution: {inf_w}x{inf_h}, "
          f"queries: {onnx_model.num_queries}, window: {onnx_model.seq_len} frames")

    # Match the reference wrapper's validated numerics (fp32, TF32 off).
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # --- depth ROI pre-scan (global, matches the original inference()) ------
    print("[bold]Computing global depth ROI...[/bold]")
    depth_roi = compute_global_depth_roi(args.npz_dir, file_list, inf_h, inf_w)

    # --- first batch: queries anchored at global frame 0 ---------------------
    batch0 = load_resized_batch(file_list, args.npz_dir, 0,
                                min(onnx_model.seq_len, total_frames),
                                inf_h, inf_w)
    x0, y0, x1, y1 = args.bbox if args.bbox else (0, 0, frame_W - 1, frame_H - 1)
    queries = unproject_bbox_queries(
        x0, y0, x1, y1, args.grid_x, args.grid_y,
        batch0[1][0], batch0[2][0], batch0[3][0], inf_h, inf_w, device="cpu")
    if queries is None:
        print("[red]ERROR: no valid query points (all depths == 0 in bbox).[/red]")
        sys.exit(1)
    num_queries = queries.shape[0]
    print(f"[bold]Tracking {num_queries} bbox points[/bold]")

    if args.support_grid_size > 0:
        support = get_grid_queries(
            args.support_grid_size,
            batch0[1][None], batch0[2][None], batch0[3][None]).squeeze(0)
        queries = torch.cat([queries, support], dim=0)
        print(f"  + {support.shape[0]} support grid points "
              f"(grid {args.support_grid_size}x{args.support_grid_size})")

    if queries.shape[0] != onnx_model.num_queries:
        print(f"[red]ERROR: {queries.shape[0]} queries do not match the updater "
              f"ONNX's fixed {onnx_model.num_queries} queries. Adjust "
              f"--grid_x/--grid_y/--support_grid_size, or pick a --bbox whose "
              f"points all have valid depth (depth holes drop points), or "
              f"re-export the updater for this query count.[/red]")
        sys.exit(1)

    # --- streamed inference --------------------------------------------------
    def batches():
        for s in range(0, total_frames, onnx_model.seq_len):
            yield load_resized_batch(file_list, args.npz_dir, s,
                                     min(s + onnx_model.seq_len, total_frames),
                                     inf_h, inf_w)

    si = Tapip3DStreamONNX(onnx_model, queries, depth_roi=depth_roi)
    # No autocast here (unlike infer_stream.py): the encoder/updater run as
    # fp32 ORT sessions and the corr pipeline was validated at fp32.
    with torch.inference_mode():
        coords_all, visibs_logits_all = si.run(batches(), total_frames)

    # --- slice to bbox queries and save -------------------------------------
    coords = coords_all[:, :num_queries].cpu().numpy()
    visibs = (torch.sigmoid(visibs_logits_all[:, :num_queries]) >=
              args.vis_threshold).cpu().numpy()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "coords.npy", coords)
    np.save(out_dir / "visibs.npy", visibs)

    meta = {
        "image_dir": str(Path(args.image_dir).absolute()),
        "npz_dir": str(Path(args.npz_dir).absolute()),
        "total_frames": total_frames,
        "num_queries": int(num_queries),
        "inference_resolution": [inf_h, inf_w],
        "bbox": [x0, y0, x1, y1] if args.bbox else None,
        "grid": [args.grid_x, args.grid_y],
        "support_grid_size": args.support_grid_size,
        "num_iters": args.num_iters,
        "model": {
            "encoder": str(Path(args.encoder).absolute()),
            "updater": str(Path(args.updater).absolute()),
            "corr_forward": str(Path(args.corr_forward).absolute()),
        },
        "frame_indices": [int(i) for i in frame_indices],
        "frame_files": [
            {"index": int(idx), "path": str(Path(path).absolute())}
            for idx, path in file_list
        ],
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Saved: coords {coords.shape}, visibs {visibs.shape}, "
          f"visible {visibs.mean():.2%}")
    print(f"[bold green]Done! Results saved to {out_dir.resolve()}[/bold green]")

    if args.visualize:
        from utils.visualize_tapip3d import render_tracks
        video_path = render_tracks(out_dir, fps=args.video_fps)
        print(f"[bold green]Video saved to {video_path}[/bold green]")


if __name__ == "__main__":
    main()
