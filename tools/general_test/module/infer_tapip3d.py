"""Streaming long-video inference with the TAPIP3D torch.export (pt2) models.

Mirror of infer_stream.py: same sliding-window scheduling and IO, but the
encoder and the fused corr+updater iteration run as torch.export programs
(Tapip3D_PT2) and the window orchestration as Tapip3DStreamPT2. Image size
and query count are derived from the graph input shapes.

Exact-N contract: the shipped iteration program
(weights/tapip3d/tapip3d_iteration_1088_bf16.pt2) has a fixed query count
(1088). The default grids produce exactly that: 8x8 bbox grid (64) + 32x32
support grid (1024). The script exits with a clear error otherwise (e.g.
depth holes in the bbox drop bbox points, or different
--grid_x/--grid_y/--support_grid_size values).

SAM3 mask sampling: when --bbox is given, the box query points are sampled
uniformly at random inside the SAM3 segmentation of the box on the first
frame (box + optional --text_prompt, default "visual") instead of on a
regular grid. Sampling is restricted to mask pixels with valid depth, so
all sampled points survive the unprojection filter and the query count is
exact. If SAM3 produces no mask, or the mask has fewer valid-depth pixels
than grid_x*grid_y, the script falls back to the regular bbox grid.

Usage:
    python infer_tapip3d.py --image_dir data/.../frames --depth_dir data/.../depth \
        --bbox 70 357 133 396 --grid_x 8 --grid_y 8 --support_grid_size 32 \
        --sam3_checkpoint weights/sam3/sam3_image_exported_bf16.pt2 \
        --text_prompt visual
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
                         compute_global_depth_roi, unproject_bbox_queries,
                         unproject_xy_queries)
from flow_models.tapip3d.utils import (Tapip3D_PT2, Tapip3DStreamPT2,
                             _DEFAULT_ENCODER, _DEFAULT_ITERATION)
from flow_models.tapip3d.utils._grid_utils import get_grid_queries
from det_seg_models.sam3 import Sam3Image, normalize_bbox
from det_seg_models.sam3.utils import box_xyxy_to_cxcywh


def parse_args():
    p = argparse.ArgumentParser(description="Streaming TAPIP3D torch.export inference")
    p.add_argument("--image_dir", type=str, required=True)
    p.add_argument("--depth_dir", type=str, required=True,
                   help="Directory with per-frame depth (.lz4) + camera pose (.npz)")
    p.add_argument("--output_dir", "-o", type=str, default="output/stream_tracks_pt2")
    p.add_argument("--encoder", type=str, default=_DEFAULT_ENCODER,
                   help="Encoder .pt2 path (image size asserted against --image_size)")
    p.add_argument("--iteration", type=str, default=_DEFAULT_ITERATION,
                   help="Iteration .pt2 path (query count auto-detected from the graph)")
    p.add_argument("--image_size", type=int, nargs=2, default=[480, 640],
                   help="Inference resolution (H W), must match the encoder graph")
    p.add_argument("--interval", type=int, default=1,
                   help="Process every Nth frame (frame sampling interval)")
    p.add_argument("--start_frame", type=int, default=0)
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--bbox", type=int, nargs=4, default=None,
                   help="Bounding box: x0 y0 x1 y1")
    p.add_argument("--grid_x", type=int, default=8,
                   help="Bbox grid points in x (8x8 default + 32x32 support = "
                        "the iteration graph's 1088 queries)")
    p.add_argument("--grid_y", type=int, default=8,
                   help="Bbox grid points in y")
    p.add_argument("--support_grid_size", type=int, default=32)
    p.add_argument("--sam3_checkpoint", type=str,
                   default="weights/sam3/sam3_image_exported_bf16.pt2",
                   help="SAM3 image .pt2 checkpoint, used to segment the box "
                        "on the first frame when --bbox is given")
    p.add_argument("--text_prompt", type=str, default="visual",
                   help="SAM3 text prompt for the box segmentation "
                        "(default 'visual')")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for the in-mask query sampling")
    p.add_argument("--num_iters", type=int, default=6,
                   help="Fused corr+updater iterations inside each window (Tapip3D_PT2)")
    p.add_argument("--vis_threshold", type=float, default=0.5)
    p.add_argument("--visualize", action="store_true",
                   help="Render and save the visualization video after inference")
    p.add_argument("--video_fps", type=int, default=10,
                   help="Frame rate of the visualization video (with --visualize)")
    return p.parse_args()


def segment_first_frame_mask(sam3, image_t, bbox, text_prompt):
    """Segment the object in the box on the first frame with SAM3.

    Args:
        sam3: Sam3Image instance
        image_t: (3, H, W) float [0, 1] frame at the inference resolution
        bbox: (x0, y0, x1, y1) in the same pixel space
        text_prompt: SAM3 text prompt (default "visual")

    Returns:
        (H, W) bool mask — union of all returned masks — or None if nothing
        was segmented.
    """
    x0, y0, x1, y1 = bbox
    h, w = image_t.shape[-2:]
    box_cxcywh = box_xyxy_to_cxcywh(
        torch.tensor([[x0, y0, x1, y1]], dtype=torch.float32))
    norm_boxes = normalize_bbox(box_cxcywh, w, h)

    # Restore the SAM3 trace-time environment around the call: the script
    # runs the TAPIP3D path with TF32 off, and importing flow_models.tapip3d
    # globally disables the flash/mem-efficient SDPA kernels — both break the
    # SAM3 exported graph (traced with TF32 on and the SDPA kernels enabled,
    # e.g. its attention output is viewed with trace-time strides).
    flash_sdp = torch.backends.cuda.flash_sdp_enabled()
    mem_eff_sdp = torch.backends.cuda.mem_efficient_sdp_enabled()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    try:
        state = sam3.predict(image_t, text_prompt=text_prompt,
                             boxes=norm_boxes, labels=[True])
    finally:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(flash_sdp)
        torch.backends.cuda.enable_mem_efficient_sdp(mem_eff_sdp)

    masks = state["masks"]  # (N, 1, H, W) bool
    if masks.shape[0] == 0:
        return None
    return masks.any(dim=0)[0]


def save_mask_visualization(out_dir, frame_t, mask, bbox, xy, text_prompt):
    """Save the SAM3 mask (npy) and a first-frame overlay (png) to out_dir.

    The overlay shows the first frame with the box, the segmented mask and
    the sampled query points. Returns the (npy, png) paths saved.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from utils.visualize.visualize_mask import to_pil, draw_box_on_image, plot_mask

    out_dir = Path(out_dir)
    np.save(out_dir / "sam3_mask.npy", mask.cpu().numpy())

    img = to_pil(frame_t)  # float CHW [0, 1] -> PIL RGB
    img = draw_box_on_image(img, bbox, color=(0, 255, 0))
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plot_mask(mask.cpu(), color="red", ax=plt.gca())
    plt.scatter(xy[:, 0], xy[:, 1], s=6, c="yellow", marker=".")
    plt.title(f"SAM3 mask ({text_prompt!r}): {int(mask.sum())} px, "
              f"{xy.shape[0]} query points")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "sam3_mask.png", dpi=120)
    plt.close()
    return out_dir / "sam3_mask.npy", out_dir / "sam3_mask.png"


def sample_points_in_mask(mask, depth, n, rng):
    """Randomly sample n pixel coordinates inside a segmentation mask.

    Samples uniformly without replacement from mask pixels whose depth is
    valid (> 0), so every sampled point survives the unprojection filter.

    Args:
        mask: (H, W) bool tensor
        depth: (H, W) depth map (numpy or torch)
        n: number of points to sample
        rng: numpy Generator

    Returns:
        (n, 2) int array of (x, y) pixel coordinates, or None if fewer than
        n candidates exist.
    """
    valid = (mask.cpu() & (depth > 0)).numpy()
    ys, xs = np.nonzero(valid)
    if xs.size < n:
        return None
    idx = rng.choice(xs.size, size=n, replace=False)
    return np.stack([xs[idx], ys[idx]], axis=-1)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[bold]Scanning {args.image_dir}...[/bold]")
    file_list, frame_H, frame_W = scan_image_folder(
        args.image_dir, args.start_frame, args.interval, args.max_frames)
    total_frames = len(file_list)
    frame_indices = [idx for idx, _ in file_list]
    print(f"  {total_frames} frames, {frame_H}x{frame_W}")

    print("[bold]Loading PT2 models...[/bold]")
    pt2_model = Tapip3D_PT2(args.encoder, args.iteration,
                            image_size=tuple(args.image_size),
                            num_iters=args.num_iters)
    inf_h, inf_w = pt2_model.image_size
    print(f"  Encoder: {args.encoder}")
    print(f"  Iteration: {args.iteration}")
    print(f"  Detected resolution: {inf_w}x{inf_h}, "
          f"queries: {pt2_model.num_queries}, window: {pt2_model.seq_len} frames")

    # Match the reference wrapper's validated numerics (fp32, TF32 off).
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # --- depth ROI pre-scan (global, matches the original inference()) ------
    print("[bold]Computing global depth ROI...[/bold]")
    depth_roi = compute_global_depth_roi(args.depth_dir, file_list, inf_h, inf_w)

    # --- first batch: queries anchored at global frame 0 ---------------------
    batch0 = load_resized_batch(file_list, args.depth_dir, 0,
                                min(pt2_model.seq_len, total_frames),
                                inf_h, inf_w)
    x0, y0, x1, y1 = args.bbox if args.bbox else (0, 0, frame_W - 1, frame_H - 1)

    # SAM3 mask-guided sampling: segment the object in the box on the first
    # frame and sample the bbox query points uniformly inside the mask. Falls
    # back to the regular bbox grid when the mask is missing or too small.
    query_source = "grid"
    bbox_xy = None
    if args.bbox is not None:
        sam3 = Sam3Image(args.sam3_checkpoint)
        mask = segment_first_frame_mask(sam3, batch0[0][0], args.bbox,
                                        args.text_prompt)
        if mask is None:
            print(f"[yellow]WARNING: SAM3 segmented nothing in the box "
                  f"(text {args.text_prompt!r}); falling back to the bbox "
                  f"grid.[/yellow]")
        else:
            n_points = args.grid_x * args.grid_y
            bbox_xy = sample_points_in_mask(
                mask, batch0[1][0], n_points, np.random.default_rng(args.seed))
            if bbox_xy is None:
                print(f"[yellow]WARNING: mask has < {n_points} valid-depth "
                      f"pixels; falling back to the bbox grid.[/yellow]")
            else:
                query_source = "mask"
                print(f"[bold]SAM3 mask: {int(mask.sum())} pixels, sampling "
                      f"{bbox_xy.shape[0]} query points in the box "
                      f"(text {args.text_prompt!r})[/bold]")
                mask_files = save_mask_visualization(
                    args.output_dir, batch0[0][0], mask, args.bbox, bbox_xy,
                    args.text_prompt)
                print(f"  Saved mask visualization: {mask_files[1].name} "
                      f"({mask_files[1].stat().st_size // 1024} KB)")

    if query_source == "mask":
        queries = unproject_xy_queries(
            bbox_xy, batch0[1][0], batch0[2][0], batch0[3][0], device="cpu")
    else:
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

    if queries.shape[0] != pt2_model.num_queries:
        print(f"[red]ERROR: {queries.shape[0]} queries do not match the iteration "
              f"graph's fixed {pt2_model.num_queries} queries. Adjust "
              f"--grid_x/--grid_y/--support_grid_size, or pick a --bbox whose "
              f"points all have valid depth (depth holes drop points), or "
              f"re-export the iteration program for this query count.[/red]")
        sys.exit(1)

    # --- streamed inference --------------------------------------------------
    def batches():
        for s in range(0, total_frames, pt2_model.seq_len):
            yield load_resized_batch(file_list, args.depth_dir, s,
                                     min(s + pt2_model.seq_len, total_frames),
                                     inf_h, inf_w)

    si = Tapip3DStreamPT2(pt2_model, queries, depth_roi=depth_roi)
    # No autocast here (unlike infer_stream.py): the eager corr prep runs at
    # fp32 and the .pt2 programs run in their exported dtype (bf16) with the
    # boundary casts inside Tapip3D_PT2.
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
        "depth_dir": str(Path(args.depth_dir).absolute()),
        "total_frames": total_frames,
        "num_queries": int(num_queries),
        "inference_resolution": [inf_h, inf_w],
        "bbox": [x0, y0, x1, y1] if args.bbox else None,
        "query_source": query_source,
        "sam3_mask_files": [str(f.name) for f in mask_files]
                            if query_source == "mask" else None,
        "text_prompt": args.text_prompt if args.bbox else None,
        "sam3_checkpoint": (str(Path(args.sam3_checkpoint).absolute())
                            if args.bbox else None),
        "seed": args.seed,
        "grid": [args.grid_x, args.grid_y],
        "support_grid_size": args.support_grid_size,
        "num_iters": args.num_iters,
        "model": {
            "encoder": str(Path(args.encoder).absolute()),
            "iteration": str(Path(args.iteration).absolute()),
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
        from utils.visualize.visualize_tapip3d import render_tracks
        video_path = render_tracks(out_dir, fps=args.video_fps)
        print(f"[bold green]Video saved to {video_path}[/bold green]")


if __name__ == "__main__":
    main()
