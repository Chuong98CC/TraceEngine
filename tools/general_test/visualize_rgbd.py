"""Visualise Astribot head RGB-D data (ground-truth depth camera).

Loads the head RGB frame and its aligned depth frame via the astribot
dataloader, recovers metric depth from the grey-scale depth image, and writes:

1. Visualisation → side-by-side ``[RGB | colour-mapped depth]`` JPEG (save_depth_vis).
2. Point cloud   → RGB-coloured ``.glb`` back-projected with the head intrinsics
   (export_glb).

Usage
-----
python tools/visualize_rgbd.py --frame_index 0 --save_viz --save_glb
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from utils.astribot_dataloader import load_rgbd
from utils.visualize_depth import export_glb, save_depth_vis


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualise Astribot head RGB-D data (depth-camera ground truth).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--camera_name", type=str, default="head_rgbd",
        help="Astribot camera name (default: %(default)s)",
    )
    parser.add_argument(
        "--frame_index", default=0, type=int,
        help="Frame index to process (default: %(default)s).",
    )
    parser.add_argument(
        "--max_depth_m", default=5.0, type=float,
        help="Far-plane clip used to recover metric depth from the grey-scale "
             "depth image (default: %(default)s).",
    )
    parser.add_argument(
        "--output", "-o", default="output/rgbd", type=str,
        help="Output directory (default: %(default)s).",
    )
    parser.add_argument(
        "--save_viz", action="store_true",
        help="Save the side-by-side visualisation frame (RGB | depth colour-map).",
    )
    parser.add_argument(
        "--save_glb", action="store_true",
        help="Save the coloured point cloud as a .glb file.",
    )
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    # ---- validate frame index ----
    frame_idx = args.frame_index

    # ---- load RGB + depth + camera params ----
    rgb_path, depth_m,  ext, ixt = load_rgbd(frame_idx, args.camera_name)

    rgb_bgr = cv2.imread(rgb_path)
    if rgb_bgr is None:
        raise FileNotFoundError(f"Cannot read RGB frame: {rgb_path}")
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

    # Align RGB to the depth resolution (shared by viz + glb).
    h_d, w_d = depth_m.shape
    if rgb.shape[:2] != (h_d, w_d):
        rgb = cv2.resize(rgb, (w_d, h_d))


    out_dir = Path(args.output)
    # ---- visualisation frame ----
    if args.save_viz:
        vis_dir = save_depth_vis(
            rgb[None],
            depth_m[None],
            out_dir,
            filenames=[f"rgbd_{frame_idx:06d}.jpg"],
        )
        print(f"Saved viz : {vis_dir / f'rgbd_{frame_idx:06d}.jpg'}")

    # ---- point cloud ----
    if args.save_glb:
        pc_path = out_dir / f"rgbd_{frame_idx:06d}.glb"
        export_glb(
            depth=depth_m[None].astype(np.float32),
            intrinsics=ixt[None],
            extrinsics=ext[None],
            images_u8=rgb[None].astype(np.uint8),
            conf=None,
            out_path=str(pc_path),
        )
        print(f"Saved glb : {pc_path}")

    print("Done.")


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
