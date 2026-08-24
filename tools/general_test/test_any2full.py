# -*- coding: utf-8 -*-
"""
test_any2full.py — Any2Full inference on Astribot head RGB-D frames, via the
torch.export (PT2) runtime, exporting a coloured point cloud (.glb).

Loads an Astribot head frame (RGB path + grayscale-derived metric depth +
camera params) through utils.astribot_dataloader.load_rgbd_head, runs
Any2Full_PT2 (preprocess -> infer -> postprocess), then back-projects the
predicted depth into a point cloud with utils.visualize_depth.export_glb.

Inputs are resized to the exported (fixed) 480x640, so any RGB/depth size is
accepted; output depth (and the point cloud) is at the exported resolution.

Examples:
  python tools/general_test/test_any2full.py \
    --pt2 weights/any2full/Any2Full_vitl_bf16.pt2 \
    --frame_idx 0 \
    --out_dir ./outputs
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from depth_models.a3f.any2full import Any2Full_PT2
from utils.astribot_dataloader import load_rgbd
from utils.visualize_depth import export_glb
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

def _save_depth_outputs(depth: np.ndarray, out_base: Path, grayscale: bool) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_base) + ".npy", depth)

    depth_min = float(np.min(depth))
    depth_max = float(np.max(depth))
    if depth_max > depth_min:
        depth_norm = (depth - depth_min) / (depth_max - depth_min)
    else:
        depth_norm = np.zeros_like(depth)

    depth_img = (depth_norm * 255.0).astype(np.uint8)
    if grayscale:
        depth_img = np.repeat(depth_img[..., np.newaxis], 3, axis=-1)
    else:
        cmap = matplotlib.colormaps.get_cmap("Spectral_r")
        depth_img = (cmap(depth_img)[:, :, :3] * 255).astype(np.uint8)

    Image.fromarray(depth_img).save(str(out_base) + ".png")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Any2Full PT2 RGBD inference on Astribot head frames -> point cloud"
    )
    parser.add_argument(
        "--pt2", type=str, default="weights/any2full/Any2Full_vitl_bf16.pt2",
        help="Torch.Export checkpoint path",
    )
    parser.add_argument(
        "--camera_name", type=str, default="head_rgbd",
        help="Astribot camera name (default: %(default)s)",
    )
    parser.add_argument(
        "--frame_idx", type=int, default=0,
        help="Astribot head frame index (default: %(default)s)",
    )

    parser.add_argument("--out_dir", type=str, default="./output/a2f")
    parser.add_argument("--denoise", action="store_true")
    parser.add_argument("--denoise_threshold", type=float, default=2.0)
    parser.add_argument("--denoise_kernel_size", type=int, default=None)
    parser.add_argument("--denoise_min_valid", type=int, default=5)
    parser.add_argument("--init_scaling", type=bool, default=True)
    parser.add_argument("--max_depth", type=float, default=10)
    parser.add_argument("--min_depth", type=float, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Any2Full_PT2(
        pt2_path=args.pt2,
        device=device,
        init_scaling=args.init_scaling,
        max_depth=args.max_depth,
        min_depth=args.min_depth,
    )

    # ---- load Astribot head frame: rgb path + metric depth + camera params ----
    rgb_path, depth_metrics, ext, ixt = load_rgbd(args.frame_idx, args.camera_name)
    print(f"[{args.frame_idx}] {rgb_path}")

    # depth_metrics is already physical metres; Any2Full_PT2's ndarray branch
    # treats it as such (depth_scale is not applied).
    rgb, dep = model.preprocess(
        rgb_path, depth_metrics,
        denoise=args.denoise,
        denoise_kwargs={
            "min_valid": args.denoise_min_valid,
            "threshold": args.denoise_threshold,
            "kernel_size": args.denoise_kernel_size,
        },
    )

    with torch.inference_mode():
        depth, disparity_pre_internal, prompt_depth_resized = model.infer(rgb, dep)
        pred = model.postprocess(depth, disparity_pre_internal, prompt_depth_resized)

    pred = pred.squeeze(0).squeeze(0).cpu().numpy()

    # ---- colour source for the point cloud ----
    rgb_bgr = cv2.imread(rgb_path)
    if rgb_bgr is None:
        raise FileNotFoundError(f"Cannot read RGB frame: {rgb_path}")
    rgb_u8 = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    if rgb_u8.shape[:2] != pred.shape:
        rgb_u8 = cv2.resize(rgb_u8, (pred.shape[1], pred.shape[0]))

    out_path = Path(args.out_dir) / f"frame_{args.frame_idx:06d}.glb"
    export_glb(
        depth=pred[None].astype(np.float32),
        intrinsics=ixt[None],
        extrinsics=ext[None],
        images_u8=rgb_u8[None].astype(np.uint8),
        conf=None,
        out_path=str(out_path),
    )

    valid = pred > 0
    if valid.any():
        print(
            f"pred depth: {pred[valid].mean():.2f}+/-{pred[valid].std():.2f}m  "
            f"valid={valid.mean():.1%}"
        )
    else:
        print("Warning: no valid predicted depth pixels.")
    print(f"Saved glb : {out_path}")


    out_base = Path(args.out_dir) / Path(rgb_path).stem
    _save_depth_outputs(pred, out_base, False)

if __name__ == "__main__":
    main()
