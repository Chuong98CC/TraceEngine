"""MoGe v3 inference on a single image, via the torch.export (PT2) runtime:
depth image (.npy/.png) + coloured point cloud (.glb).

Loads the exported dense graph + the eager sparse refiner, runs MoGe v3 on a
single image and exports:
  - metric depth as .npy and a colour-coded .png (Spectral_r, min-max
    normalized over valid pixels),
  - a coloured point cloud (.glb) back-projected with the model's estimated
    intrinsics and identity extrinsics (monocular output is in camera space;
    masked-out pixels are zeroed).

The graph is compiled for a fixed 640x480 input (non-preserving resize), so
any input size is accepted; depth is output at that fixed resolution.

Examples:
  python tools/general_test/module/infer_moge3.py \
    -i assets/astribot_test_imgs/head_stereo_left/frame_000210.jpg \
    --out_dir ./output/moge3
"""

import argparse
import time
from pathlib import Path
import os
from typing import Union, Optional
import cv2
import matplotlib
import numpy as np
from PIL import Image

from depth_models.moge3.moge_pt2 import MoGev3_PT2
from utils.visualize.visualize_depth import export_glb

def save_glb(
    save_path: Union[str, os.PathLike],
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_uvs: np.ndarray,
    texture: np.ndarray,
    vertex_normals: Optional[np.ndarray] = None,
):
    import trimesh
    import trimesh.visual
    from PIL import Image

    trimesh.Trimesh(
        vertices=vertices,
        vertex_normals=vertex_normals,
        faces=faces,
        visual = trimesh.visual.texture.TextureVisuals(
            uv=vertex_uvs,
            material=trimesh.visual.material.PBRMaterial(
                baseColorTexture=Image.fromarray(texture),
                metallicFactor=0.5,
                roughnessFactor=1.0
            )
        ),
        process=False
    ).export(save_path)


def save_ply(
    save_path: Union[str, os.PathLike],
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_colors: np.ndarray,
    vertex_normals: Optional[np.ndarray] = None,
):
    import trimesh
    import trimesh.visual
    from PIL import Image

    trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_colors=vertex_colors,
        vertex_normals=vertex_normals,
        process=False
    ).export(save_path)

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
        description="MoGe v3 monocular metric depth on a single image -> point cloud"
    )
    parser.add_argument("-i", "--input", required=True, help="Input image path (jpg/png).")
    parser.add_argument(
        "--pt2", type=str, default="weights/moge3/moge3_l.pt2",
        help="Path to the exported graph checkpoint.",
    )
    parser.add_argument(
        "--refiner", type=str, default=None,
        help="Path to the refiner companion checkpoint (defaults to the .pt2 path with _refiner.pt).",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device (must be CUDA).")
    parser.add_argument("--refine_steps", type=int, default=3, help="Sparse refinement steps.")
    parser.add_argument("--out_dir", type=str, default="./output/moge3")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = MoGev3_PT2(args.pt2, refiner_path=args.refiner, device=args.device,
                       refine_steps=args.refine_steps)

    # ---- load input image: cv2 frames are BGR, flip to RGB at the model
    # boundary (repo-wide RGB pixel-space contract — the model decodes any
    # ImageInput via utils.image_io; image_rgb is reused for the GLB) ----
    image_bgr = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {args.input}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    print(f"[input] {args.input}  size={image_bgr.shape[1]}x{image_bgr.shape[0]}")

    t0 = time.perf_counter()
    out = model.infer(image_rgb)
    elapsed = time.perf_counter() - t0
    print(f"inference took {elapsed * 1000:.1f} ms")

    # out: batched fp32 CUDA tensors; depth/mask at the graph's fixed size.
    pred = out["depth"].squeeze(0).cpu().numpy().astype(np.float32)  # (H, W)
    # masked-out pixels come back as +inf -> zero them like the other tools.
    pred[~np.isfinite(pred)] = 0.0

    # ---- colour source for the point cloud, at the depth resolution ----
    rgb_u8 = image_rgb
    if rgb_u8.shape[:2] != pred.shape:
        rgb_u8 = cv2.resize(rgb_u8, (pred.shape[1], pred.shape[0]))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_base = out_dir / Path(args.input).stem
    _save_depth_outputs(pred, out_base, False)
    print(f"Saved depth: {out_base}.npy, {out_base}.png")

    # ---- point cloud: model-estimated intrinsics, identity extrinsics ----
    K = out["intrinsics"].squeeze(0).cpu().numpy().astype(np.float64)  # (3, 3)
    glb_path = out_dir / f"{Path(args.input).stem}.glb"
    export_glb(
        depth=pred[None],
        intrinsics=K[None],
        extrinsics=np.eye(4, dtype=np.float64)[None],  # camera space
        images_u8=rgb_u8[None].astype(np.uint8),
        conf=None,
        out_path=str(glb_path),
    )
    print(f"Saved glb : {glb_path}")

    valid = pred > 0
    if valid.any():
        print(
            f"pred depth: {pred[valid].mean():.2f}+/-{pred[valid].std():.2f}m  "
            f"valid={valid.mean():.1%}"
        )
    else:
        print("Warning: no valid predicted depth pixels.")


if __name__ == "__main__":
    main()
