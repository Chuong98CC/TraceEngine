"""MoGe v3 inference on a folder of images, via the torch.export (PT2) runtime.

Runs the exported dense graph + the eager sparse refiner on each image of an
input folder — or on a single image file, or on one image of a folder
selected by index into the sorted file list. By default nothing is written to
disk; per-image outputs are opt-in:
  - --save-depth: <stem>.npz with the metric depth (H, W fp32 metres) and the
    model's estimated intrinsics (3x3, rescaled to pixel units),
  - --visualize: a colour-coded depth .png (Spectral_r heatmap, min-max
    normalized over valid pixels) + two textured .glb meshes — the export_glb
    mesh back-projected with the estimated intrinsics and identity
    extrinsics (camera space; masked-out pixels zeroed), and an
    official-MoGe-recipe reference mesh ({stem}_mesh.glb) built straight from
    the model's camera-space points with utils3d (depth-edge-cleaned grid
    mesh; uncentred).

The graph is compiled for a fixed 640x480 input (non-preserving resize), so
any input size is accepted; depth is output at that fixed resolution.

Examples:
  # whole folder, data only (one npz per image)
  python tools/general_test/module/infer_moge3.py \
    -i assets/astribot_test_imgs/head_stereo_left --save-depth \
    --out_dir ./output/moge3

  # single image file, full outputs
  python tools/general_test/module/infer_moge3.py \
    -i assets/astribot_test_imgs/head_stereo_left/frame_000210.jpg \
    --save-depth --visualize --out_dir ./output/moge3

  # one frame of a folder: frame-idx is a 0-based index into the sorted
  # file list (not a frame number)
  python tools/general_test/module/infer_moge3.py \
    -i assets/astribot_test_imgs/head_stereo_left --frame-idx 2 \
    --save-depth --visualize --out_dir ./output/moge3
"""

import argparse
import os
import time
from pathlib import Path
from typing import Optional, Union

import cv2
import matplotlib
import numpy as np
from PIL import Image

import utils3d_moge as utils3d

from depth_models.moge3.moge_pt2 import MoGev3_PT2
from utils.visualize.visualize_depth import export_glb

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(input_path: Path) -> list[Path]:
    """Image paths to process from ``input_path``: the file itself when it is
    a single image, otherwise the sorted images of a folder (extension
    filter)."""
    if input_path.is_file():
        return [input_path]
    images = sorted(
        p for p in input_path.iterdir() if p.suffix.lower() in IMAGE_EXTS
    )
    if not images:
        exts = ", ".join(sorted(IMAGE_EXTS))
        raise FileNotFoundError(
            f"No image files ({exts}) found in {input_path}"
        )
    return images


def save_mesh_glb(
    save_path: Union[str, os.PathLike],
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_uvs: np.ndarray,
    texture: np.ndarray,
    vertex_normals: Optional[np.ndarray] = None,
):
    """Write a textured trimesh .glb; mirrors the official MoGe io.save_glb."""
    import trimesh
    import trimesh.visual

    trimesh.Trimesh(
        vertices=vertices,
        vertex_normals=vertex_normals,
        faces=faces,
        visual=trimesh.visual.texture.TextureVisuals(
            uv=vertex_uvs,
            material=trimesh.visual.material.PBRMaterial(
                baseColorTexture=Image.fromarray(texture),
                metallicFactor=0.5,
                roughnessFactor=1.0,
            ),
        ),
        process=False,
    ).export(save_path)


def build_mesh(points, image, normal, depth, mask, threshold):
    """Official MoGe mesh construction; mirrors moge/scripts/infer.py.

    Meshes the model's camera-space point map into a textured grid mesh,
    first dropping pixels on depth discontinuities (3x3 depth range above
    ``threshold`` relative) so no triangle bridges a silhouette. ``image`` is
    uint8 RGB at the grid resolution (used as the texture; vertex colours are
    a side product). Returns (faces, vertices, vertex_colors, vertex_uvs,
    vertex_normals) already in OpenGL export conventions: vertices flipped
    ``* [1, -1, -1]`` and UVs y-flipped to the bottom-left texture origin.
    """
    height, width = points.shape[:2]
    mask_cleaned = mask & ~utils3d.np.depth_map_edge(depth, rtol=threshold)
    if normal is None:
        faces, vertices, vertex_colors, vertex_uvs = utils3d.np.build_mesh_from_map(
            points,
            image.astype(np.float32) / 255,
            utils3d.np.uv_map(height, width),
            mask=mask_cleaned,
            tri=True,
        )
        vertex_normals = None
    else:
        faces, vertices, vertex_colors, vertex_uvs, vertex_normals = utils3d.np.build_mesh_from_map(
            points,
            image.astype(np.float32) / 255,
            utils3d.np.uv_map(height, width),
            normal,
            mask=mask_cleaned,
            tri=True,
        )
    # OpenGL conventions for export: x right, y up, z backward; texture (0,0) left-bottom.
    vertices, vertex_uvs = vertices * [1, -1, -1], vertex_uvs * [1, -1] + [0, 1]
    if vertex_normals is not None:
        vertex_normals = vertex_normals * [1, -1, -1]
    return faces, vertices, vertex_colors, vertex_uvs, vertex_normals


def _save_depth_npz(depth: np.ndarray, intrinsics: np.ndarray,
                    out_base: Path) -> None:
    """Save metric depth + model-estimated intrinsics (3x3, pixel units) as a
    single ``<out_base>.npz`` (keys ``depth`` / ``intrinsics``)."""
    np.savez(str(out_base) + ".npz", depth=depth, intrinsics=intrinsics)


def _save_depth_png(depth: np.ndarray, out_base: Path) -> None:
    """Save a colour-coded depth heatmap (Spectral_r, min-max normalized over
    the valid range)."""
    depth_min = float(np.min(depth))
    depth_max = float(np.max(depth))
    if depth_max > depth_min:
        depth_norm = (depth - depth_min) / (depth_max - depth_min)
    else:
        depth_norm = np.zeros_like(depth)

    depth_img = (depth_norm * 255.0).astype(np.uint8)
    cmap = matplotlib.colormaps.get_cmap("Spectral_r")
    depth_img = (cmap(depth_img)[:, :, :3] * 255).astype(np.uint8)

    Image.fromarray(depth_img).save(str(out_base) + ".png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MoGe v3 monocular metric depth on a folder of images "
        "-> per-image depth npz and/or visualization outputs"
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Input image folder (jpg/png/bmp/webp), or a single image path.",
    )
    parser.add_argument(
        "--frame-idx", type=int, default=None,
        help="0-based index into the sorted image list to process only that "
             "frame; default: process all images.",
    )
    parser.add_argument(
        "--save-depth", action="store_true",
        help="Save per-image <stem>.npz with the metric depth and the "
             "model-estimated intrinsics (3x3, pixel units).",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Save per-image colour heatmap .png and the two mesh .glb files.",
    )
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

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"No such file or directory: {input_path}")
    images = list_images(input_path)
    if args.frame_idx is not None:
        if not 0 <= args.frame_idx < len(images):
            raise SystemExit(
                f"--frame-idx {args.frame_idx} out of range: "
                f"{len(images)} image(s) from {input_path}"
            )
        images = [images[args.frame_idx]]
    print(f"[input] {len(images)} image(s) from {input_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for image_path in images:
        # ---- load input image: cv2 frames are BGR, flip to RGB at the model
        # boundary (repo-wide RGB pixel-space contract — the model decodes any
        # ImageInput via utils.image_io; image_rgb is reused for the GLB) ----
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        print(f"\n[{image_path}]  size={image_bgr.shape[1]}x{image_bgr.shape[0]}")

        t0 = time.perf_counter()
        out = model.infer(image_rgb)
        elapsed = time.perf_counter() - t0
        print(f"inference took {elapsed * 1000:.1f} ms")

        # out: batched fp32 CUDA tensors; depth/mask at the graph's fixed size.
        pred = out["depth"].squeeze(0).cpu().numpy().astype(np.float32)  # (H, W)
        # masked-out pixels come back as +inf -> zero them like the other tools.
        pred[~np.isfinite(pred)] = 0.0

        # ---- model-estimated intrinsics in pixel units. MoGe's intrinsics
        # are expressed in utils3d's normalized-uv space (principal point at
        # 0.5, focal in image-extent units), while export_glb back-projects on
        # an integer pixel grid — rescale to pixels so the geometry matches
        # the model's own unprojected points (uv_map samples pixel centers at
        # (j + 0.5)/W, hence the -0.5 on the principal point). Shared by the
        # npz and the meshes. ----
        K = out["intrinsics"].squeeze(0).cpu().numpy().astype(np.float64)  # (3, 3)
        h, w = pred.shape
        K[0, 0] *= w          # fx in pixels
        K[1, 1] *= h          # fy in pixels
        K[0, 2] = K[0, 2] * w - 0.5
        K[1, 2] = K[1, 2] * h - 0.5

        out_base = out_dir / image_path.stem

        # ---- metric data (--save-depth) ----
        if args.save_depth:
            _save_depth_npz(pred, K, out_base)
            print(f"Saved depth: {out_base}.npz")

        # ---- visualization (--visualize): heatmap + meshes ----
        if args.visualize:
            # colour source for the point cloud, at the depth resolution
            rgb_u8 = image_rgb
            if rgb_u8.shape[:2] != pred.shape:
                rgb_u8 = cv2.resize(rgb_u8, (pred.shape[1], pred.shape[0]))

            _save_depth_png(pred, out_base)
            print(f"Saved heatmap: {out_base}.png")

            # textured surface mesh via export_glb (official-MoGe-style);
            # identity extrinsics: monocular output is in camera space
            glb_path = out_dir / f"{image_path.stem}.glb"
            export_glb(
                depth=pred[None],
                intrinsics=K[None],
                extrinsics=np.eye(4, dtype=np.float64)[None],  # camera space
                images_u8=rgb_u8[None].astype(np.uint8),
                conf=None,
                out_path=str(glb_path),
                mesh=True,
            )
            print(f"Saved glb : {glb_path}")

            # # official MoGe reference mesh: utils3d grid mesh built straight
            # # from the model's camera-space points (no re-unprojection),
            # # cleaned at depth edges with rtol=0.04 -> {stem}_mesh.glb
            # threshold = 0.04  # relative depth-edge tolerance for the quad grid
            # glb_path_2 = out_dir / f"{image_path.stem}_mesh.glb"
            # faces, vertices, vertex_colors, vertex_uvs, vertex_normals = build_mesh(
            #     points=out["points"][0].cpu().numpy(),
            #     image=rgb_u8,
            #     normal=out["normal"][0].cpu().numpy(),
            #     depth=out["depth"][0].cpu().numpy(),
            #     mask=out["mask"][0].cpu().numpy(),
            #     threshold=threshold,
            # )
            # save_mesh_glb(glb_path_2, vertices, faces, vertex_uvs, rgb_u8,
            #               vertex_normals)
            # print(f"Saved glb : {glb_path_2}")

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
