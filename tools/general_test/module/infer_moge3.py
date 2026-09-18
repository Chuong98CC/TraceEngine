"""MoGe v3 inference on a folder of images, via the torch.export (PT2) runtime.

Runs the exported dense graph + the eager sparse refiner on each image of an
input folder — or on a single image file, or on one image of a folder
selected by index into the sorted file list. The model returns five outputs at
the graph's fixed size — ``points`` (camera-space metres), ``depth`` (metres),
``mask``, ``intrinsics`` and ``normal``. ``mask`` needs no handling of its
own: the postprocess has already applied it, so a masked-out pixel comes back
as ``+inf`` depth. ``points`` is only consumed by the commented-out
reference-mesh block.

By default nothing is written to disk; per-image outputs are opt-in:
  - --save-depth: <stem>.npz with the metric depth (H, W fp32 metres) and the
    model's estimated intrinsics (3x3, rescaled to pixel units),
  - --save-normal: <stem>_normal.npy (H, W, 3 fp32), the model's normal map,
  - --save-packed: <stem>_octahedral.png, one uint8 RGB image carrying the
    normals and the depth ([Oct_U, Oct_V, log depth]; see
    pack_octahedral_depth / unpack_octahedral_depth). The depth channel uses
    the repo's log-depth codec over [--depth-min, --depth-max] so it decodes
    straight back to metres; invalid pixels pack to (0, 0, 0),
  - --visualize: three .png — <stem>_depth.png (Spectral_r heatmap, min-max
    normalized over the valid pixels), <stem>_normal.png (the normal map as
    RGB, xyz -> RGB) and the packed <stem>_octahedral.png above — plus the
    textured .glb mesh back-projected with the estimated intrinsics and
    identity extrinsics (camera space; masked-out pixels zeroed).

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
    --save-depth --save-normal --save-packed --visualize \
    --out_dir ./output/moge3

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
from utils.depth_utils import MAX_DEPTH, MIN_DEPTH, LogDepthToUint8Transform
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


def _save_depth_png(depth: np.ndarray, valid: np.ndarray, out_base: Path) -> None:
    """Save a colour-coded depth heatmap (Spectral_r) as <out_base>_depth.png.

    Min-max normalized over the *valid* pixels only: the invalid ones are
    zeroed, so including them would pin the minimum at 0 and squash the whole
    scene into the top of the colormap.
    """
    depth_img = np.zeros(depth.shape, dtype=np.uint8)
    if valid.any():
        depth_min = float(depth[valid].min())
        depth_max = float(depth[valid].max())
        if depth_max > depth_min:
            depth_img = (
                np.clip((depth - depth_min) / (depth_max - depth_min), 0.0, 1.0)
                * 255.0
            ).astype(np.uint8)

    cmap = matplotlib.colormaps.get_cmap("Spectral_r")
    Image.fromarray((cmap(depth_img)[:, :, :3] * 255).astype(np.uint8)).save(
        str(out_base) + "_depth.png"
    )


def _save_normal_png(normal: np.ndarray, valid: np.ndarray, out_base: Path) -> None:
    """Save the model's normal map as <out_base>_normal.png.

    Channels are xyz -> RGB with the usual [-1, 1] -> [0, 255] rescale, so the
    result is a standard normal map. Its R/G pair carries the same signal as
    the packed image's octahedral pair, which makes the two directly
    comparable. Invalid pixels are black.
    """
    rgb = np.clip((np.asarray(normal, dtype=np.float32) + 1.0) * 0.5 * 255.0,
                  0.0, 255.0).astype(np.uint8)
    rgb[~valid] = 0
    Image.fromarray(rgb).save(str(out_base) + "_normal.png")


def _save_octahedral_png(packed: np.ndarray, out_base: Path) -> None:
    """Save a packed [Oct_U, Oct_V, log depth] image as
    <out_base>_octahedral.png (see pack_octahedral_depth)."""
    Image.fromarray(packed).save(str(out_base) + "_octahedral.png")


def pack_octahedral_depth(
    normals: np.ndarray,
    depth_m: np.ndarray,
    valid: np.ndarray | None = None,
    depth_transform: LogDepthToUint8Transform | None = None,
) -> np.ndarray:
    """Pack a normal map and a metric depth map into one uint8 RGB image.

    The layout is [Oct_U, Oct_V, depth], so a single 8-bit PNG carries both
    signals. U/V are the L1 octahedral projection of the unit normal rescaled
    to [0, 255]; depth is the repo's log-depth codec
    (:class:`LogDepthToUint8Transform`) over ``[min_depth_m, max_depth_m]``, so
    the depth channel decodes straight back to metres. A linear normalization
    would instead thin out the far range, where most of the scene sits.

    Pixels that are not valid — outside the mask, or carrying a non-finite /
    non-positive depth such as MoGe's ``+inf`` masked-out sentinel — pack to
    ``(0, 0, 0)``. On the way back the *depth channel* is the validity marker:
    the codec decodes 0 to 0.0 m, the same 0-is-invalid convention as every
    ``.lz4`` depth file, whereas the octahedral pair still decodes to the unit
    vector ``(0, 0, -1)`` — a plausible-looking back-facing normal. Read the
    depth channel, not the normals, to find the invalid pixels.

    Two *valid* depths are indistinguishable from that invalid 0, exactly as in
    the .lz4 storage format: one at or below ``min_depth_m``, and one at or
    above ``max_depth_m``, which saturates to the top code and decodes back as
    exactly ``max_depth_m``. Pick the range to bracket the scene — a monocular
    metric model's background easily runs past a sensor-oriented default.

    Args:
        normals: (H, W, 3) unit normals (any consistent frame).
        depth_m: (H, W) metric depth. Metres (float) or uint16 millimetres:
            the codec auto-detects the latter, and the array is deliberately
            not cast here so that detection still works.
        valid: (H, W) bool, or None to treat every pixel as valid.
        depth_transform: :class:`LogDepthToUint8Transform` the depth channel is
            encoded with, or None for the default [MIN_DEPTH, MAX_DEPTH] range.

    Returns:
        (H, W, 3) uint8 image.
    """
    if depth_transform is None:
        depth_transform = LogDepthToUint8Transform()
    normals = np.asarray(normals, dtype=np.float32)
    depth_m = np.asarray(depth_m)

    # Non-finite / non-positive depth is invalid whether or not a mask is
    # handed in: the codec counts +inf as a valid measurement and would clip
    # it to the far end of the range instead of preserving the invalid 0.
    valid_mask = np.isfinite(depth_m) & (depth_m > 0.0)
    if valid is not None:
        valid_mask &= np.asarray(valid, dtype=bool)

    # Project to the L1 octahedron.
    l1_norm = np.sum(np.abs(normals), axis=-1, keepdims=True) + 1e-8
    p = normals[..., :2] / l1_norm

    # Unfold the back-facing hemisphere onto the diamond's outer edges. The
    # sign must be +/-1, never np.sign: sign(0) is 0, which would zero the
    # coordinate the fold just derived from the other one.
    back = normals[..., 2] < 0
    p[back] = (1.0 - np.abs(p[back, ::-1])) * np.where(p[back] >= 0.0, 1.0, -1.0)

    # Scale to [0, 255], then encode the depth channel on its log grid.
    u = ((p[..., 0] + 1.0) * 0.5 * 255.0).astype(np.uint8)
    v = ((p[..., 1] + 1.0) * 0.5 * 255.0).astype(np.uint8)
    # The integer fill keeps an integer depth_m integer, so the codec still
    # sees the uint16-mm arrays it is documented to auto-detect.
    d = np.asarray(depth_transform.encode(np.where(valid_mask, depth_m, 0)),
                   dtype=np.uint8)

    packed = np.stack([u, v, d], axis=-1)
    packed[~valid_mask] = 0
    return packed


def unpack_octahedral_depth(
    packed_img: np.ndarray,
    depth_transform: LogDepthToUint8Transform | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`pack_octahedral_depth`.

    Args:
        packed_img: (H, W, 3) uint8 image, [Oct_U, Oct_V, log depth].
        depth_transform: the codec the image was packed with, or None for the
            default [MIN_DEPTH, MAX_DEPTH] range. Must match the pack side, or
            the depths come back on the wrong scale.

    Returns:
        ``(normals (H, W, 3) float32 unit vectors, depth_m (H, W) float32
        metres)``. A zero depth channel marks an invalid pixel and comes back
        as 0.0 m; that pixel's normals decode to ``(0, 0, -1)``, so test
        validity on the depth channel rather than on the normals.

    Example:
        >>> packed = np.asarray(Image.open("frame_210_octahedral.png"))
        >>> normals, depth_m = unpack_octahedral_depth(packed)
    """
    if depth_transform is None:
        depth_transform = LogDepthToUint8Transform()
    packed_img = np.asarray(packed_img, dtype=np.uint8)

    # 1. Unscale back to [-1, 1] for the octahedral coordinates.
    u = packed_img[..., 0].astype(np.float32) / 255.0 * 2.0 - 1.0
    v = packed_img[..., 1].astype(np.float32) / 255.0 * 2.0 - 1.0

    # 2. Reconstruct z on the octahedron.
    z = 1.0 - (np.abs(u) + np.abs(v))

    # 3. Unfold back-facing hemisphere where z < 0.
    below = z < 0.0

    # Avoid zero-sign issues (np.sign(0) returns 0; keep sign as +1 or -1)
    sign_u = np.where(u >= 0.0, 1.0, -1.0)
    sign_v = np.where(v >= 0.0, 1.0, -1.0)

    x = u.copy()
    y = v.copy()

    x[below] = (1.0 - np.abs(v[below])) * sign_u[below]
    y[below] = (1.0 - np.abs(u[below])) * sign_v[below]

    # 4. Stack and normalize to unit length.
    normals = np.stack([x, y, z], axis=-1)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals / np.maximum(norm, 1e-8)

    # 5. Depth channel: the log codec straight back to metres (0 -> invalid).
    depth_m = depth_transform.decode(packed_img[..., 2])

    return normals.astype(np.float32), depth_m

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
        "--save-normal", action="store_true",
        help="Save per-image <stem>_normal.npy (H, W, 3 float32), the model's "
             "normal map.",
    )
    parser.add_argument(
        "--save-packed", action="store_true",
        help="Save per-image <stem>_octahedral.png: one uint8 RGB carrying "
             "the normals and the depth ([Oct_U, Oct_V, log depth]). Written "
             "by --visualize too.",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Save the per-image <stem>_depth.png heatmap, "
             "<stem>_normal.png, <stem>_octahedral.png and <stem>.glb mesh.",
    )
    parser.add_argument(
        "--depth-min", type=float, default=MIN_DEPTH,
        help=f"Lower end of the log-depth range used by the packed image's "
             f"depth channel (metres, default {MIN_DEPTH}).",
    )
    parser.add_argument(
        "--depth-max", type=float, default=MAX_DEPTH,
        help=f"Upper end of the log-depth range used by the packed image's "
             f"depth channel (metres, default {MAX_DEPTH}).",
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
    parser.add_argument("--refine_steps", type=int, default=1, help="Sparse refinement steps.")
    parser.add_argument("--out_dir", type=str, default="./output/moge3")
    args = parser.parse_args()
    if args.depth_min <= 0.0 or args.depth_min >= args.depth_max:
        parser.error(
            f"--depth-min ({args.depth_min}) must be > 0 and < --depth-max "
            f"({args.depth_max}): the packed image's depth channel is "
            f"log-encoded over that range."
        )
    return args


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

        # ---- model outputs, all at the graph's fixed size: depth (H, W)
        # metres, normal (H, W, 3) and intrinsics (3, 3), plus the camera-space
        # point map (H, W, 3) that only the commented-out reference-mesh block
        # below consumes. ----
        depth_raw = out["depth"].squeeze(0).cpu().numpy().astype(np.float32)  # (H, W)
        # The postprocess already applies the model's mask, writing +inf over
        # every pixel it rejected, so the depth's finiteness *is* that mask —
        # no second validity test to keep in sync. Reading it off the depth
        # also keeps a stray non-finite depth out of the consumers below, which
        # is what the mask alone (it survives the +z test on an overflowing
        # logz) would not. One source of truth for the three PNGs, the packer,
        # the mesh and the summary.
        valid = np.isfinite(depth_raw)  # (H, W)
        # masked-out pixels come back as +inf -> zero them like the other tools.
        pred = np.where(valid, depth_raw, 0.0).astype(np.float32)
        normal = out["normal"].squeeze(0).cpu().numpy().astype(np.float32)  # (H, W, 3)

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

        # ---- normal data (--save-normal). No mask file: the model's mask is
        # already applied to the depth, so it is recoverable as ``depth > 0``
        # from the npz --save-depth writes, and from the packed image's depth
        # channel. ----
        if args.save_normal:
            np.save(str(out_base) + "_normal.npy", normal)
            print(f"Saved normal: {out_base}_normal.npy")

        # ---- packed normals + depth in one RGB image. --visualize lists the
        # same file among its three PNGs, hence the either/or. ----
        if args.save_packed or args.visualize:
            packed = pack_octahedral_depth(
                normal, pred, valid,
                LogDepthToUint8Transform(min_depth_m=args.depth_min,
                                         max_depth_m=args.depth_max),
            )
            _save_octahedral_png(packed, out_base)
            print(f"Saved packed: {out_base}_octahedral.png")

            # The codec saturates rather than fails, so report how much of the
            # scene landed on each end of the range: a range that does not
            # bracket the scene is otherwise invisible in the packed image,
            # which still looks plausible. Both ends decode to a single value
            # (min -> invalid 0.0 m, max -> exactly max_depth_m).
            n_valid = int(valid.sum())
            near = int((valid & (pred <= args.depth_min)).sum())
            far = int((valid & (pred >= args.depth_max)).sum())
            if near or far:
                print(
                    f"packed depth: {near} px at/below {args.depth_min}m, "
                    f"{far} px at/above {args.depth_max}m "
                    f"({far / max(n_valid, 1):.1%} of valid clipped) — widen "
                    f"--depth-min/--depth-max to bracket the scene"
                )

        # ---- visualization (--visualize): three PNGs + the mesh ----
        if args.visualize:
            # colour source for the point cloud, at the depth resolution
            rgb_u8 = image_rgb
            if rgb_u8.shape[:2] != pred.shape:
                rgb_u8 = cv2.resize(rgb_u8, (pred.shape[1], pred.shape[0]))

            _save_depth_png(pred, valid, out_base)
            _save_normal_png(normal, valid, out_base)
            print(f"Saved heatmap: {out_base}_depth.png")
            print(f"Saved normal : {out_base}_normal.png")

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

        if valid.any():
            print(
                f"pred depth: {pred[valid].mean():.2f}+/-{pred[valid].std():.2f}m  "
                f"valid={valid.mean():.1%}"
            )
        else:
            print("Warning: no valid predicted depth pixels.")


if __name__ == "__main__":
    main()
