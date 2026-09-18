"""MoGe v3 inference on a folder of images, via the torch.export (PT2) runtime.

Runs the exported dense graph + the eager sparse refiner on each image of an
input folder — or on a single image file, or on one image of a folder
selected by index into the sorted file list. The model returns five outputs at
the graph's fixed size — ``points`` (camera-space metres), ``depth`` (metres),
``mask``, ``intrinsics`` and ``normal`` — of which depth, intrinsics and normal
are used here. ``points`` is deliberately not carried: the postprocess derives
it by back-projecting the depth through the intrinsics, so it holds nothing
that pair does not. ``mask`` needs no handling of its own either — the
postprocess has already applied it, so a masked-out pixel comes back as
``+inf`` depth.

By default nothing is written to disk; per-image outputs are opt-in:
  - --save-depth: <stem>.npz with the metric depth (H, W fp32 metres) and the
    model's estimated intrinsics (3x3, rescaled to pixel units),
  - --save-normal: <stem>_normal.npy (H, W, 3 fp32), the model's normal map,
  - --save-packed: <stem>_octahedral.png, one uint8 RGB carrying the normals
    and the affine log-depth ([Oct_U, Oct_V, logz]; see pack_octahedral_logz /
    unpack_octahedral_logz). The depth channel is a linear rescaling of logz
    over the frame's own affine-z range, intersected with the
    [--depth-min, --depth-max] rails. Code 0 is reserved for invalid, so valid
    data uses codes 1..255 and an out-of-range pixel clips to a bound instead
    of vanishing into the sentinel,
  - --save-scale: <stem>_scale.npz — shift, metric_scale, intrinsics, the
    encoded z_min/z_max and the image size. Implied by --save-packed, since
    the packed image cannot be decoded without it,
  - --visualize: two .png — <stem>_depth.png (Spectral_r heatmap of the
    *metric* depth, min-max normalized over the valid pixels) and
    <stem>_normal.png (the normal map as RGB, xyz -> RGB) — plus the textured
    .glb mesh back-projected with the estimated intrinsics and identity
    extrinsics (camera space; masked-out pixels zeroed). Human-viewable
    artefacts only: the packed image is a data container, so it stays behind
    --save-packed.

The packed image plus its sidecar carry the model's whole output::

    normals, logz, valid = unpack_octahedral_logz(packed, z_min, z_max)
    depth_m = (np.exp(logz) + shift) * metric_scale

and the point map comes back by back-projecting ``depth_m`` through
``intrinsics``. The cost is 8-bit quantization — normals to octahedral
precision (~0.03 on the unit vector) and depth to ~0.6% — which is what
--save-depth and --save-normal are still for.

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
import time
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from PIL import Image

from depth_models.moge3.moge_pt2 import MoGev3_PT2
from utils.visualize.visualize_depth import export_glb

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

#: Bounds for the packed image's affine-z channel, in metres. The encoded range
#: is the intersection of these with the frame's own range (see
#: ``adaptive_logz_range``), so they are a rail rather than the encoding range:
#: a scene that fits inside them is encoded over its own, tighter one. MoGe's
#: affine frame is scale-free, so one pair of bounds covers scenes whose metric
#: scale differs by a lot — the .lz4 format's [MIN_DEPTH, MAX_DEPTH] does not.
DEFAULT_Z_MIN = 0.25
DEFAULT_Z_MAX = 3.0

#: Depth channel codes: 0 is reserved for invalid, valid data uses 1..255.
LOGZ_CODE_MIN = 1
LOGZ_CODE_MAX = 255
LOGZ_LEVELS = LOGZ_CODE_MAX - LOGZ_CODE_MIN  # 254 steps between those codes

#: Frame correction applied to the normals before the octahedral projection.
#: MoGe emits normals in the OpenCV camera frame, where a surface facing the
#: camera has nz = -1, so the bulk of the data sits on the -z axis. The
#: octahedral map is well conditioned at its pole and degenerate at the
#: opposite one — measured 12.5 codes of movement per degree of normal change
#: at the far pole against 1.2 in the best-conditioned band. On the far pole
#: that turns a 0.56-degree step between neighbouring pixels of one flat
#: surface into a 359-code jump: the blocky blotches on every flat wall.
#: Flipping z moves the projection pole onto the data. Its own inverse, and
#: applied on both sides, so pack/unpack still round-trip in MoGe's frame.
#:
#: float32 on purpose: a float64 factor here would promote the whole projection
#: to float64, where the ``+1e-8`` in the L1 norm no longer rounds away and
#: axis normals land one code short of the edge.
POLE_FLIP = np.array([1.0, 1.0, -1.0], dtype=np.float32)


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


def _save_depth_npz(depth: np.ndarray, intrinsics: np.ndarray,
                    out_base: Path) -> None:
    """Save metric depth + model-estimated intrinsics (3x3, pixel units) as a
    single ``<out_base>.npz`` (keys ``depth`` / ``intrinsics``)."""
    np.savez(str(out_base) + ".npz", depth=depth, intrinsics=intrinsics)


def _save_scale_npz(
    shift: float,
    metric_scale: float,
    intrinsics: np.ndarray,
    z_min: float,
    z_max: float,
    shape: tuple[int, int],
    out_base: Path,
) -> None:
    """Save the scalars that decode a packed image back to metres and points.

    The packed image is not self-contained: its depth channel is ``logz``
    rescaled over ``[log z_min, log z_max]``, so both bounds are needed before
    the codes mean anything, and turning ``z`` into metres needs the two
    per-image scalars the postprocess fit. With all of them, everything the
    graph returned is recoverable::

        normals, logz, valid = unpack_octahedral_logz(packed, z_min, z_max)
        depth_m = (np.exp(logz) + shift) * metric_scale
        # points: back-project depth_m through intrinsics (see the module doc)

    ``intrinsics`` is 3x3 in pixels for a ``shape``-sized grid, so the point
    map is reconstructable too — the focal is an independent per-image
    estimate and appears nowhere in the packed image.
    """
    h, w = shape
    np.savez(
        str(out_base) + "_scale.npz",
        shift=np.float64(shift),
        metric_scale=np.float64(metric_scale),
        intrinsics=intrinsics,
        z_min=np.float64(z_min),
        z_max=np.float64(z_max),
        image_h=np.int32(h),
        image_w=np.int32(w),
    )


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
    """Save a packed [Oct_U, Oct_V, logz] image as <out_base>_octahedral.png
    (see pack_octahedral_logz). Undecodable without the sidecar's z_min/z_max."""
    Image.fromarray(packed).save(str(out_base) + "_octahedral.png")


def adaptive_logz_range(
    logz: np.ndarray,
    valid: np.ndarray | None,
    z_min: float,
    z_max: float,
) -> tuple[float, float]:
    """Tighten ``[z_min, z_max]`` to the affine-z range this frame actually uses.

    Quantization error is set by the encoded span, so a scene occupying a
    narrow slice of the caller's bounds should be encoded over that slice —
    measured over the test frames, at 0.29-0.62% against 0.97% for a fixed
    span. The caller's bounds stay a hard rail: the result never escapes them.

    ``(z_min, z_max)`` come back unchanged when the data cannot tighten them,
    which is the no-valid-pixels case and the disjoint case (intersecting there
    would invert the range). Both fall back to the bounds, which clip rather
    than fail.

    Args:
        logz: (H, W) natural log of the affine z.
        valid: (H, W) bool, or None to use every finite pixel.
        z_min: caller's lower affine-z bound, metres.
        z_max: caller's upper affine-z bound, metres.

    Returns:
        ``(z_min, z_max)`` in metres, within the caller's bounds.
    """
    logz = np.asarray(logz, dtype=np.float64)
    usable = np.isfinite(logz)
    if valid is not None:
        usable &= np.asarray(valid, dtype=bool)
    if not usable.any():
        return z_min, z_max

    data_min = float(np.exp(logz[usable].min()))
    data_max = float(np.exp(logz[usable].max()))
    lo, hi = max(z_min, data_min), min(z_max, data_max)
    if not lo < hi:
        # Data clear of the bounds, or a scene flat enough that the span would
        # divide by zero.
        return z_min, z_max
    return lo, hi


def _logz_encode(logz: np.ndarray, z_min: float, z_max: float) -> np.ndarray:
    """logz -> uint8 codes 1..255. Code 0 is left reserved for invalid."""
    lo, hi = np.log(z_min), np.log(z_max)
    norm = np.clip((logz - lo) / (hi - lo), 0.0, 1.0)
    # 1 + norm * 254 floors into 1..255; the cast truncates toward zero.
    return (LOGZ_CODE_MIN + norm * LOGZ_LEVELS).astype(np.uint8)


def _logz_decode(code: np.ndarray, z_min: float, z_max: float) -> np.ndarray:
    """uint8 codes -> logz. Code 0 is invalid and decodes to 0.0."""
    lo, hi = np.log(z_min), np.log(z_max)
    logz = lo + (code.astype(np.float32) - LOGZ_CODE_MIN) / LOGZ_LEVELS * (hi - lo)
    return np.where(code > 0, logz, 0.0).astype(np.float32)


def pack_octahedral_logz(
    normals: np.ndarray,
    logz: np.ndarray,
    valid: np.ndarray | None = None,
    z_min: float = DEFAULT_Z_MIN,
    z_max: float = DEFAULT_Z_MAX,
) -> np.ndarray:
    """Pack a normal map and a log-depth map into one uint8 RGB image.

    The layout is [Oct_U, Oct_V, logz]. U/V are the L1 octahedral projection of
    the unit normal rescaled to [0, 255]; the last channel is the affine
    log-depth the MoGe graph produces, rescaled linearly over
    ``[log z_min, log z_max]``.

    The projection is taken about -z rather than +z, because MoGe's normals are
    in the camera frame and their bulk sits on -z (see :data:`POLE_FLIP`).
    :func:`unpack_octahedral_logz` undoes that, so both sides speak in MoGe's
    frame.

    ``logz`` rather than metres because the graph's output is already
    logarithmic: encoding it directly skips an exp-then-log round trip and, the
    real reason, the ``+0.001 m`` offset the .lz4 codec adds before its log,
    which biases the bottom of its range by up to a quarter of a code.

    The channel decodes back to metres through the sidecar's two scalars::

        depth_m = (exp(logz) + shift) * metric_scale

    Codes run 1..255 and **code 0 is reserved for invalid** — a deliberate
    departure from the .lz4 convention, where 0 means both "invalid" and "at or
    below min_depth". Reserving it matters here because the encoded range is
    adaptive: its lower bound is the scene's own minimum, so under the .lz4
    convention every frame's *nearest* surfaces would decode as masked-out
    (measured at 21-356 px/frame). With 0 reserved, an out-of-range *valid*
    pixel clips to code 1 or 255 and stays valid; only a masked pixel reads
    back invalid.

    Pixels that are not valid — outside the mask, or carrying a non-finite
    logz such as MoGe's ``+inf`` masked-out sentinel — pack to ``(0, 0, 0)``.
    The octahedral pair of such a pixel still decodes to the unit vector
    ``(0, 0, -1)``, so read validity off the depth channel, not the normals.

    Args:
        normals: (H, W, 3) unit normals (any consistent frame).
        logz: (H, W) natural log of the affine z.
        valid: (H, W) bool, or None to treat every non-finite-logz pixel as
            valid.
        z_min: lower end of the encoded range, metres of affine z.
        z_max: upper end of the encoded range, metres of affine z.

    Returns:
        (H, W, 3) uint8 image.
    """
    normals = np.asarray(normals, dtype=np.float32)
    logz = np.asarray(logz, dtype=np.float32)

    # Project about the data rather than about +z (see POLE_FLIP).
    normals = normals * POLE_FLIP

    # A non-finite logz is invalid whether or not a mask is handed in: MoGe's
    # masked-out sentinel arrives as +inf, and left alone it would clip to the
    # far end of the range instead of preserving the invalid 0.
    valid_mask = np.isfinite(logz)
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

    # Scale to [0, 255], then encode the depth channel on its log grid. The
    # placeholder keeps a -inf or NaN away from the log; those pixels are
    # zeroed wholesale below regardless.
    u = ((p[..., 0] + 1.0) * 0.5 * 255.0).astype(np.uint8)
    v = ((p[..., 1] + 1.0) * 0.5 * 255.0).astype(np.uint8)
    d = _logz_encode(np.where(valid_mask, logz, 0.0).astype(np.float64),
                     z_min, z_max)

    packed = np.stack([u, v, d], axis=-1)
    packed[~valid_mask] = 0
    return packed


def unpack_octahedral_logz(
    packed_img: np.ndarray,
    z_min: float = DEFAULT_Z_MIN,
    z_max: float = DEFAULT_Z_MAX,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inverse of :func:`pack_octahedral_logz`.

    Args:
        packed_img: (H, W, 3) uint8 image, [Oct_U, Oct_V, logz].
        z_min: lower bound the image was packed with, metres. Must match the
            pack side (``z_min`` in the sidecar), or the depths come back on
            the wrong scale.
        z_max: upper bound the image was packed with, metres.

    Returns:
        ``(normals (H, W, 3) float32, logz (H, W) float32, valid (H, W) bool)``.
        ``valid`` is the depth channel being non-zero. Test validity on it, not
        on the other two: an invalid pixel's logz is 0.0, which is a real depth
        (z = 1 m), and its normals decode to ``(0, 0, -1)``, a plausible-looking
        back-facing normal.

    Example:
        >>> packed = np.asarray(Image.open("frame_000210_octahedral.png"))
        >>> sidecar = np.load("frame_000210_scale.npz")
        >>> normals, logz, valid = unpack_octahedral_logz(
        ...     packed, float(sidecar["z_min"]), float(sidecar["z_max"])
        ... )
        >>> depth_m = (np.exp(logz) + sidecar["shift"]) * sidecar["metric_scale"]
    """
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

    # Back out of the packer's projection frame, into MoGe's camera frame.
    normals = normals * POLE_FLIP

    # 5. Depth channel back to logz (code 0 -> invalid).
    code = packed_img[..., 2]
    return (normals.astype(np.float32),
            _logz_decode(code, z_min, z_max),
            code > 0)

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
             "the normals and the affine log-depth ([Oct_U, Oct_V, logz]). "
             "Always writes <stem>_scale.npz too — the image cannot be "
             "decoded without it. Not implied by --visualize.",
    )
    parser.add_argument(
        "--save-scale", action="store_true",
        help="Save per-image <stem>_scale.npz: shift, metric_scale, "
             "intrinsics, the encoded z_min/z_max and the image size. Together "
             "with the packed image this rebuilds the model's whole output.",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Save the per-image <stem>_depth.png heatmap, "
             "<stem>_normal.png and <stem>.glb mesh.",
    )
    parser.add_argument(
        "--depth-min", type=float, default=DEFAULT_Z_MIN,
        help=f"Lower rail for the packed image's affine-z range, metres "
             f"(default {DEFAULT_Z_MIN}). The encoded range is this "
             f"intersected with the frame's own range, so it bites only where "
             f"the scene runs below it.",
    )
    parser.add_argument(
        "--depth-max", type=float, default=DEFAULT_Z_MAX,
        help=f"Upper rail for the packed image's affine-z range, metres "
             f"(default {DEFAULT_Z_MAX}); see --depth-min.",
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
        # metres, normal (H, W, 3) and intrinsics (3, 3). The graph also
        # returns a camera-space point map, dropped here: the postprocess
        # derives it by back-projecting the depth through the intrinsics, so
        # depth + intrinsics already carry it. ----
        depth_raw = out["depth"].squeeze(0).cpu().numpy().astype(np.float32)  # (H, W)
        # Both are per-image scalars (shape (1,)), not vectors — the postprocess
        # fits one shift for z and one global scale. They are part of what a
        # reader needs to turn the packed image back into metres.
        shift = float(out["shift"].reshape(-1)[0])
        metric_scale = float(out["metric_scale"].reshape(-1)[0])

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

        # Affine z = exp(logz): the model's scale-free depth, before the
        # per-image shift and metric scale the postprocess applies. Recovered
        # by inverting it (depth = (z + shift) * metric_scale) — the graph's
        # raw logz never leaves moge_pt2, so this log is unavoidable here. The
        # -inf placeholder cannot leak: every consumer gates on `valid`.
        logz = np.where(valid, np.log(pred / metric_scale - shift), -np.inf)

        # The packed channel encodes the frame's own affine-z span, intersected
        # with the caller's rails (see adaptive_logz_range). Computed here so
        # --save-scale alone still records the range the packed image would use.
        z_min, z_max = adaptive_logz_range(logz, valid,
                                           args.depth_min, args.depth_max)
        # ---- model-estimated intrinsics in pixel units. MoGe's intrinsics
        # are expressed in utils3d's normalized-uv space (principal point at
        # 0.5, focal in image-extent units), while export_glb back-projects on
        # an integer pixel grid — rescale to pixels so a back-projection is
        # consistent with the model's own unprojection (uv_map samples pixel
        # centers at (j + 0.5)/W, hence the -0.5 on the principal point).
        # Shared by the npz and the mesh. ----
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

        # ---- packed normals + depth in one RGB image (--save-packed only).
        # A data container rather than a visualization, so --visualize does
        # not imply it — that flag writes only the human-viewable artefacts. ----
        if args.save_packed:
            packed = pack_octahedral_logz(normal, logz, valid, z_min, z_max)
            _save_octahedral_png(packed, out_base)
            print(f"Saved packed: {out_base}_octahedral.png "
                  f"(affine z {z_min:.3f}-{z_max:.3f}m)")

            # The channel saturates rather than fails, so report how much of
            # the scene landed on each end: a range that does not bracket the
            # scene is otherwise invisible in the packed image, which still
            # looks plausible. Adaptive tightening usually makes these zero;
            # they fire when the caller's rails cut into the scene. Both ends
            # stay *valid* (code 1 / 255, not the 0 sentinel), so those pixels
            # survive — their depth is just only good to the bound.
            n_valid = int(valid.sum())
            affine_z = np.exp(logz)
            near = int((valid & (affine_z <= z_min)).sum())
            far = int((valid & (affine_z >= z_max)).sum())
            if near or far:
                print(
                    f"packed depth: {near} px at/below {z_min:.3f}m affine z, "
                    f"{far} px at/above {z_max:.3f}m "
                    f"({(near + far) / max(n_valid, 1):.1%} of valid clipped) "
                    f"— widen --depth-min/--depth-max to bracket the scene"
                )

        # ---- sidecar (--save-scale). Implied by --save-packed, because the
        # packed image is undecodable without its z_min/z_max and its depth
        # channel is metric-less without the two scalars. ----
        if args.save_scale or args.save_packed:
            _save_scale_npz(shift, metric_scale, K, z_min, z_max, pred.shape,
                            out_base)
            print(f"Saved scale : {out_base}_scale.npz")

        # ---- visualization (--visualize): two PNGs + the mesh ----
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

        if valid.any():
            print(
                f"pred depth: {pred[valid].mean():.2f}+/-{pred[valid].std():.2f}m  "
                f"valid={valid.mean():.1%}"
            )
        else:
            print("Warning: no valid predicted depth pixels.")


if __name__ == "__main__":
    main()
