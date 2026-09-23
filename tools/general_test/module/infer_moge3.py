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
    and the affine depth ([Oct_U, Oct_V, depth]; see utils.normal_depth_pack).
    The depth channel encodes the frame's own affine-z range, intersected with
    the [--depth-min, --depth-max] rails. Code 0 is reserved for invalid, so
    valid data uses codes 1..255 and an out-of-range pixel clips to a bound
    instead of vanishing into the sentinel,
  - --save-blend-Dab: <stem>_blend_Dab.png, the depth plus the frame's
    high-frequency L, ridden in Lab's L channel, with the input frame's own
    chroma in a and b. Normals are not carried. It still reads as a picture,
    which is the point: the detail term is what puts the surface markings back
    on an otherwise bare depth ramp, and it is what the depth costs — see
    utils.normal_depth_pack and DAB_DETAIL_ALPHA. The ramp runs between
    DEFAULT_L_DARK and DEFAULT_L_BRIGHT, with DEFAULT_FAR_IS_BRIGHT choosing
    which end the distance gets -- by default the far surfaces are the bright
    ones and the near ones the dark. A reader needs the resolved ends, which is
    why they go in the sidecar,
  - --save-Alb-norm: <stem>_alb_normal.png — a Retinex albedo of the frame's
    Lab L in R, with the normal's nx and ny in G and B. The only carrier with
    no depth in it, and the only one that is lossy rather than merely
    quantized: two channels cannot say which side of the fold a normal is on,
    so the reader takes it to be on the pole's side, and the small share of
    pixels pointing the other way come back mirrored (measured: 0.184% at more
    than 10 degrees, worst 49). See utils.normal_depth_pack,
  - --save-scale: <stem>_scale.npz — shift, metric_scale, intrinsics, the pole,
    the encoded z_min/z_max and the image size, plus the albedo percentiles
    when --save-Alb-norm asked for them and the Dab ramp's ends when
    --save-blend-Dab did. Implied by --save-packed, since the packed image
    cannot be decoded without it, and by --save-blend-Dab for the same reason —
    those two share one sidecar, because they carry the same depth code over
    the same range,
  - --visualize: two .png — <stem>_depth.png (Spectral_r heatmap of the
    *metric* depth, min-max normalized over the valid pixels) and
    <stem>_normal.png (the normal map as RGB, xyz -> RGB) — plus the textured
    .glb mesh back-projected with the estimated intrinsics and identity
    extrinsics (camera space; masked-out pixels zeroed). Human-viewable
    artefacts only: the packed image is a data container, so it stays behind
    --save-packed.

The packed image plus its sidecar carry the model's whole output::

    sidecar = NormalDepthPack.load_scale(scale_path)
    normals, depth_z, valid = pack.decode_with_scale(packed, sidecar)
    depth_m = DepthScale.from_dict(sidecar).to_metric(depth_z)

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
from utils.normal_depth_pack import (
    DEFAULT_Z_MAX,
    DEFAULT_Z_MIN,
    MOGE_POLE,
    DepthScale,
    NormalDepthPack,
    encode_alb_normal,
    encode_dab,
    encode_fused,
    encode_l_normal,
)
from utils.visualize.visualize_depth import export_glb

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ---------------------------------------------------------------- tuning
# Everything worth experimenting with, in one block so it can be tuned without
# hunting through the file. Each is the default of the call that uses it and is
# passed explicitly there, so editing a value here changes the output; the
# shared codec keeps its own defaults for callers that are not this tool.

# --save-blend-Dab: how much of the frame's high-frequency L to lay back over
# the depth, and the filter that extracts it. The term is what puts surface
# markings on an otherwise bare depth ramp, and it is also what the depth
# costs -- the two share one channel. Measured on a real frame, alpha=0.1 with
# this filter takes the depth from a guarantee of 1.0% to a p99 of ~2.9%.
# sigma_color is the main dial: it sets how far a strong edge is allowed into
# the residual (see the codec's DAB_DETAIL_* for what each does).
DEFAULT_ALPHA = 0.0
DEFAULT_D = 15
DEFAULT_SIGMA_COLOR = 0.2
DEFAULT_SIGMA_SPACE = 9

#: The gate the depth ramp lives inside, and which end of it is bright.
#: Kept as limits rather than as ends so the direction can be flipped without
#: touching them -- and note that unlike everything above, these are a property
#: of the *encoding*: a reader cannot recover the depth without them, which is
#: why the tool writes the resolved ends into the sidecar.
#:
#: Keeping the ramp off 0 and 100 is the point. At the full span the nearest
#: surface lands near-black and the farthest near-white, so the image reads as
#: an exposure problem rather than as a depth map. Narrowing the span further
#: costs depth accuracy, because each code becomes a smaller step in L for the
#: 8-bit round trip to resolve: measured over the ramp, 0-100 and 10-90 both
#: give a 0.98% max error, 20-85 gives 1.97%, 40-65 gives 3.93%.
DEFAULT_L_DARK = 0.1
DEFAULT_L_BRIGHT = 99.9

#: Which end of the ramp the far surfaces get. True puts the distance *behind*
#: the subject at the bright end, which is how a photograph of a lit room
#: usually falls; False puts the near surfaces there instead.
DEFAULT_FAR_IS_BRIGHT = True


#: Floor for a frame-derived ramp end. The encoder needs both ends above zero
#: -- an L* of 0 is pure black, which is what an invalid pixel is written as --
#: and the sRGB round trip reaches black at around L* 0.17, so the floor sits
#: clear of that rather than on it. sRGB 4 at this value.
DEFAULT_L_FLOOR = 1.0

#: Smallest luminance span worth scaling to. Below it the two ends would land
#: on the same value, which the encoder rejects, so one flat frame would abort
#: the run; the gate stands in instead.
DEFAULT_MIN_L_SPAN = 2.0


def _dab_ramp(dark: float, bright: float, far_is_bright: bool) -> tuple[float, float]:
    """Resolve the gate and the direction into the L* at each end of the ramp.

    Returns ``(l_far, l_near)``, which is what both the encoder and a reader
    need -- the gate and the flag are how this tool spells them, not what the
    encoding stores.
    """
    return (bright, dark) if far_is_bright else (dark, bright)

# --save-fused reuses DEFAULT_ALPHA above: the detail term is converted into
# depth codes so one setting means the same texture strength in both carriers.

# --save-L-norm: how far the normal's nx, ny are stretched into Lab's a and b.
# Like the ramp ends below, this is a property of the *encoding* -- a reader
# cannot recover the pair without it, so it goes in the sidecar. 56 is the
# largest that fits the gamut's narrower channel at mid luminance; past it the
# mid-tones clip too and the whole frame loses the normals rather than just its
# bright and dark ends. See the codec's LN_SCALE for why this carrier is the
# weakest of the three at carrying them.
DEFAULT_L_NORM_SCALE = 56.0

# --save-Alb-norm: the Retinex split's illumination scale and range, plus a
# pre-filter on the log image. The pre-filter matters because the split's
# residual is stretched hard -- its p1-p99 span is only ~0.37, so the
# normalization multiplies every step in it by ~680x, and the pixel-level
# noise already in L lands on flat surfaces as visible speckle. Measured over
# six cameras it takes flat-region noise from 2.15x to 1.22x while the
# retained fine detail only moves from 1.58x to 1.45x. A larger denoise
# sigma_r (0.25) starts flattening real texture instead, and raising sigma_r
# past its knee near 1.0 absorbs real albedo into the illumination.
DEFAULT_SIGMA_S = 30
DEFAULT_SIGMA_R = 1.0
DEFAULT_DENOISE_SIGMA_S = 3
DEFAULT_DENOISE_SIGMA_R = 0.15


def _scaled_dab_ramp(l_star, far_is_bright: bool,
                     floor: float = DEFAULT_L_FLOOR,
                     min_span: float = DEFAULT_MIN_L_SPAN) -> tuple[float, float]:
    """The Dab ramp's ends, taken from the frame's own luminance.

    Where :func:`_dab_ramp` is handed fixed limits, this reads them off the
    image: the brightest pixel becomes one end of the depth ramp and the
    darkest the other, so a low-contrast frame gets a low-contrast ramp instead
    of being stretched over the gate.

    The floor is not part of that idea, it is the sentinel's: an L* of 0 is
    pure black, which is exactly what an invalid pixel is written as, so a
    frame with a black pixel would make the two indistinguishable. A frame
    with no span at all has nothing to scale to and falls back to the gate.
    """
    low = max(float(np.min(l_star)), floor)
    high = min(float(np.max(l_star)), 100.0)
    if high - low < min_span:
        return _dab_ramp(DEFAULT_L_DARK, DEFAULT_L_BRIGHT, far_is_bright)
    return _dab_ramp(low, high, far_is_bright)


def extract_albedo_retinex(l_channel, sigma_s=DEFAULT_SIGMA_S,
                           sigma_r=DEFAULT_SIGMA_R,
                           denoise_sigma_s=DEFAULT_DENOISE_SIGMA_S,
                           denoise_sigma_r=DEFAULT_DENOISE_SIGMA_R):
    """
    Separates L* into an illumination-invariant albedo channel and an illumination map.

    Parameters:
        l_channel: (H, W) uint8 or float32 array in [0, 255]
        sigma_s: Spatial standard deviation (smooths over large lighting gradients)
        sigma_r: Range/intensity standard deviation in log space (preserves texture edges)
        denoise_sigma_s: Spatial extent of the pre-filter. 0 disables it.
        denoise_sigma_r: Range sigma of the pre-filter, same units as sigma_r.

    Returns:
        albedo: (H, W) uint8 in [0, 255], free of slow-moving shadows/highlights
        illumination: (H, W) uint8 in [0, 255], smooth ambient light map
        p_low, p_high: the 1st and 99th percentiles the albedo was scaled
            between. Returned because the scaling is per-image and otherwise
            uninvertible: without them ``albedo`` cannot be turned back into
            the reflectance it came from, and two frames' albedos are not
            comparable. --save-Alb-norm records them in the sidecar.

    Both sigmas are read against an image spanning about 5 log units, so they
    are easy to set an order of magnitude too small -- 0.25 is 5% of that span,
    which leaves the bilateral treating almost every texture edge as one to
    preserve, and the illumination hugging the input. The residual is then
    almost pure fine detail and the stretch below blows it up. Raise sigma_r
    until the result stops changing rather than until it looks smooth: the
    curve has a knee near 1.0, and past it the filter starts absorbing real
    albedo into the illumination.
    """
    # 1. Normalize L to (0, 1] and move to log domain
    l_float = l_channel.astype(np.float32) / 255.0
    l_log = np.log(np.maximum(l_float, 1e-4))

    # 2. Pre-filter, in log space so the noise is additive and one range sigma
    #    fits dark and bright regions alike. This is a denoise, not the
    #    illumination estimate -- it is deliberately far smaller than step 3.
    if denoise_sigma_s > 0:
        l_log = cv2.bilateralFilter(
            l_log,
            d=-1,
            sigmaColor=denoise_sigma_r,
            sigmaSpace=denoise_sigma_s
        )

    # 3. Estimate illumination via Bilateral Filter in log domain
    # Large d or spatial sigma captures wide shadow gradients
    illum_log = cv2.bilateralFilter(
        l_log,
        d=-1,
        sigmaColor=sigma_r,
        sigmaSpace=sigma_s
    )

    # 4. Extract reflectance (albedo) in log domain
    albedo_log = l_log - illum_log

    # 5. Map back to linear domain
    albedo = np.exp(albedo_log)
    illum = np.exp(illum_log)

    # 6. Contrast-normalize albedo to [0, 255]
    # Percentile clipping prevents outlier artifacts from compressing dynamic range
    p_low, p_high = np.percentile(albedo, (1.0, 99.0))
    albedo_scaled = np.clip((albedo - p_low) / (p_high - p_low + 1e-6), 0.0, 1.0)
    albedo_u8 = (albedo_scaled * 255.0).astype(np.uint8)

    illum_scaled = np.clip(illum, 0.0, 1.0)
    illum_u8 = (illum_scaled * 255.0).astype(np.uint8)

    return albedo_u8, illum_u8, p_low, p_high

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
    pack: NormalDepthPack,
    scale: DepthScale,
    intrinsics: np.ndarray,
    z_min: float,
    z_max: float,
    shape: tuple[int, int],
    out_base: Path,
    **extra: object,
) -> None:
    """Save the sidecar that decodes a packed image back to metres and points.

    Everything the codec does not own, plus everything it does: the two
    per-image scalars the postprocess fit, the pole, the range the depth
    channel was encoded over and the image size (see
    :meth:`NormalDepthPack.scale_dict`). ``intrinsics`` is 3x3 in pixels for a
    ``shape``-sized grid, so the point map is reconstructable too — the focal
    is an independent per-image estimate and appears nowhere in the packed
    image.

    ``**extra`` carries whatever a particular carrier needs and the codec does
    not own — currently the albedo percentiles, which only --save-Alb-norm
    knows and only its reader needs.
    """
    np.savez(
        str(out_base) + "_scale.npz",
        **pack.scale_dict(scale, z_min, z_max, shape, intrinsics=intrinsics),
        **extra,
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
    """Save a packed [Oct_U, Oct_V, depth] image as <out_base>_octahedral.png
    (see utils.normal_depth_pack). Undecodable without its sidecar."""
    Image.fromarray(packed).save(str(out_base) + "_octahedral.png")


def _save_scale_dab_png(blended: np.ndarray, out_base: Path) -> None:
    """Save a scale-Dab blend as <out_base>_scale_Dab.png.

    The same format as <out_base>_blend_Dab.png -- only the ramp's ends differ,
    and those live in the sidecar rather than in the image.
    """
    Image.fromarray(blended).save(str(out_base) + "_scale_Dab.png")


def _save_blend_dab_png(blended: np.ndarray, out_base: Path) -> None:
    """Save a Lab(depth, a, b) blend as <out_base>_blend_Dab.png.

    Still a data container -- decode_dab recovers the depth with the same
    sidecar -- but one that reads as a picture, so unlike the octahedral image
    it is not nonsense to look at.
    """
    Image.fromarray(blended).save(str(out_base) + "_blend_Dab.png")


def _save_fused_png(image: np.ndarray, out_base: Path) -> None:
    """Save a [Oct_U, Oct_V, depth + detail] image as <out_base>_fused.png.

    Laid out exactly like --save-packed's image, so the same reader decodes it
    and the surface markings are the only difference.
    """
    Image.fromarray(image).save(str(out_base) + "_fused.png")


def _save_l_normal_png(image: np.ndarray, out_base: Path) -> None:
    """Save an [L, nx, ny] carrier as <out_base>_L_normal.png.

    The frame's own luminance with its colour replaced by the surface normals,
    so it reads as a photograph shot under strange light.
    """
    Image.fromarray(image).save(str(out_base) + "_L_normal.png")


def _save_alb_normal_png(image: np.ndarray, out_base: Path) -> None:
    """Save an [albedo, nx, ny] carrier as <out_base>_alb_normal.png.

    The albedo is the frame's Lab L through a Retinex split, so unlike the
    normal channels it is not a geometric quantity -- it is what the surface
    would look like unlit.
    """
    Image.fromarray(image).save(str(out_base) + "_alb_normal.png")


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
             "the normals and the affine depth ([Oct_U, Oct_V, depth]). "
             "Always writes <stem>_scale.npz too — the image cannot be "
             "decoded without it. Not implied by --visualize.",
    )
    parser.add_argument(
        "--save-scale", action="store_true",
        help="Save per-image <stem>_scale.npz: shift, metric_scale, "
             "intrinsics, the pole, the encoded z_min/z_max and the image "
             "size. Together with the packed image this rebuilds the model's "
             "whole output.",
    )
    parser.add_argument(
        # An explicit dest: argparse would otherwise spell the attribute
        # save_blend_Dab, carrying the flag's capital into the code.
        "--save-blend-Dab", action="store_true", dest="save_blend_dab",
        help="Save per-image <stem>_blend_Dab.png: the input frame's own chroma "
             "in Lab's a/b, with the depth code plus the frame's high-frequency "
             "L in the L channel. Carries the depth only (no normals) and still "
             "reads as a picture. The detail term is what shows surface "
             "markings, and is also what the depth costs: ~1% of the depth for "
             "a bare ramp against a p99 of ~3% with it. Implies "
             "<stem>_scale.npz, same as --save-packed; the two share one "
             "sidecar.",
    )
    parser.add_argument(
        # Explicit dest, same reason as --save-blend-Dab above: argparse would
        # spell the attribute save_Alb_norm.
        "--save-Alb-norm", action="store_true", dest="save_alb_norm",
        help="Save per-image <stem>_alb_normal.png: an albedo map (Retinex on "
             "the frame's Lab L) in R, the normal's nx and ny in G and B. The "
             "hemisphere for the missing nz comes from the declared pole, so "
             "pixels pointing the other way come back mirrored. Implies "
             "<stem>_scale.npz, which also gains the albedo's percentiles.",
    )
    parser.add_argument(
        # Explicit dest, same reason as --save-blend-Dab above.
        "--save-L-norm", action="store_true", dest="save_l_norm",
        help="Save per-image <stem>_L_normal.png: the frame's own Lab L with "
             "the normal's nx and ny in a and b. Reads as a photograph under "
             "strange light. Carries no depth and no albedo, and carries the "
             "normals poorly -- the sRGB gamut limits a and b by the pixel's "
             "own luminance, so they survive at mid grey and not at the ends. "
             "Implies <stem>_scale.npz, which records the scale.",
    )
    parser.add_argument(
        "--save-fused", action="store_true",
        help="Save per-image <stem>_fused.png: the packed image "
        "([Oct_U, Oct_V, depth]) with the frame's high-frequency L laid over "
        "the depth channel, so the surfaces carry their markings instead of "
        "reading as a flat ramp. Decoded by the same reader as --save-packed; "
        "the detail is what the depth costs.",
    )
    parser.add_argument(
        # Explicit dest, same reason as --save-blend-Dab above.
        "--save-scale-Dab", action="store_true", dest="save_scale_dab",
        help="Save per-image <stem>_scale_Dab.png: as --save-blend-Dab, but "
             "the ramp's ends come from the frame's own luminance extremes "
             "rather than the fixed gate, so a low-contrast frame gets a "
             "low-contrast ramp instead of being stretched. Implies "
             "<stem>_scale.npz, which records the ends a reader needs.",
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
    if args.save_blend_dab and args.save_scale_dab:
        parser.error(
            "--save-blend-Dab and --save-scale-Dab share one <stem>_scale.npz, "
            "and it records a single ramp -- theirs differ, so a reader could "
            "not tell which image the ends belonged to. Run them separately."
        )
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

        # Affine z: the model's scale-free depth, before the per-image shift
        # and metric scale the postprocess applies. Recovered by inverting that
        # (depth = (z + shift) * metric_scale) — the graph's raw z never leaves
        # moge_pt2. The zero placeholder cannot leak: every consumer gates on
        # `valid`, and the codec masks non-positive depths of its own accord.
        depth_z = np.where(valid, pred / metric_scale - shift, 0.0).astype(np.float32)

        # The codec, with MoGe's normal convention declared: the model emits
        # normals in the OpenCV camera frame, so camera-facing surfaces point
        # along -z. Declaring that puts the projection pole on the data, which
        # is what keeps flat surfaces from coming out as hard-edged colour
        # blocks.
        pack = NormalDepthPack(pole=MOGE_POLE, z_min=args.depth_min,
                               z_max=args.depth_max)

        # The packed channel encodes the frame's own affine-z span, intersected
        # with the caller's rails. Resolved here so --save-scale alone still
        # records the range the packed image would use.
        z_min, z_max = pack.resolve_range(depth_z, valid)
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

        # The input at the model's output resolution. Depth is 640x480 by
        # construction, so the frame's chroma has to be resampled onto that
        # grid to blend against it — and the mesh wants the same thing.
        rgb_model = image_rgb
        if rgb_model.shape[:2] != pred.shape:
            rgb_model = cv2.resize(rgb_model, (pred.shape[1], pred.shape[0]))

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
            packed = pack.encode(normal, depth_z, valid, z_range=(z_min, z_max))
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
            # Strictly outside, not merely equal: with an adaptive range the
            # bounds *are* the frame's extremes, so a non-strict test reports
            # the two boundary pixels on every frame. A pixel sitting exactly
            # on a bound encodes to code 1 / 255 and decodes back exactly.
            n_valid = int(valid.sum())
            near = int((valid & (depth_z < z_min)).sum())
            far = int((valid & (depth_z > z_max)).sum())
            if near or far:
                print(
                    f"packed depth: {near} px at/below {z_min:.3f}m affine z, "
                    f"{far} px at/above {z_max:.3f}m "
                    f"({(near + far) / max(n_valid, 1):.1%} of valid clipped) "
                    f"— widen --depth-min/--depth-max to bracket the scene"
                )

        # ---- depth in Lab's L channel, the frame's chroma in a/b
        # (--save-blend-Dab only). Format details live in
        # utils.normal_depth_pack; what matters here is that it is the *same*
        # depth code the octahedral image carries, so one sidecar decodes both
        # and a reader needs no second vocabulary. No normals: this carrier
        # holds the depth alone.
        dab_extra: dict[str, object] = {}
        if args.save_blend_dab or args.save_scale_dab:
            if args.save_scale_dab:
                # The one difference from --save-blend-Dab: the ends are read
                # off the frame instead of the gate, so the depth ramp spans
                # the luminance the picture actually uses.
                _dab_l_far, _dab_l_near = _scaled_dab_ramp(
                    cv2.cvtColor(rgb_model.astype(np.float32) / 255.0,
                                 cv2.COLOR_RGB2LAB)[..., 0],
                    DEFAULT_FAR_IS_BRIGHT,
                )
                _dab_tag, _dab_name = "Scaledab", "_scale_Dab.png"
                _dab_saver = _save_scale_dab_png
            else:
                _dab_l_far, _dab_l_near = _dab_ramp(
                    DEFAULT_L_DARK, DEFAULT_L_BRIGHT, DEFAULT_FAR_IS_BRIGHT
                )
                _dab_tag, _dab_name = "Blend   ", "_blend_Dab.png"
                _dab_saver = _save_blend_dab_png

            blended = encode_dab(
                rgb_model, depth_z, valid, z_range=(z_min, z_max),
                detail_alpha=DEFAULT_ALPHA,
                detail_d=DEFAULT_D,
                detail_sigma_color=DEFAULT_SIGMA_COLOR,
                detail_sigma_space=DEFAULT_SIGMA_SPACE,
                l_far=_dab_l_far, l_near=_dab_l_near,
            )
            _dab_saver(blended, out_base)
            print(f"Saved {_dab_tag}: {out_base}{_dab_name} "
                  f"(affine z {z_min:.3f}-{z_max:.3f}m, "
                  f"L* {_dab_l_near:.1f}-{_dab_l_far:.1f})")

            # The ramp's ends are part of the encoding, not of the codec, so a
            # reader has to be told them -- without this a retuned span decodes
            # silently wrong, which looks exactly like a correct decode.
            dab_extra = {
                "dab_l_far": np.float64(_dab_l_far),
                "dab_l_near": np.float64(_dab_l_near),
            }

            # The ramp's ends are part of the encoding, not of the codec, so a
            # reader has to be told them -- without this a retuned span decodes
            # silently wrong, which looks exactly like a correct decode.
            dab_extra = {
                "dab_l_far": np.float64(_dab_l_far),
                "dab_l_near": np.float64(_dab_l_near),
            }

        # ---- packed normals + depth, with the frame's texture over the
        # depth channel (--save-fused only). Same layout as the packed image,
        # so the detail term is the whole of the difference.
        if args.save_fused:
            fused = encode_fused(
                normal, depth_z, rgb_model, valid, z_range=(z_min, z_max),
                pack=pack,
                detail_alpha=DEFAULT_ALPHA,
                detail_d=DEFAULT_D,
                detail_sigma_color=DEFAULT_SIGMA_COLOR,
                detail_sigma_space=DEFAULT_SIGMA_SPACE,
            )
            _save_fused_png(fused, out_base)
            print(f"Saved fused : {out_base}_fused.png "
                  f"(affine z {z_min:.3f}-{z_max:.3f}m, detail a={DEFAULT_ALPHA})")

        # ---- the frame's own luminance + the normal's nx, ny in a/b
        # (--save-L-norm only). Reads as a photograph under strange light; the
        # normals are there but do not survive it well -- see the codec.
        l_norm_extra: dict[str, object] = {}
        if args.save_l_norm:
            _save_l_normal_png(
                encode_l_normal(rgb_model, normal, valid,
                                scale=DEFAULT_L_NORM_SCALE),
                out_base,
            )
            # The scale is part of the encoding: a reader that guesses it comes
            # back with normals of the wrong magnitude, which looks like a
            # normal map rather than like an error.
            l_norm_extra = {"l_normal_scale": np.float64(DEFAULT_L_NORM_SCALE)}
            print(f"Saved Lnorm : {out_base}_L_normal.png "
                  f"(nx,ny scale {DEFAULT_L_NORM_SCALE:.0f})")

        # ---- albedo + the normal's nx, ny (--save-Alb-norm only). The one
        # carrier here with no depth in it: R is a Retinex albedo, G and B are
        # the normal's x and y, and the hemisphere for the missing nz is
        # resolved by the reader from the pole rather than stored.
        albedo_extra: dict[str, object] = {}
        if args.save_alb_norm:
            # The helper's contract is L in [0, 255]; cv2's *float* Lab hands
            # back L in [0, 100]. Feeding it the raw value would darken the
            # albedo by a factor of 2.55 while still looking like an albedo,
            # which is exactly the kind of error that survives a glance.
            lab_model = cv2.cvtColor(
                rgb_model.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB
            )
            albedo, _, p_low, p_high = extract_albedo_retinex(
                lab_model[..., 0] / 100.0 * 255.0
            )

            _save_alb_normal_png(
                encode_alb_normal(albedo, normal, valid), out_base
            )
            # Recorded because the albedo's scaling is per-image: without these
            # the channel cannot be turned back into reflectance, and no two
            # frames' albedos are comparable.
            albedo_extra = {
                "albedo_p_low": np.float64(p_low),
                "albedo_p_high": np.float64(p_high),
            }
            print(f"Saved alb   : {out_base}_alb_normal.png "
                  f"(albedo p1-p99 {p_low:.4f}-{p_high:.4f})")

        # ---- sidecar (--save-scale). Implied by --save-packed, because the
        # packed image is undecodable without its z_min/z_max and its depth
        # channel is metric-less without the two scalars. --save-blend-Dab
        # needs it for exactly the same reason; --save-Alb-norm for the
        # albedo percentiles, which nothing else records. ----
        if (args.save_scale or args.save_packed or args.save_blend_dab
                or args.save_alb_norm):
            _save_scale_npz(
                pack,
                DepthScale(shift=shift, metric_scale=metric_scale),
                K,
                z_min,
                z_max,
                pred.shape,
                out_base,
                **albedo_extra,
                **dab_extra,
                **l_norm_extra,
            )
            print(f"Saved scale : {out_base}_scale.npz")

        # ---- visualization (--visualize): two PNGs + the mesh ----
        if args.visualize:
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
                images_u8=rgb_model[None].astype(np.uint8),
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
