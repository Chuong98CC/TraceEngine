"""Quantization error of the uint8 log-depth codec vs legacy uint16 mm .lz4.

Streaming depth .lz4 files written before 2026-09-03 (e.g. under
``eps_data/depth_pose_uint16``) store raw uint16 millimetres, lz4-frame
compressed. The current codec (``utils.depth_utils.LogDepthToUint8Transform``,
the format of ``save_depth_lz4`` / ``load_depth_lz4``) instead log-encodes
metres to 8 bits over ``[MIN_DEPTH, MAX_DEPTH]`` — ~2.6% log steps instead
of 1 mm steps.

This script loads one legacy uint16 mm file, rounds it through the codec
encode -> decode round trip exactly as the save/load path would, and reports
the average / max error caused by the 8-bit log quantization for depths
< MAX_DEPTH. Depths >= MAX_DEPTH clip to MAX_DEPTH in the codec and are
reported separately (excluded from the error stats).

Note: ``load_depth_lz4`` itself refuses legacy uint16 files (byte-count
mismatch), so the raw uint16 buffer is decompressed directly here.

Usage
-----
    uv run python test/depth_quantize_error.py \
        --path /data/astri_making_coffee_v1/eps_data/depth_pose_uint16/\
ep000000/subtask_00/depth_cam_head_stereo_left/frame_000000.lz4
"""

import argparse
from pathlib import Path

import lz4.frame
import numpy as np

from utils.depth_utils import MAX_DEPTH, MIN_DEPTH, LogDepthToUint8Transform

DEFAULT_PATH = (
    "/data/astri_making_coffee_v1/eps_data/depth_pose_uint16/"
    "ep000000/subtask_00/depth_cam_head_stereo_left/frame_000000.lz4"
)


def load_uint16_mm(path: Path) -> np.ndarray:
    """Decompress a legacy .lz4 file holding raw uint16 mm (little-endian)."""
    decoded = lz4.frame.decompress(Path(path).read_bytes())
    if len(decoded) % 2:
        raise ValueError(f"{path}: odd decompressed byte count {len(decoded)} "
                         f"-- not a uint16 depth map")
    return np.frombuffer(decoded, dtype="<u2")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Round a legacy uint16 mm .lz4 depth file through the "
                    "uint8 log codec and report quantization error "
                    "(depth < MAX_DEPTH only).",
    )
    parser.add_argument("--path", default=DEFAULT_PATH,
                        help="Legacy uint16 mm .lz4 depth file")
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        parser.error(f"--path not found: {path}")

    gt_mm = load_uint16_mm(path)  # raw uint16 mm, as originally saved
    n_px = gt_mm.size
    if n_px == 0:
        parser.error(f"{path}: empty depth map")

    codec = LogDepthToUint8Transform()  # [MIN_DEPTH, MAX_DEPTH] = [0.001, 1.501]
    # Round trip: encode the uint16 mm input as-is (auto-detects mm -> /1000),
    # then decode back to float32 metres.
    dec_m = codec.decode(codec.encode(gt_mm))

    # Input expressed in metres (1 mm grid) for the comparison;
    # analyse pixels with 0 < depth < MAX_DEPTH.
    gt_m = gt_mm.astype(np.float32) / 1000.0
    band = (gt_m > 0) & (gt_m < MAX_DEPTH)    # analysed range
    far = gt_m >= MAX_DEPTH                           # clips to MAX_DEPTH
    n_band = int(band.sum())

    print(f"file   : {path}")
    print(f"pixels : {n_px}   uint16 mm range {gt_mm.min()}..{gt_mm.max()}")
    if n_band == 0:
        print("no pixels with 0 < depth < MAX_DEPTH; nothing to compare")
        return
    frac = 100.0 * n_band / n_px
    print(f"band   : {n_band}/{n_px} pixels ({frac:.1f}%) with "
          f"0 < depth < MAX_DEPTH={MAX_DEPTH} m")

    err_mm = (dec_m - gt_m) * 1000.0           # signed, codec-metres vs mm truth
    err_mm = err_mm[band]
    abs_mm = np.abs(err_mm)
    gt_band = gt_m[band]
    rel_pct = abs_mm / (gt_band * 1000.0) * 100.0
    i_max = int(np.argmax(abs_mm))

    print(f"\nquantization error (uint8 log codec vs uint16 mm ground truth)\n"
          f"  pixels below MIN_DEPTH that collapse to 0: "
          f"{int((band & (dec_m <= 0.0)).sum())} (of {n_band})")
    print(f"  mean |err| : {abs_mm.mean():8.3f} mm   "
          f"({100.0 * abs_mm.mean() / (gt_band * 1000.0).mean():.3f} % of mean depth)")
    print(f"  p50/p90/p99: {np.percentile(abs_mm, [50, 90, 99])[0]:8.3f} / "
          f"{np.percentile(abs_mm, [50, 90, 99])[1]:.3f} / "
          f"{np.percentile(abs_mm, [50, 90, 99])[2]:.3f} mm")
    print(f"  max |err|  : {abs_mm[i_max]:8.3f} mm  at true depth "
          f"{gt_mm[band][i_max]} mm")
    print(f"  signed mean: {err_mm.mean():8.3f} mm "
          f"(negative = truncation biases decoded depth low)")
    print(f"  mean rel err: {rel_pct.mean():.3f} %    max rel err: "
          f"{rel_pct.max():.3f} %")

    n_far = int(far.sum())
    if n_far:
        print(f"\nnote: {n_far} pixels ({100.0 * n_far / n_px:.1f}%) at depth "
              f">= MAX_DEPTH would clip to {MAX_DEPTH} m (excluded above)")


if __name__ == "__main__":
    main()
