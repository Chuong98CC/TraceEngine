"""Multi-image matching with the exported RoMaV2PT2 model.

Finds points visible in ALL given images. Points are anchored in the first
image (sampled with balanced sampling from the (0, 1) pair) and projected
into every other image through that pair's dense warp field.

Strategies:
    reference  -- match image 0 against every other image (N-1 calls);
                  positions come straight from each pair's warp field.
    cycle      -- additionally match all remaining pairs (N(N-1)/2 calls
                  total) and keep only points whose tracked positions agree
                  with every pair's warp in both directions (round-trip
                  error below --cycle-err).

Point selection is either the threshold method (--overlap-th, default) or
the top-k method (--top-k) -- exactly one, never both.

Usage:
    python multi_match.py imgA.jpg imgB.jpg [imgC.jpg ...] \
        [--strategy reference|cycle] [--num-corresp 500] \
        [--overlap-th 0.25 | --top-k 100] \
        [--model weights/romav2/romav2.pt2] [--out cache/multi_match_matches.npz] \
        [--viz cache/multi_match_vis.jpg]

With 2 images this reduces to the plain pairwise case :
one match call, and the overlap filter is the only difference from
model.sample() output.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from det_seg_models.romav2.romav2 import RoMaV2PT2
from det_seg_models.romav2.utils import to_pixel


def visualize(
    image_paths: list[str], matches_px: np.ndarray, out_path: str
) -> None:
    """Draw matches on images stacked side-by-side (all resized to image 0)."""
    imgs = [cv2.imread(p) for p in image_paths]
    H0, W0 = imgs[0].shape[:2]
    imgs = [cv2.resize(im, (W0, H0)) for im in imgs]
    stacked = np.hstack(imgs)
    palette = np.array(
        [
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
            [0, 255, 255],
            [255, 0, 255],
            [255, 128, 0],
            [128, 0, 255],
        ]
    )
    for m, pts in enumerate(matches_px):
        color = tuple(int(c) for c in palette[m % len(palette)])
        for j, (x, y) in enumerate(pts):
            cx, cy = int(x) + j * W0, int(y)
            cv2.circle(stacked, (cx, cy), 3, color, -1)
            if j > 0:
                px, py = int(pts[j - 1][0]) + (j - 1) * W0, int(pts[j - 1][1])
                cv2.line(stacked, (px, py), (cx, cy), color, 1)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, stacked)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "images",
        nargs="+",
        help="2+ image paths; matching points are anchored in the first image",
    )
    parser.add_argument(
        "--strategy",
        choices=["reference", "cycle"],
        default="reference",
        help="'reference': match image 0 vs the rest; 'cycle': match all pairs "
        "and keep only round-trip-consistent points (default: reference)",
    )
    parser.add_argument(
        "--num-corresp",
        type=int,
        default=500,
        help="candidate points sampled in image 0 before filtering (default: 500)",
    )
    sel = parser.add_mutually_exclusive_group()
    sel.add_argument(
        "--overlap-th",
        type=float,
        help="threshold method: minimum overlap confidence required in every "
        "image (default: 0.25); not allowed with --top-k",
    )
    sel.add_argument(
        "--top-k",
        type=int,
        help="top-k method: keep the K best candidates ranked by worst-case "
        "overlap across all images instead of thresholding; not allowed with "
        "--overlap-th",
    )
    parser.add_argument(
        "--cycle-err",
        type=float,
        default=0.01,
        help="max round-trip warp error in normalized coords, cycle mode "
        "(default: 0.01)",
    )
    parser.add_argument(
        "--model",
        help=f"path to the exported .pt2 program ",
    )
    parser.add_argument(
        "--out",
        help=f"where to save matches as .npz ",
    )
    parser.add_argument(
        "--viz",
        help=f"where to save the side-by-side visualization ",
    )
    parser.add_argument(
        "--no-viz", action="store_true", help="skip saving the visualization"
    )
    args = parser.parse_args()

    if len(args.images) < 2:
        parser.error("expected at least 2 images")

    model = RoMaV2PT2(args.model)

    positions, mask = model.match(
        args.images,
        strategy=args.strategy,
        num_corresp=args.num_corresp,
        overlap_th=args.overlap_th,
        top_k=args.top_k,
        cycle_th=args.cycle_err,
    )

    # Convert to per-image pixel coordinates.
    dims = [cv2.imread(p).shape[:2] for p in args.images]  # (H, W) each
    matches_px = torch.stack(
        [
            to_pixel(positions[:, j], H=dims[j][0], W=dims[j][1])
            for j in range(len(args.images))
        ],
        dim=1,
    )  # (K, N, 2) pixel coords

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, matches=matches_px.cpu().numpy(), image_paths=np.array(args.images))

    kept = int(mask.sum().item())
    sampled = int(mask.shape[0])
    select = f"top-k {args.top_k}" if args.top_k is not None else f"overlap-th {args.overlap_th}"
    print(
        f"strategy={args.strategy} select={select}: kept {kept}/{sampled} points "
        f"visible in all {len(args.images)} images"
    )
    print(f"saved matches to {args.out}")

    if not args.no_viz:
        visualize(args.images, matches_px.cpu().numpy(), args.viz)
        print(f"saved visualization to {args.viz}")


if __name__ == "__main__":
    main()
