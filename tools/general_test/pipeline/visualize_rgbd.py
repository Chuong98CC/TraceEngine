"""Visualise RGB-D data: either the Astribot head RGB-D camera (ground-truth
depth camera) or an explicit folder pair (RGB images + depth .lz4 files).

Two mutually exclusive input sources:

1. **Camera mode** (``--camera_name``): loads the head RGB frame and its
   aligned depth frame via the astribot dataloader, recovers metric depth from
   the grey-scale depth image.
2. **Folder mode** (``--rgb_dir`` + ``--depth_npz_dir``): pairs RGB images
   (``<stem>.jpg/.jpeg/.png``) with depth .lz4 files (``<stem>.lz4``, raw
   uint16 mm — see utils.astribot_dataloader.load_depth_lz4) and pose NPZ
   files (``<stem>.npz`` with ``extrinsics`` 3x4/4x4 and ``intrinsics`` 3x3
   keys) by stem, and processes the pair at position ``--frame_index`` in
   the sorted stems.

Exactly one of the two sources must be set.

Writes, per frame:

1. Visualisation → side-by-side ``[RGB | colour-mapped depth]`` JPEG (save_depth_vis).
2. Point cloud   → RGB-coloured ``.glb`` back-projected with the intrinsics
   (export_glb).

Usage
-----
python tools/general_test/pipeline/visualize_rgbd.py --camera_name head_rgbd \
    --frame_index 0 --save_viz --save_glb
python tools/general_test/pipeline/visualize_rgbd.py --rgb_dir path/to/rgb \
    --depth_npz_dir path/to/depth --save_viz --save_glb
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from utils.astribot_dataloader import _scale_intrinsics_matrix, load_depth_lz4, load_rgbd
from utils.visualize.visualize_depth import export_glb, save_depth_vis


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualise RGB-D data (Astribot camera or RGB/depth-npz folders).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--camera_name", type=str, default=None,
        help="Astribot camera name (e.g. head_rgbd). Mutually exclusive with "
             "--rgb_dir/--depth_npz_dir.",
    )
    parser.add_argument(
        "--rgb_dir", type=str, default=None,
        help="Folder of RGB images (<stem>.jpg/.jpeg/.png). Mutually exclusive "
             "with --camera_name.",
    )
    parser.add_argument(
        "--depth_npz_dir", type=str, default=None,
        help="Folder of depth .lz4 files (<stem>.lz4, raw uint16 mm) and pose "
             "NPZ files (<stem>.npz with 'extrinsics' and 'intrinsics' keys). "
             "Mutually exclusive with --camera_name.",
    )
    parser.add_argument(
        "--frame_index", default=0, type=int,
        help="Frame index to process in camera mode (default: %(default)s).",
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
# Folder-mode loading
# ---------------------------------------------------------------------------

def _load_folder_pair(rgb_dir: Path, npz_dir: Path, stem: str):
    """Load ``(rgb, depth_m, ext, ixt)`` for one stem from the folder pair.

    ``<npz_dir>/<stem>.lz4`` holds the raw uint16 mm depth (see
    utils.astribot_dataloader.load_depth_lz4), reshaped to the RGB frame
    size; ``<npz_dir>/<stem>.npz`` holds ``extrinsics`` (3x4/4x4) and
    ``intrinsics`` (3x3). The intrinsics are recorded at the depth
    resolution, so resizing RGB to the depth resolution needs no intrinsic
    rescaling here.
    """
    img_path = None
    for ext in (".jpg", ".jpeg", ".png"):
        candidate = rgb_dir / f"{stem}{ext}"
        if candidate.exists():
            img_path = candidate
            break
    if img_path is None:
        raise FileNotFoundError(f"No RGB image with stem {stem!r} in {rgb_dir}")
    rgb_bgr = cv2.imread(str(img_path))
    if rgb_bgr is None:
        raise FileNotFoundError(f"Cannot read RGB frame: {img_path}")
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

    lz4_path = npz_dir / f"{stem}.lz4"
    if not lz4_path.exists():
        raise FileNotFoundError(f"No depth lz4 with stem {stem!r}: {lz4_path}")
    depth_m = (load_depth_lz4(lz4_path, shape=rgb.shape[:2]) / 1000.0).astype(np.float32)

    npz_path = npz_dir / f"{stem}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"No pose npz with stem {stem!r}: {npz_path}")
    with np.load(npz_path) as data:
        assert "extrinsics" in data, f"{npz_path} missing required 'extrinsics' key"
        assert "intrinsics" in data, f"{npz_path} missing required 'intrinsics' key"
        ext = data["extrinsics"].astype(np.float32)
        ixt = data["intrinsics"].astype(np.float32)

    return rgb, depth_m, ext, ixt


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def _save_outputs(rgb, depth_m, ext, ixt, tag: str, out_dir: Path, args) -> None:
    """Write the viz JPEG and/or .glb for one frame (``tag`` = output stem)."""
    # ---- visualisation frame ----
    if args.save_viz:
        vis_dir = save_depth_vis(
            rgb[None],
            depth_m[None],
            out_dir,
            filenames=[f"{tag}.jpg"],
        )
        print(f"Saved viz : {vis_dir / f'{tag}.jpg'}")

    # ---- point cloud ----
    if args.save_glb:
        pc_path = out_dir / f"{tag}.glb"
        export_glb(
            depth=depth_m[None].astype(np.float32),
            intrinsics=ixt[None],
            extrinsics=ext[None],
            images_u8=rgb[None].astype(np.uint8),
            conf=None,
            out_path=str(pc_path),
        )
        print(f"Saved glb : {pc_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    # ---- input source: camera XOR folders ----
    use_camera = args.camera_name is not None
    use_folders = args.rgb_dir is not None or args.depth_npz_dir is not None
    assert use_camera != use_folders, (
        "Set exactly one input source: --camera_name, or --rgb_dir with "
        "--depth_npz_dir."
    )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if use_camera:
        # ---- camera mode (single frame) ----
        rgb_path, depth_m, ext, ixt, ixt_res = load_rgbd(args.frame_index, args.camera_name)

        rgb_bgr = cv2.imread(rgb_path)
        if rgb_bgr is None:
            raise FileNotFoundError(f"Cannot read RGB frame: {rgb_path}")
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

        # Align RGB to the depth resolution (shared by viz + glb), and rescale
        # the intrinsics from their recorded resolution (ixt_res) to the depth
        # resolution so the point-cloud back-projection stays consistent.
        h_d, w_d = depth_m.shape
        if rgb.shape[:2] != (h_d, w_d):
            rgb = cv2.resize(rgb, (w_d, h_d))
        if (int(ixt_res[0]), int(ixt_res[1])) != (w_d, h_d):
            ixt = _scale_intrinsics_matrix(ixt, int(ixt_res[0]), int(ixt_res[1]), w_d, h_d)

        _save_outputs(
            rgb, depth_m, ext, ixt, f"rgbd_{args.camera_name}_{args.frame_index:06d}", out_dir, args,
        )
    else:
        # ---- folder mode (single frame, selected by index into sorted stems) ----
        assert args.rgb_dir is not None and args.depth_npz_dir is not None, (
            "Folder mode requires both --rgb_dir and --depth_npz_dir."
        )
        rgb_dir, npz_dir = Path(args.rgb_dir), Path(args.depth_npz_dir)
        stems = sorted(
            {p.stem for p in rgb_dir.iterdir() if p.is_file()}
            & {p.stem for p in npz_dir.glob("*.npz")}
        )
        assert stems, f"No matching RGB/depth pairs between {rgb_dir} and {npz_dir}"
        assert args.frame_index < len(stems), (
            f"--frame_index {args.frame_index} out of range: "
            f"only {len(stems)} matching pairs"
        )

        stem = stems[args.frame_index]
        print(f"Processing {stem}")
        rgb, depth_m, ext, ixt = _load_folder_pair(rgb_dir, npz_dir, stem)
        _save_outputs(rgb, depth_m, ext, ixt, f"rgbd_{stem}", out_dir, args)

    print("Done.")


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
