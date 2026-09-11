"""Visualise RGB-D data — an Astribot head RGB-D frame (ground-truth depth)
or an explicit RGB/depth folder pair. Exactly one input source:

- ``--camera_name``: the Astribot head RGB-D camera; metric depth is
  recovered from the grey-scale depth image of ``--frame_index``.
- ``--rgb_dir`` + ``--depth_npz_dir``: an RGB image folder
  (``frame_<idx>.jpg/.jpeg/.png``) plus a ``depth_pose/<camera>`` folder
  (``depth.lz4`` container + ``poses.npz``, see utils.depth_pose_io);
  ``--frame_index`` picks the position in the sorted frame indices present
  in both.

Per selected frame it writes a side-by-side ``[RGB | depth colour-map]``
JPEG (--save_viz) and a coloured .glb point cloud (--save_glb). This is a
debug/visualisation helper, not part of the depth-streaming pipeline.

Usage
-----
    python tools/general_test/pipeline/visualize_rgbd.py
        --camera_name head_rgbd --frame_index 0 --save_viz --save_glb
    python tools/general_test/pipeline/visualize_rgbd.py
        --rgb_dir path/to/rgb --depth_npz_dir path/to/depth --save_glb
"""

import argparse
import re
from pathlib import Path

import cv2
import numpy as np

from utils.astribot_dataloader import _scale_intrinsics_matrix, load_rgbd
from utils.depth_pose_io import DepthPoseReader
from utils.streaming_utils import load_stream_data
from utils.visualize.visualize_depth import export_glb, save_depth_vis

#: RGB frame file names of the folder mode (the frame_<idx> stem of the
#: image folders).
_FRAME_RE = re.compile(r"^frame_(\d+)\.(?:jpg|jpeg|png)$")


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
        help="Depth_pose camera folder (depth.lz4 container + poses.npz, see "
             "utils.depth_pose_io). Mutually exclusive with --camera_name.",
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

def _load_folder_pair(rgb_dir: Path, depth_dir: Path, frame_index: int):
    """Load ``(rgb, depth_m, ext, ixt)`` for one frame from the folder pair.

    ``depth_dir`` is a depth_pose camera folder (``depth.lz4`` +
    ``poses.npz``): ``load_stream_data`` decodes this absolute frame's depth
    to float32 metres and returns its pose (``extrinsics`` 3x4/4x4,
    ``intrinsics`` 3x3). RGB is resized to the recorded depth resolution —
    the intrinsics are recorded at the depth resolution, so no intrinsic
    rescaling is needed (the same convention as the camera mode).
    """
    depth_m, ext, ixt = load_stream_data(depth_dir, frame_index)
    h_d, w_d = depth_m.shape

    stem = f"frame_{int(frame_index):06d}"
    img_path = None
    for suffix in (".jpg", ".jpeg", ".png"):
        candidate = rgb_dir / f"{stem}{suffix}"
        if candidate.exists():
            img_path = candidate
            break
    if img_path is None:
        raise FileNotFoundError(f"No RGB image with stem {stem!r} in {rgb_dir}")
    rgb_bgr = cv2.imread(str(img_path))
    if rgb_bgr is None:
        raise FileNotFoundError(f"Cannot read RGB frame: {img_path}")
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[:2] != (h_d, w_d):
        rgb = cv2.resize(rgb, (w_d, h_d))

    return rgb, depth_m, ext.astype(np.float32), ixt.astype(np.float32)


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
        # ---- folder mode (single frame, index into the frame indices the
        # image folder and the depth_pose store share) ----
        assert args.rgb_dir is not None and args.depth_npz_dir is not None, (
            "Folder mode requires both --rgb_dir and --depth_npz_dir."
        )
        rgb_dir, depth_dir = Path(args.rgb_dir), Path(args.depth_npz_dir)
        with DepthPoseReader(depth_dir) as reader:
            depth_indexes = {int(i) for i in reader.frame_indices}
        image_indexes = {int(m.group(1)) for p in rgb_dir.iterdir()
                         if p.is_file() and (m := _FRAME_RE.match(p.name))}
        frame_indexes = sorted(image_indexes & depth_indexes)
        assert frame_indexes, (
            f"No frame_<idx> RGB images of {rgb_dir} match the frame indices "
            f"of {depth_dir}"
        )
        assert args.frame_index < len(frame_indexes), (
            f"--frame_index {args.frame_index} out of range: "
            f"only {len(frame_indexes)} matching frames"
        )

        frame_index = frame_indexes[args.frame_index]
        print(f"Processing frame {frame_index}")
        rgb, depth_m, ext, ixt = _load_folder_pair(rgb_dir, depth_dir, frame_index)
        _save_outputs(rgb, depth_m, ext, ixt,
                      f"rgbd_frame_{int(frame_index):06d}", out_dir, args)

    print("Done.")


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
