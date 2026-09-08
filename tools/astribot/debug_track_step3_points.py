"""Debug Step 4: track ONE Step-3 prompt's init points with the TAPIP3D
torch.export programs, in the infer_tapip3d.py style — a single controlled
run whose query set, anchor and frame window mirror run_step4_traces.py, and
whose output layout + 2D rendering use the known-good infer_tapip3d.py /
render_tracks conventions.

Unlike run_step4_traces.py (which groups prompts into per-role passes) this
tool tracks a single Step-3b prompt directory, so the anchor/row selection
of one prompt can be studied in isolation; the frames are still decoded
online from the LeRobotDataset and the geometry comes from the saved Step-2
depth_pose npz/lz4 files (nothing extracted to disk except the debug output
and the dumped render frames).

Run flow (identical to infer_tapip3d.py):
  1. read the Step-3 prompt (init_points.npz: keypoints (K, N, 2) at its
     key-frames, masks (N, H, W); init_points.json);
  2. discover the camera's Step-2 stems under depth_pose/<ep>/subtask_XX/
     depth_<cam>/ and build the trace window over the prompt's key-frame
     envelope (span_stems);
  3. anchor on the window's leading stem (run_step4 semantics: keypoints of
     the first key-frame column, tested against the anchor's depth/mask) and
     select the usable rows (<= 64 role points);
  4. assemble the exact-N queries (role rows + 32x32 support grid) and run
     Tapip3DStreamPT2 over the stems from the anchor on;
  5. save coords/visibs of the role points + metadata, dump the frames and
     render the 2D tracks video via render_tracks (the infer_tapip3d
     --visualize path).

Examples
--------
    python tools/astribot/debug_track_step3_points.py \
        --repo-id Kronze157/astri_making_coffee_vlva \
        --data-root /data/astri_making_coffee --episode-idx 0 \
        --subtask-idx 3 --camera-idx 0 --prompt brown_cup \
        --out-dir /tmp/debug_cup
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from flow_models.tapip3d.utils import (
    Tapip3D_PT2,
    Tapip3DStreamPT2,
    _DEFAULT_ENCODER,
    _DEFAULT_ITERATION,
)
from flow_models.tapip3d.utils._grid_utils import get_grid_queries
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from utils.depth_utils import load_depth_lz4
from utils.file_io.image_io import to_image_tensor
from utils.keyframe_utils import sampling_points_root, span_stems
from utils.streaming_utils import (
    compute_global_depth_roi,
    load_npz_batch,
    resize_batch_to_inference,
    unproject_xy_queries,
)
from utils.visualize.visualize_mask import to_pil

#: max tracked role keypoints (the shipped iteration graph is 64 object
#: query slots + 32x32 = 1024 support grid slots = 1088 total).
MAX_OBJECT_QUERIES = 64
SUPPORT_GRID_SIZE = 32
DEFAULT_IMAGE_SIZE = (480, 640)


def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser(
        description="Debug Step 4: track one Step-3 prompt's init points "
                    "with TAPIP3D in the infer_tapip3d.py style (online "
                    "frames, Step-2 depth_pose geometry, 2D tracks video).")
    p.add_argument("--repo-id", "-id", required=True,
                   help="dataset repo id as seen by LeRobotDataset")
    p.add_argument("--data-root", "-d", required=True,
                   help="root of the local dataset copy")
    p.add_argument("--episode-idx", type=int, default=0)
    p.add_argument("--subtask-idx", type=int, required=True,
                   help="sub-task segment ordinal (subtask_XX)")
    p.add_argument("--camera-idx", type=int, required=True,
                   help="index into the dataset's camera_keys (the trace "
                        "camera dir is named after the key's last segment)")
    p.add_argument("--prompt", type=str, required=True,
                   help="Step-3b prompt slug (folder under init_points/"
                        "<ep>/subtask_XX/<camera>/<prompt>)")
    p.add_argument("--prompt-dir", default=None,
                   help="override the Step-3 prompt directory entirely "
                        "(debug: inject a synthetic init_points.npz)")
    p.add_argument("--out-dir", "-o", required=True,
                   help="debug output root; frames/ subdir + tracks.mp4 "
                        "are written here")
    p.add_argument("--out-root", default=None,
                   help="eps_data root holding sampling_points/ and "
                        "depth_pose/ (default: <data-root>/eps_data)")
    p.add_argument("--anchor-frame", type=int, default=None,
                   help="override the anchor stem (default: run_step4's — "
                        "the window's leading stem)")
    p.add_argument("--max-role-points", type=int, default=MAX_OBJECT_QUERIES,
                   help="cap the number of tracked role points "
                        "(default: %(default)s)")
    p.add_argument("--y-min", type=float, default=None,
                   help="debug: drop role keypoints above this y (mask-res px)")
    p.add_argument("--y-max", type=float, default=None,
                   help="debug: drop role keypoints below this y (mask-res px)")
    p.add_argument("--encoder", default=_DEFAULT_ENCODER)
    p.add_argument("--iteration", default=_DEFAULT_ITERATION,
                   help="fused corr+updater .pt2 artifact (query count "
                        "auto-detected from the graph)")
    p.add_argument("--num-iters", type=int, default=6)
    p.add_argument("--vis-threshold", type=float, default=0.5)
    p.add_argument("--frames-dir", default=None,
                   help="debug: decode frames from this folder "
                        "(frame_<stem>.jpg files — the infer_tapip3d feed "
                        "path) instead of online from the dataset")
    p.add_argument("--no-depth-roi", action="store_true",
                   help="disable the global depth-ROI pre-scan (debug)")
    p.add_argument("--device", default=None, choices=["cuda", "cpu"])
    return p.parse_args(argv)


def _load_prompt_dir(pdir: Path) -> dict:
    """Step-3b prompt output (see run_step4_traces._read_prompt_dir)."""
    with np.load(pdir / "init_points.npz") as data:
        keypoints = data["keypoints"].astype(np.float32)   # (K, N, 2)
        frame_indices = np.asarray(data["frame_indices"], dtype=np.int64)
        masks = np.asarray(data["masks"], dtype=bool)      # (N, H, W)
    with open(pdir / "init_points.json") as f:
        meta = json.load(f)
    return {
        "slug": pdir.name,
        "dir": pdir,
        "prompt": meta.get("prompt", pdir.name),
        "num_keypoints": int(meta.get("num_keypoints", len(keypoints))),
        "keypoints": keypoints,
        "frame_indices": frame_indices,
        "masks": masks,
    }


def _geometry_at(depth_dir: Path, idx: int):
    """(depth (H, W) metres, intrinsics (3, 3), extrinsics (4, 4)) of one
    Step-2 stem — the run_step4_traces._geometry_at triplet."""
    with np.load(depth_dir / f"frame_{idx:06d}.npz") as data:
        shape = tuple(int(v) for v in data["shape"])
        depth = load_depth_lz4(depth_dir / f"frame_{idx:06d}.lz4", shape)
        extr = data["extrinsics"] if "extrinsics" in data else data["extrinsic"]
        if extr.shape == (3, 4):
            extr = np.vstack([extr, [0.0, 0.0, 0.0, 1.0]])
        intrs = data["intrinsics"] if "intrinsics" in data else data["intrinsic"]
    return (depth.astype(np.float32), intrs.astype(np.float32),
            extr.astype(np.float32))


def _usable_rows(prompt: dict, kf_j: int, depth: np.ndarray, max_rows: int):
    """Rows of key-frame column kf_j usable on ``depth``: keypoint inside
    the key-frame's SAM3 mask (when the mask exists) and on a valid depth
    pixel (keypoints rescaled from the mask resolution to the depth
    resolution first) — mirrors run_step4_traces._row_pixels."""
    kp = prompt["keypoints"][:, kf_j]                 # (K, 2)
    mask_j = prompt["masks"][kf_j]
    k = len(kp)
    valid = np.ones(k, dtype=bool)
    if mask_j.any():
        x = np.round(kp[:, 0]).astype(np.int64)
        y = np.round(kp[:, 1]).astype(np.int64)
        np.clip(x, 0, mask_j.shape[1] - 1, out=x)
        np.clip(y, 0, mask_j.shape[0] - 1, out=y)
        valid &= mask_j[y, x]
    if not valid.any():
        return np.empty(0, dtype=np.int64)
    h, w = mask_j.shape
    dh, dw = depth.shape
    px = kp.copy()
    if (h, w) != (dh, dw):
        px[:, 0] *= (dw - 1) / (w - 1)
        px[:, 1] *= (dh - 1) / (h - 1)
    ji_x = np.round(px[:, 0]).astype(np.int64)
    ji_y = np.round(px[:, 1]).astype(np.int64)
    np.clip(ji_x, 0, dw - 1, out=ji_x)
    np.clip(ji_y, 0, dh - 1, out=ji_y)
    valid &= depth[ji_y, ji_x] > 0
    rows = np.nonzero(valid)[0]
    if len(rows) > max_rows:
        rows = rows[:max_rows]
    return rows


def main() -> None:
    args = parse_args()
    # validated numerics of the TAPIP3D path (fp32, TF32 off)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_root = Path(args.out_root) if args.out_root else \
        Path(args.data_root) / "eps_data"

    # --- Step-3 prompt -------------------------------------------------------
    ds_meta = LeRobotDatasetMetadata(repo_id=args.repo_id, root=args.data_root)
    cam_key = ds_meta.camera_keys[args.camera_idx]
    cam = cam_key.rsplit(".", 1)[-1]
    init_root = sampling_points_root(args.data_root) / "init_points"
    pdir = Path(args.prompt_dir) if args.prompt_dir else \
        (init_root / f"ep{args.episode_idx:06d}"
         / f"subtask_{args.subtask_idx:02d}" / cam / args.prompt)
    prompt = _load_prompt_dir(pdir)
    print(f"prompt: {prompt['prompt']!r} keyframes "
          f"{prompt['frame_indices'].tolist()} "
          f"keypoints {prompt['keypoints'].shape} "
          f"masks {prompt['masks'].shape}")

    # --- Step-2 stems of the camera -----------------------------------------
    depth_dir = (out_root / "depth_pose" / f"ep{args.episode_idx:06d}"
                 / f"subtask_{args.subtask_idx:02d}" / f"depth_{cam}")
    if not depth_dir.is_dir():
        raise FileNotFoundError(depth_dir)
    stems = sorted(int(p.stem.rsplit("_", 1)[-1])
                   for p in depth_dir.glob("*.npz"))
    print(f"camera {cam}: {len(stems)} Step-2 stems "
          f"{stems[0]}..{stems[-1]} in {depth_dir}")

    # --- trace window + anchor (run_step4 semantics) -------------------------
    window = span_stems(stems, [int(prompt["frame_indices"][0])],
                        [int(prompt["frame_indices"][-1])])
    anchor_abs = args.anchor_frame if args.anchor_frame is not None \
        else window[0]
    if anchor_abs not in window:
        raise ValueError(f"--anchor-frame {anchor_abs} not in the window "
                         f"{window[0]}..{window[-1]}")
    depth, intrs, extr = _geometry_at(depth_dir, anchor_abs)
    # key-frame column: the anchor stem itself when it is one of the
    # prompt's key-frames, else the first column (object static before its
    # first key-frame — run_step4's _usable_at logic)
    kfs = list(prompt["frame_indices"])
    j = kfs.index(anchor_abs) if anchor_abs in kfs else 0
    rows = _usable_rows(prompt, j, depth, args.max_role_points)
    if args.y_min is not None or args.y_max is not None:
        kp_y = prompt["keypoints"][rows, j, 1]
        keep = np.ones(len(rows), dtype=bool)
        if args.y_min is not None:
            keep &= kp_y >= args.y_min
        if args.y_max is not None:
            keep &= kp_y <= args.y_max
        rows = rows[keep]
    if len(rows) == 0:
        print(f"no usable rows on the anchor {anchor_abs} (col {j})")
        sys.exit(1)
    print(f"window {window[0]}..{window[-1]}  anchor {anchor_abs}  "
          f"col {j}  {len(rows)} usable rows of {len(prompt['keypoints'])}")

    # --- exact-N queries: role points + 32x32 support grid ------------------
    kp = prompt["keypoints"][rows, j]
    h, w = prompt["masks"][j].shape
    dh, dw = depth.shape
    px = kp.copy()
    if (h, w) != (dh, dw):
        px[:, 0] *= (dw - 1) / (w - 1)
        px[:, 1] *= (dh - 1) / (h - 1)
    init = unproject_xy_queries(px, depth, intrs, extr)
    assert init is not None, "rows already depth-filtered"
    print(f"role query px (depth res {dw}x{dh}): x {px[:, 0].min():.0f}.."
          f"{px[:, 0].max():.0f}  y {px[:, 1].min():.0f}..{px[:, 1].max():.0f}")

    print("Loading TAPIP3D .pt2 artifacts...")
    pt2 = Tapip3D_PT2(args.encoder, args.iteration,
                      image_size=DEFAULT_IMAGE_SIZE,
                      num_iters=args.num_iters)
    need = pt2.num_queries - init.shape[0]
    if need < 0 or pt2.num_queries != SUPPORT_GRID_SIZE ** 2 + MAX_OBJECT_QUERIES:
        raise SystemExit(
            f"iteration graph has {pt2.num_queries} fixed queries, expected "
            f"{SUPPORT_GRID_SIZE ** 2 + MAX_OBJECT_QUERIES}")
    support = get_grid_queries(
        SUPPORT_GRID_SIZE,
        torch.from_numpy(depth)[None], torch.from_numpy(intrs)[None],
        torch.from_numpy(extr)[None]).squeeze(0)
    if support.shape[0] < need:
        raise SystemExit(f"support grid {support.shape[0]} < needed {need}")
    queries = torch.cat([init, support[:need]])
    print(f"{len(rows)} role queries + {need} support = "
          f"{queries.shape[0]} (graph {pt2.num_queries})")

    # --- streamed inference over the stems from the anchor on ----------------
    steps = window[window.index(anchor_abs):]
    inf_h, inf_w = pt2.image_size
    file_list = [(int(t), None) for t in steps]
    if args.frames_dir is not None:
        # the infer_tapip3d feed path: decode the dumped frame_<stem>.jpg files
        frames = torch.stack([
            to_image_tensor(str(Path(args.frames_dir) / f"frame_{t:06d}.jpg"))
            for t in steps])
        print(f"frames decoded from {args.frames_dir} "
              f"(stem {steps[0]}..{steps[-1]})")
    else:
        ds = LeRobotDataset(repo_id=args.repo_id, root=args.data_root,
                            download_videos=False)
        frames = torch.stack(
            [to_image_tensor(np.asarray(to_pil(ds[t][cam_key]), dtype=np.uint8))
             for t in steps])

    def batches():
        for s in range(0, len(steps), pt2.seq_len):
            end = min(s + pt2.seq_len, len(steps))
            video = frames[s:end]
            geo = load_npz_batch(str(depth_dir), file_list, s, end)
            yield resize_batch_to_inference(video, geo, inf_h, inf_w)

    depth_roi = None if args.no_depth_roi else \
        compute_global_depth_roi(str(depth_dir), file_list, inf_h, inf_w)
    si = Tapip3DStreamPT2(pt2, queries, depth_roi=depth_roi)
    with torch.inference_mode():
        coords_all, visibs_logits_all = si.run(batches(), len(steps))

    n_role = init.shape[0]
    coords = coords_all[:, :n_role].cpu().numpy()
    visibs = (torch.sigmoid(visibs_logits_all[:, :n_role]) >=
              args.vis_threshold).cpu().numpy()
    print(f"tracked {len(steps)} steps; visible per row "
          f"{[int(v.sum()) for v in visibs]}")

    # --- save + render (infer_tapip3d output layout) -------------------------
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    frame_files = []
    from PIL import Image
    for t, s in enumerate(steps):
        np_frame = frames[t].permute(1, 2, 0).numpy()  # CHW uint8 -> HWC
        path = frames_dir / f"frame_{s:06d}.jpg"
        Image.fromarray(np_frame).save(path)
        frame_files.append({"index": int(s), "path": str(path)})

    np.save(out_dir / "coords.npy", coords)
    np.save(out_dir / "visibs.npy", visibs)
    np.save(out_dir / "queries.npy", queries[:n_role].cpu().numpy())
    meta = {
        "image_dir": str(frames_dir.absolute()),
        "depth_dir": str(depth_dir.absolute()),
        "total_frames": int(len(steps)),
        "num_queries": int(n_role),
        "inference_resolution": [inf_h, inf_w],
        "query_source": "step3_init_points",
        "prompt": prompt["prompt"],
        "anchor_frame": int(anchor_abs),
        "keyframe_column": int(j),
        "query_keypoint_rows": [int(i) for i in rows],
        "pixels": [[float(x), float(y)] for x, y in px],
        "steps": [int(s) for s in steps],
        "frame_indices": [int(s) for s in steps],
        "frame_files": frame_files,
        "num_iters": args.num_iters,
        "vis_threshold": args.vis_threshold,
        "model": {"encoder": str(Path(args.encoder).absolute()),
                  "iteration": str(Path(args.iteration).absolute())},
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    from utils.visualize.visualize_tapip3d import render_tracks
    video_path = render_tracks(out_dir, fps=10)
    print(f"video saved to {video_path}")


if __name__ == "__main__":
    main()
