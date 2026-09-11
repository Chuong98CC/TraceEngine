"""Step 4 online — per-sub-task 3D point tracking with the TAPIP3D
torch.export programs, straight from the Step-2/Step-3 results and the
LeRobotDataset (nothing extracted to disk).

For every sub-task of the selected episodes it loads the per-prompt init
points of Step 3b (SAM3 masks + RoMAv2 keypoints under the Step-3
sampling_points folder — <root>/ep{ep:03d}/subtask_{k:02d}/sampling_points/
init_points/<camera>/), anchors them on the Step-2 depth + pose outputs
(<root>/ep{ep:03d}/subtask_{k:02d}/depth_pose/<camera>/ — one indexed
depth.lz4 container + poses.npz per (sub-task, camera), see
utils.depth_pose_io), and tracks the 3D positions of the points with
TAPIP3D over the sub-task's streamed frames — the RGB frames are decoded
**online** from the dataset, the geometry from the saved depth_pose store:

<root> is the episodes root (--out-dir, default <data-root>/episodes); every
Step-1..4 artifact of an episode lives under it (utils.astribot_paths).

    Step-2 stems + Step-3 keypoints (per prompt)
               │
               ▼  one TAPIP3D pass per role (object / manipulator)
    ┌──────────────────────────────┐
    │  coords + visibs per prompt  │
    └──────────────────────────────┘

Each role pass traces its prompts over the Step-2 stems inside a window
bounded by the prompts' Step-3 key-frames (span_stems): from the last
stem at-or-before the earliest first key-frame to the first stem
at-or-after the latest last key-frame. Every "stem" here is an absolute
dataset frame index — the container's frame_indices (utils.depth_pose_io),
the same indices Step 3 puts in its npz/JSON — never a ``frame_%06d`` name.

- the **manipulator** key-frames span the whole sub-task ([start frame ..
  last frame]), so its pass tracks from the sub-task's first stem to its
  last, as before;
- the **object** key-frames span only the transport ([gripper close ..
  gripper open] — Step 3b samples between the sub-task's 2nd and 2nd-to-
  last key-frame), so its pass tracks the stems from just before the
  close to just after the open; the object is static outside that span,
  so its close-frame keypoints stay exact on the stem right before the
  close (the gripper has not occluded them yet).

A keypoint is *usable* on a key-frame when it is a surviving Step-3
keypoint lying inside that key-frame's SAM3 mask (masks[j].any() missing
-> unconstrained) with valid depth at its pixel. Roles come from the
dataset annotations meta/subtasks.csv ([object, manipulator] of the
sub-task's row): Step 3a recorded the segment's canonical sub-task label
(subtask_index) in the detections JSON, and that label resolves the row —
a segment without a recorded label is tracked unlabelled, never
role-matched by the segment ordinal.

The shipped TAPIP3D iteration graph has a fixed query count (1088), so
each pass tracks up to 64 object / 128 manipulator role keypoints (the
manipulator is seeded denser so enough points survive the static
filters) + a full-frame support grid trimmed/padded deterministically to
reach exactly 1088 (a full 128-query manipulator pass runs 960 support
points: the 32x32 grid minus the cells nearest the role keypoints'
anchor-frame pixels — see _support_queries — so the surviving support
spreads away from the tracked points; random padding points are sampled
among the anchor frame's valid-depth pixels,
np.random.default_rng(seed + role_index)). The filters only remove
points, so the output keeps at most MAX_KEPT_KEYPOINTS (64) keypoints
per prompt — a 128-seed manipulator prompt that outlives the filters
with more survivors is truncated to its first 64 in seed order.

Every camera of a sub-task is tracked separately over its own Step-2
depth_pose/<camera> store — the cameras come from the per-camera
init-points subtrees of Step 3b (.../subtask_{k:02d}/sampling_points/
init_points/<camera>/), the role labels from that camera's Step-3a
detections JSON (.../subtask_{k:02d}/sampling_points/detections/
<camera>.json: one file per (episode, sub-task, camera), holding just that
sub-task's payload). Output, per camera under
<root>/ep{ep:03d}/subtask_{k:02d}/traces/<camera>/ and per prompt under
<prompt_slug>/: coords.npy (T, Q, 3) world-space traces, visibs.npy
(T, Q) visibility flags, queries.npy (Q, 4) query points (home frame, x,
y, z) and metadata.json — the per-prompt file is the only record Step 4
leaves of an outcome (there is no camera-level roll-up).
Visualization: tools/astribot/visualize_step4_traces.py renders
per-camera videos of the traces.

Examples
--------
    # Track every sub-task of episode 0 (object + manipulator roles)
    python tools/astribot/run_step4_traces.py
        --repo-id Kronze157/astribot_making_coffee_vlva_full
        --data-root /data/astri_making_coffee --episode-idxes 0
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets import LeRobotDatasetMetadata
from tqdm import tqdm

from flow_models.tapip3d.utils import (
    Tapip3D_PT2,
    Tapip3DStreamPT2,
    _DEFAULT_ENCODER,
    _DEFAULT_ITERATION,
)
from flow_models.tapip3d.utils._grid_utils import get_grid_queries
from tools.astribot.extract_frames import DataExtract
from utils import astribot_paths as ap
from utils.depth_pose_io import DepthPoseReader
from utils.file_io.image_io import to_image_tensor
from utils.keyframe_utils import load_subtask_meta, span_stems
from utils.streaming_utils import (
    compute_global_depth_roi,
    load_npz_batch,
    load_stream_data,
    resize_batch_to_inference,
    unproject_xy_queries,
)
from utils.visualize.visualize_mask import to_pil

#: role output order of the sub-task annotations (meta/subtasks.csv columns).
ROLE_ORDER = ("object", "manipulator")

#: keep the 64-keypoint density.
ROLE_MAX_KEYPOINTS = {"object": 64, "manipulator": 128}
SUPPORT_GRID_SIZE = 32
MAX_KEPT_KEYPOINTS = ROLE_MAX_KEYPOINTS["object"]
MIN_PIXEL_MOVING=5

def _role_keypoint_cap(role: str | None) -> int:
    """Max role keypoints of one pass: 128 for the manipulator, 64 for
    object and role-less (unlabelled) prompts."""
    return ROLE_MAX_KEYPOINTS.get(role, ROLE_MAX_KEYPOINTS["object"])


def _cap_keypoint_survivors(sel: np.ndarray, cap: int
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Per-prompt output cap on the filtered survivors.

    A prompt's survivor mask (True = survived the filters, in seed order)
    is truncated to its first ``cap`` survivors. Returns (sel_capped,
    over) where over holds the original positions capped away (empty when
    nothing was). The input mask is not mutated.
    """
    keep_idx = np.nonzero(sel)[0]
    if len(keep_idx) <= cap:
        return sel, np.empty(0, dtype=np.int64)
    out = sel.copy()
    out[keep_idx[cap:]] = False
    return out, keep_idx[cap:]

def _slugify(text: str) -> str:
    """Prompt folder slug — mirrors run_object_init_points.py."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Step 4 online: track the Step-3 keypoints with TAPIP3D "
                    "over the Step-2 depth + pose outputs — one pass per role "
                    "(each pass traces the Step-2 stems around its prompts' "
                    "key-frames: the object's close..open transport, the "
                    "manipulator's whole sub-task), RGB frames decoded online "
                    "from the dataset."
    )
    parser.add_argument("--repo-id", "-id", required=True,
                        help="dataset repo id as seen by LeRobotDataset")
    parser.add_argument("--data-root", "-d", required=True,
                        help="root of the local dataset copy")
    parser.add_argument("--camera-idxes", "-c", nargs="+", type=int,
                        default=None,
                        help="dataset camera indices eligible for tracking "
                             "(default: all; the tracked cameras of each "
                             "sub-task are the ones with Step-3 init "
                             "points on disk)")
    parser.add_argument("--episode-idxes", "-e", nargs="*", type=int,
                        default=None,
                        help="only process these episode indices (default: "
                             "all episodes with Step-3 init points on disk)")
    parser.add_argument("--max-episodes", "-x", type=int, default=None,
                        help="cap the number of processed episodes")
    parser.add_argument("--out-dir", "-o", default=None,
                        help="episodes root (default: <data-root>/episodes); "
                             "per (episode, sub-task, camera): the Step-2 "
                             "depth_pose read "
                             "(…/subtask_{k:02d}/depth_pose/<camera>/), the "
                             "Step-3 sampling_points inputs "
                             "(…/sampling_points/{detections,init_points}/) "
                             "and the traces write "
                             "(…/subtask_{k:02d}/traces/<camera>/) all live "
                             "under it")
    parser.add_argument("--vis-threshold", type=float, default=0.5,
                        help="sigmoid visibility threshold for visibs "
                             "(default: %(default)s)")
    parser.add_argument("--filter-visible", action="store_true",
                        help="drop the tracked columns that are not always "
                             "visible, or whose invisible runs are long or "
                             "reappear far from where they were last "
                             "visible (see --max-invisible-stems / "
                             "--max-reappear-ratio; the reappearance "
                             "displacement is pixel-space, projected "
                             "through the reappearance stem's own pose, "
                             "so camera motion between the stems cancels "
                             "and briefly-occluded movers are not read "
                             "as track failures)")
    parser.add_argument("--filter-static-pixel", action="store_true",
                        help="additionally drop the columns that never "
                             "move in the tracked camera's pixels: the "
                             "max displacement of the column's "
                             "reprojection (per stem, through that "
                             "stem's own pose) from its first-appear "
                             "pixel is at most --min-motion-pixels. "
                             "Pixel motion is robust to depth noise and "
                             "pose drift — depth-noise wander along the "
                             "viewing ray barely moves the "
                             "reprojection. Applied after "
                             "--filter-visible")
    parser.add_argument("--max-invisible-stems", type=int, default=8,
                        help="an invisible run of this many consecutive "
                             "trace stems or more drops the column (the "
                             "threshold is strict; default: %(default)s)")
    parser.add_argument("--max-reappear-ratio", type=float, default=3.0,
                        help="the reappearance allowance of a bounded "
                             "invisible run is this many times the "
                             "column's fastest per-stem pixel step while "
                             "continuously visible — a genuinely fast "
                             "mover blinks out and comes back ~1x its own "
                             "step (kept), a snapped static column jumps "
                             "against a ~0px envelope (dropped); the "
                             "criterion is strict; default: %(default)s)")
    parser.add_argument("--min-motion-pixels", type=float, default=MIN_PIXEL_MOVING,
                        help="a column whose max pixel displacement from "
                             "its first-appear stem is at most this many "
                             "pixels (at the tracking resolution) is "
                             "static and dropped by --filter-static-pixel "
                             "(the criterion is strict; default: "
                             "%(default)s px)")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for the support-grid padding "
                             "(per role: seed + 0/1) (default: %(default)s)")
    parser.add_argument("--skip-done", action="store_true",
                        help="skip prompts whose coords.npy already exists")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _read_prompt_dir(pdir: Path) -> dict:
    """Step-3b prompt output: npz arrays + init_points.json (raises
    FileNotFoundError for a dir without the npz, e.g. an empty prompt)."""
    npz_path = pdir / "init_points.npz"
    with np.load(npz_path) as data:
        keypoints = data["keypoints"].astype(np.float32)   # (K, N, 2)
        frame_indices = np.asarray(data["frame_indices"], dtype=np.int64)
        masks = np.asarray(data["masks"], dtype=bool)      # (N, H, W)
    with open(pdir / "init_points.json") as f:
        meta = json.load(f)
    return {
        "slug": pdir.name,
        "dir": pdir,
        "prompt": meta.get("prompt", pdir.name),
        "camera_key": meta.get("camera_key"),
        "num_keypoints": int(meta.get("num_keypoints", len(keypoints))),
        "empty_reason": meta.get("empty_reason"),
        "keypoints": keypoints,
        "frame_indices": frame_indices,
        "masks": masks,
    }


def _prompt_role(subtask_row: dict | None, prompt_text: str) -> str | None:
    """Role of a Step-3 prompt: the sub-task annotation column ('object' or
    'manipulator') whose value the prompt matches (exact text, then slug);
    None when the annotations have no such row/entry."""
    row = subtask_row or {}
    for role in ROLE_ORDER:
        value = row.get(role)
        if value and (value == prompt_text or _slugify(value) == _slugify(prompt_text)):
            return role
    return None


def _geometry_at(depth_dir, frame_index: int
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(depth (H, W) f32 metres, extrinsics (4, 4), intrinsics (3, 3)) of one
    absolute frame, read from the depth_pose camera folder.

    The pose store keeps the extrinsics as (3, 4) (see
    DepthPoseReader.pose_at); the query helpers invert them, so the
    homogeneous bottom row is appended here — the same (4, 4) the tracking
    batches get from load_npz_batch.
    """
    depth, extr, intrs = load_stream_data(depth_dir, frame_index)
    extr = np.vstack([extr, [0.0, 0.0, 0.0, 1.0]]).astype(np.float32)
    return depth, extr, intrs.astype(np.float32)


def _row_pixels(keypoints: np.ndarray, masks: np.ndarray, j: int,
                depth: np.ndarray, max_rows: int):
    """Rows of one key-frame whose keypoint is usable there.

    A row is usable when its keypoint at key-frame j lies inside the
    key-frame's SAM3 mask (when the mask exists) and on a valid depth
    pixel (depth > 0, keypoints rescaled from the key-frame resolution to
    the saved depth resolution first). Returns (rows, px_keyframe, px_depth)
    with rows in the Step-3 rank order, capped at max_rows."""
    kp = keypoints[:, j]                      # (K, 2) key-frame pixels
    k = len(kp)
    valid = np.ones(k, dtype=bool)
    mask_j = masks[j]
    if mask_j.any():
        x = np.round(kp[:, 0]).astype(np.int64)
        y = np.round(kp[:, 1]).astype(np.int64)
        np.clip(x, 0, mask_j.shape[1] - 1, out=x)
        np.clip(y, 0, mask_j.shape[0] - 1, out=y)
        valid &= mask_j[y, x]
    if not valid.any():
        return np.empty(0, dtype=np.int64), None, None

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
    return rows, kp[rows], px[rows]


def _grid_pixels(world: torch.Tensor, intrs: torch.Tensor,
                 extr: torch.Tensor) -> torch.Tensor:
    """(G, 2) continuous anchor-frame pixels of the unprojected grid
    points: world -> camera with extr, then /z * (fx, fy) + (cx, cy) —
    the exact inverse of the cells' unprojection (get_grid_queries), so
    the recovered pixels stay aligned with the returned rows."""
    h = torch.cat([world, torch.ones_like(world[:, :1])], dim=-1)  # (G, 4)
    cam = (extr @ h.t()).t()[:, :3]                                # (G, 3)
    fx, fy = intrs[0, 0], intrs[1, 1]
    cx, cy = intrs[0, 2], intrs[1, 2]
    z = cam[:, 2]
    return torch.stack([cam[:, 0] / z * fx + cx,
                        cam[:, 1] / z * fy + cy], dim=-1)


def _drop_cells_near_px(grid_px: np.ndarray, query_px: np.ndarray,
                        drop_budget: int) -> np.ndarray:
    """Grid cells to drop so the surviving support spreads away from the
    tracked points.

    Cyclically, each query removes its closest not-yet-dropped grid cell
    (anchor-frame pixel distance, squared Euclidean) until drop_budget
    cells are gone — the even per-query split of the budget (a query
    whose neighbours were already claimed falls back to its next
    closest). Deterministic: cells are claimed in query rank order;
    among equal distances the lowest-index cell wins.

    Args:
        grid_px: (G, 2) continuous grid-cell pixels, in grid row order.
        query_px: (Q, 2) keypoint pixels in the same frame.
        drop_budget: number of cells to drop (<= G).

    Returns:
        (G,) bool mask, True = drop the cell.
    """
    drop = np.zeros(grid_px.shape[0], dtype=bool)
    if drop_budget <= 0:
        return drop
    q = query_px.shape[0]
    d2 = ((grid_px[:, None, :] - query_px[None, :, :]) ** 2).sum(-1)  # (G, Q)
    for i in range(drop_budget):
        dist = d2[:, i % q].copy()
        dist[drop] = np.inf
        drop[int(np.argmin(dist))] = True
    return drop


def _support_queries(depth: np.ndarray, intrs: np.ndarray, extr: np.ndarray,
                     n_points: int, rng: np.random.Generator,
                     query_px: np.ndarray | None = None) -> torch.Tensor:
    """n_points support queries at the anchor frame: the full-frame
    SUPPORT_GRID_SIZE^2 grid (valid-depth pixels only), trimmed to
    n_points; a shortfall is padded with uniformly sampled valid-depth
    pixels. Returns (n_points, 4) with home frame 0.

    With query_px ((Q, 2), the role keypoints' anchor-frame pixels at the
    depth resolution — the same space the grid cells are sampled in) the
    trim is query-aware: the grid cells nearest the keypoints are dropped
    (see _drop_cells_near_px), so the surviving support spreads away from
    the tracked points instead of sampling the same surfaces. Without
    query_px the grid is trimmed by an even linspace spread.
    """
    dh, dw = depth.shape
    depth_t = torch.from_numpy(depth).float()
    intrs_t = torch.from_numpy(intrs).float()
    extr_t = torch.from_numpy(extr).float()
    grid = get_grid_queries(SUPPORT_GRID_SIZE, depth_t[None], intrs_t[None],
                            extr_t[None]).squeeze(0)          # (G, 4)
    n = grid.shape[0]
    if n > n_points and query_px is not None:
        # a pass with more role keypoints than the graph's 64 object-query
        # slots must shed support cells 1-for-1; drop those nearest the
        # keypoints and keep the rest in grid order
        drop = _drop_cells_near_px(
            _grid_pixels(grid[:, 1:], intrs_t, extr_t).numpy(),
            np.asarray(query_px, dtype=np.float32), int(n - n_points))
        return grid[~drop].contiguous()
    if n >= n_points:
        # even spread fallback (no query pixels)
        pick = np.linspace(0, n - 1, n_points).round().astype(int)
        return grid[pick].contiguous()
    parts = [grid]
    ys, xs = np.nonzero(depth > 0)
    if xs.size == 0:
        raise ValueError("anchor depth has no valid pixels (> 0)")
    idx = rng.choice(xs.size, size=n_points - n, replace=False)
    xy = np.stack([xs[idx], ys[idx]], axis=-1).astype(np.float32)
    pad = unproject_xy_queries(xy, depth, intrs, extr)
    if pad is None or pad.shape[0] < n_points - n:
        raise ValueError("could not sample enough valid-depth support pixels")
    return torch.cat(parts + [pad])


class SubtaskTraceExtract(DataExtract):
    """Per-sub-task TAPIP3D tracking over the Step-2/Step-3 results.

    Reuses DataExtract for dataset introspection and camera-key naming;
    the per-sub-task geometry comes from the saved depth_pose folders and
    the RGB frames are decoded online from the LeRobotDataset, one window
    at a time (no frames on disk).
    """

    def __init__(self, args):
        args.mode = "videos"  # DataExtract needs one of its modes; only the
                             # dataset/camera machinery is reused
        args.one_per_task = False  # DataExtract's episode picker reads this
                                   # flag; Step 4 selects episodes via
                                   # --episode-idxes (see run())
        if args.camera_idxes is None:
            keys = LeRobotDatasetMetadata(repo_id=args.repo_id,
                                          root=args.data_root).camera_keys
            # every camera stays selectable: the tracked cameras of each
            # sub-task are the ones with Step-3 init-points subtrees
            args.camera_idxes = list(range(len(keys)))
        super().__init__(args)
        # DataExtract already resolved the episodes root (<data-root>/episodes,
        # or --out-dir) into self.out_dir; reuse that inherited root rather
        # than deriving it a second time. Every path this tool reads or
        # writes is built from it with the astribot_paths helpers: the Step-3
        # inputs (…/subtask_{k:02d}/sampling_points/{detections,init_points}/),
        # the Step-2 depth_pose store and the Step-4 traces all live under it.
        self.episodes_root = self.out_dir
        #: {subtask_index: annotation row}; role resolution degrades to
        #: unlabelled prompts (anchored like the object) when missing.
        try:
            self.subtask_meta = load_subtask_meta(args.data_root)
        except FileNotFoundError:
            self.subtask_meta = {}
        self._pt2 = None
        # per-camera state, set by _process_camera()
        self.ep_idx = self.k = self.cam_key = None
        self.cam_subdir = None
        self.seg_depth_dir = None
        self.seg_stems: list[int] = []
        self.args = args

    # --- model setup --------------------------------------------------------

    def _ensure_pt2(self) -> Tapip3D_PT2:
        if self._pt2 is None:
            print("Loading TAPIP3D .pt2 artifacts...")
            # the shipped artifacts and their fixed graph config (encoder
            # 480x640, 1088-query iteration graph, 6 window iterations) —
            # see _DEFAULT_ENCODER / _DEFAULT_ITERATION in tapip3d.py
            self._pt2 = Tapip3D_PT2()
            # the graph total is fixed at the export-time split (32x32
            # support grid + 64 object-query slots); a pass splits the
            # total between its role keypoints and the support block, so
            # the role caps (64/128) never change the total
            expected = SUPPORT_GRID_SIZE ** 2 + ROLE_MAX_KEYPOINTS["object"]
            if self._pt2.num_queries != expected:
                raise SystemExit(
                    f"iteration graph has {self._pt2.num_queries} fixed "
                    f"queries, expected {expected} "
                    f"({SUPPORT_GRID_SIZE}x{SUPPORT_GRID_SIZE} support grid "
                    f"+ {ROLE_MAX_KEYPOINTS['object']} object slots) — "
                    f"re-export the iteration program for this query count")
        return self._pt2

    # --- dataset access -----------------------------------------------------

    def _cam_key_for_subdir(self, subdir: str) -> str:
        """Dataset camera key whose subdir (key_frames naming) matches a
        Step-3 camera_key."""
        for key in self.cam_keys.values():
            if self._camera_subdir(key) == subdir:
                return key
        raise ValueError(
            f"camera {subdir!r} of the init points has no dataset camera "
            f"among {sorted({self._camera_subdir(k) for k in self.cam_keys.values()})}")

    def _detection_label(self, ep_idx: int, k: int, cam: str) -> int | None:
        """Canonical subtask label Step 3a recorded for one (episode,
        sub-task, camera) in its own detections JSON
        (…/subtask_{k:02d}/sampling_points/detections/<camera>.json, whose
        payload carries just that sub-task) — the label resolves the
        sub-task annotation row of the segment's role split. None when the
        JSON is missing or carries no label (folder-mode payloads): the
        prompts then degrade to unlabelled tracking, never to
        segment-ordinal rows."""
        path = ap.detections_dir(self.episodes_root, ep_idx, k) / f"{cam}.json"
        if not path.is_file():
            return None
        label = json.loads(path.read_text()).get("subtask_index")
        return int(label) if label is not None else None

    def _frames_u8(self, steps: list[int]) -> torch.Tensor:
        """(T, 3, H, W) uint8 CHW frames of the tracked camera, decoded
        online from the dataset at the absolute step indices."""
        key = self.cam_key
        ds = self._ensure_dataset()
        return torch.stack(
            [to_image_tensor(np.asarray(to_pil(ds[t][key]), dtype=np.uint8))
             for t in steps])

    # --- anchor / query assembly -------------------------------------------

    def _candidate_frames(self, prompts: list[dict],
                          window: list[int]) -> list[int]:
        """Chronological anchor candidates of one role pass inside its trace
        window: the window's first stem, then the prompts' key-frames that
        are stems of the window."""
        cands = [window[0]]
        for t in sorted({int(i) for p in prompts
                         for i in p["frame_indices"]}):
            if t in window and t != cands[0]:
                cands.append(t)
        return cands

    def _usable_at(self, prompt: dict, kf_abs: int, depth: np.ndarray,
                   max_rows: int):
        """(rows, px_keyframe, px_depth) of a prompt's keypoints usable on
        the Step-2 stem kf_abs: the prompt's own key-frame column when the
        stem is one of its key-frames, else the first column's keypoints
        tested against kf_abs's depth — valid for the window's leading stem
        (at-or-before the first key-frame), where the object is still
        static. Rows are capped at max_rows (the role's per-prompt cap).
        Empty rows when nothing is usable there."""
        kfs = list(prompt["frame_indices"])
        j = kfs.index(kf_abs) if kf_abs in kfs else 0
        return _row_pixels(prompt["keypoints"], prompt["masks"], j, depth,
                           max_rows)

    # --- orchestration ------------------------------------------------------

    def _init_cameras(self, ep_idx: int, subtask_k: int) -> list[str]:
        """Camera subdirs of one (episode, sub-task) Step-3b init-points
        folder — the cameras that sub-task tracks (each owns its own
        …/subtask_{k:02d}/sampling_points/init_points/<camera>/ subtree);
        [] when Step 3b wrote none (or the folder is missing)."""
        path = ap.init_points_dir(self.episodes_root, ep_idx, subtask_k)
        if not path.is_dir():
            return []
        return sorted(p.name for p in path.iterdir() if p.is_dir())

    def _tracked_episodes(self) -> list[int]:
        """Episodes with Step-3b init points on disk, sorted."""
        return [ep for ep in ap.discover_episodes(self.episodes_root)
                if any(self._init_cameras(ep, k) for k in
                       ap.discover_subtasks(self.episodes_root, ep))]

    def run(self) -> None:
        root = self.episodes_root
        if not root.is_dir():
            raise FileNotFoundError(
                f"need the episodes root ({root}) with the Step-3 init "
                f"points and the Step-2 depth + pose: run "
                f"run_step3_init_points.py and run_step2_depth_stream.py "
                f"first")
        discovered = self._tracked_episodes()
        eps = [e for e in self.ep_idxes if e in discovered]
        if self.args.episode_idxes is not None:
            missing = sorted(set(self.ep_idxes) - set(discovered))
            if missing:
                print(f"skip episode(s) {missing}: no Step-3 init points on disk")
        print(f"\n{len(eps)} episode(s) selected: {eps}")
        for ep_idx in tqdm(eps, desc="episodes"):
            self._process_episode(ep_idx)
        print(f"\ndone: {len(eps)} episode(s) -> {root}")

    def _process_episode(self, ep_idx: int) -> None:
        self.ep_idx = ep_idx
        tid = self._episode_task_id(ep_idx)
        print(f"\nepisode {ep_idx}: task {tid} "
              f"({self._task_description(tid)})")
        subtasks = [k for k in ap.discover_subtasks(self.episodes_root, ep_idx)
                    if self._init_cameras(ep_idx, k)]
        for k in subtasks:
            self._process_segment(k)

    def _process_segment(self, k: int) -> None:
        """Track every camera of the sub-task: each camera owns an
        init-points subtree (…/subtask_{k:02d}/sampling_points/init_points/
        <camera>/, one per camera from Step 3b) and is tracked over its own
        Step-2 depth_pose/<camera> store into the sub-task's
        traces/<camera>/."""
        self.k = k
        for cam in self._init_cameras(self.ep_idx, k):
            self._process_camera(cam)

    def _process_camera(self, cam: str) -> None:
        """TAPIP3D tracking of one (sub-task, camera): prompts come from the
        camera's init-points subtree, the geometry from the same camera's
        Step-2 depth_pose/<camera> store, the role labels from that camera's
        Step-3a detections JSON."""
        k = self.k
        seg_init = ap.init_points_dir(self.episodes_root, self.ep_idx, k, cam)
        prompts = []
        for pdir in sorted(seg_init.iterdir()):
            if not pdir.is_dir() or not (pdir / "init_points.npz").is_file():
                continue
            try:
                prompt = _read_prompt_dir(pdir)
            except FileNotFoundError as e:
                print(f"  [subtask {k:02d}] {cam}: skip {pdir.name}: {e}")
                continue
            if prompt["num_keypoints"] <= 0:
                print(f"  [subtask {k:02d}] {cam}: skip {pdir.name}: no "
                      f"Step-3 keypoints ({prompt['empty_reason'] or 'empty'})")
                continue
            prompts.append(prompt)
        if not prompts:
            print(f"  [subtask {k:02d}] {cam}: skip, no usable Step-3 "
                  f"prompts")
            return

        self.cam_subdir = cam
        try:
            self.cam_key = self._cam_key_for_subdir(cam)
        except ValueError as e:
            print(f"  [subtask {k:02d}] {cam}: skip, {e}")
            return
        label = self._detection_label(self.ep_idx, k, cam)
        print(f"  [subtask {k:02d}] camera {cam}")

        self.seg_depth_dir = ap.depth_pose_dir(self.episodes_root,
                                               self.ep_idx, k, cam)
        if not DepthPoseReader.is_complete(self.seg_depth_dir):
            print(f"  [subtask {k:02d}] {cam}: skip, no Step-2 outputs "
                  f"under {self.seg_depth_dir} (run "
                  f"run_step2_depth_stream.py for this camera)")
            return
        with DepthPoseReader(self.seg_depth_dir) as reader:
            # the tracked stems are the container's absolute frame indices
            self.seg_stems = [int(i) for i in reader.frame_indices]
        if not self.seg_stems:
            print(f"  [subtask {k:02d}] {cam}: skip, no Step-2 steps in "
                  f"{self.seg_depth_dir}")
            return

        if label is None:
            print(f"  [subtask {k:02d}] {cam}: no ground-truth label "
                  f"recorded for it in the Step-3a detections JSON — "
                  f"tracking its prompts unlabelled (re-run Step 3 to "
                  f"regenerate the JSON with labels)")
        row = self.subtask_meta.get(label) if label is not None else None
        roles: dict[str | None, list[dict]] = {}
        for p in prompts:
            role = _prompt_role(row, p["prompt"])
            if role is None:
                print(f"    [{p['slug']}] warning: prompt {p['prompt']!r} "
                      f"matches no object/manipulator entry of sub-task {k}; "
                      f"tracking it in the unlabelled pass")
            roles.setdefault(role, []).append(p)
        # object and manipulator first, then any unlabelled prompts. The
        # outcomes are only counted for the report below: Step 4 writes no
        # camera-level metadata roll-up, the per-prompt metadata.json beside
        # every coords.npy is the record of what happened (and what
        # visualize_step4_traces.py reads).
        written = {"prompts": 0, "passes": 0}
        order = [r for r in list(ROLE_ORDER) + [None] if r in roles]
        for role in order:
            self._process_role(role, roles[role], written)
        if not written["prompts"] and not written["passes"]:
            if self.args.skip_done:
                print(f"  [subtask {k:02d}] {cam}: all prompts already "
                      f"tracked (--skip-done)")
            else:
                print(f"  [subtask {k:02d}] {cam}: skip, nothing to track")

    def _process_role(self, role: str | None, prompts: list[dict],
                      written: dict) -> None:
        """One TAPIP3D pass for a role: pick the anchor key-frame, assemble
        the exact-N queries and track the role's keypoints over the trace
        window's stems from the anchor on. Counts the per-prompt outcomes
        into ``written``."""
        k = self.k
        label = role or "unlabelled"
        cam_dir = ap.traces_dir(self.episodes_root, self.ep_idx, k,
                                self.cam_subdir)
        pdirs = [str(cam_dir / p["slug"]) for p in prompts]
        if self.args.skip_done and all(
                (Path(d) / "coords.npy").is_file() for d in pdirs):
            print(f"  [subtask {k:02d}] {label} pass: skip (outputs exist)")
            return
        if self.args.skip_done:
            for p, d in zip(prompts, pdirs):
                if (Path(d) / "coords.npy").is_file():
                    print(f"    [{p['slug']}] skip: {d}/coords.npy exists")
            prompts = [p for p, d in zip(prompts, pdirs)
                       if not (Path(d) / "coords.npy").is_file()]
            pdirs = [d for d in pdirs if not (Path(d) / "coords.npy").is_file()]

        # --- trace window + anchor: first candidate with usable rows --------
        # The pass traces the stems inside its prompts' key-frame envelope
        # (span_stems): the object's [close .. open] transport gets the stem
        # right before the close through the stem right after the open; a
        # full-span prompt keeps the whole sub-task (as before).
        window = span_stems(
            self.seg_stems,
            [int(p["frame_indices"][0]) for p in prompts],
            [int(p["frame_indices"][-1]) for p in prompts])
        cap = _role_keypoint_cap(role)
        depth = None
        anchor_abs = None
        usable: dict[str, tuple] = {}
        for cand in self._candidate_frames(prompts, window):
            depth, extr, intrs = _geometry_at(self.seg_depth_dir, cand)
            for p in prompts:
                usable[p["slug"]] = self._usable_at(p, cand, depth, cap)
            if any(len(u[0]) for u in usable.values()):
                anchor_abs = cand
                break
        if anchor_abs is None:
            reason = "no usable keypoints on the pass's candidate frames " \
                     f"(window {window[0]}..{window[-1]})"
            print(f"  [subtask {k:02d}] {label} pass: empty ({reason})")
            for p, d in zip(prompts, pdirs):
                self._save_empty(p, role, d, reason, written)
            return

        # --- object queries: rows per prompt, rank order, <= role cap ---------
        per_prompt = []          # (prompt, rows, px_keyframe, px_depth)
        n_obj = 0
        for p, d in zip(prompts, pdirs):
            rows, px_kf, px_dep = usable[p["slug"]]
            n_here = len(rows)
            if n_here and n_obj < cap:
                n_here = min(n_here, cap - n_obj)
                rows, px_kf, px_dep = rows[:n_here], px_kf[:n_here], px_dep[:n_here]
                n_obj += n_here
            else:
                rows, px_kf, px_dep = None, None, None
            per_prompt.append((p, d, rows, px_kf, px_dep))
        n_obj = sum(len(r[2]) if r[2] is not None else 0 for r in per_prompt)
        if n_obj == 0:
            reason = f"pass query cap of {cap} reached by the earlier prompts"
            print(f"  [subtask {k:02d}] {label} pass: empty ({reason})")
            for p, d in zip(prompts, pdirs):
                self._save_empty(p, role, d, reason, written)
            return

        # --- exact-N query assembly ------------------------------------------
        xy_blocks = [r[4] for r in per_prompt if r[4] is not None]
        xy = np.concatenate(xy_blocks)     # keypoint px on the anchor depth
        init = unproject_xy_queries(xy, depth, intrs, extr)
        assert init is not None, "anchor rows already depth-filtered"
        need = self._ensure_pt2().num_queries - init.shape[0]
        pass_seed = self.args.seed + (ROLE_ORDER.index(role)
                                      if role in ROLE_ORDER else 2)
        rng = np.random.default_rng(pass_seed)
        support = _support_queries(depth, intrs, extr, need, rng,
                                   query_px=xy)
        queries = torch.cat([init, support])
        if queries.shape[0] != self._pt2.num_queries:
            raise SystemExit(
                f"assembled {queries.shape[0]} queries, expected the "
                f"iteration graph's {self._pt2.num_queries}")
        print(f"  [subtask {k:02d}] {label} pass: {n_obj} role keypoints "
              f"+ {need} support queries, anchor {anchor_abs} "
              f"(seed {pass_seed})")

        # --- sequence: the trace window's stems from the anchor on ------------
        steps = window[window.index(anchor_abs):]
        if len(steps) < self._pt2.seq_len:
            print(f"    warning: {len(steps)} steps < the {self._pt2.seq_len}-"
                  f"frame window; the trace will stay at the anchor points "
                  f"(all visibs false)")
        t0 = time.perf_counter()
        tracked = self._track(steps, queries)
        print(f"    tracked {len(steps)} steps in "
              f"{time.perf_counter() - t0:.1f}s")
        if (self.args.filter_visible or self.args.filter_static_pixel):
            # The stream filtered inside run(): coords/visibs hold the
            # surviving columns only (visibs already bool at
            # --vis-threshold); keep_t (full-width, original column order)
            # and drop_reasons map the survivors back to the prompts.
            coords, visibs, keep_t, drop_reasons = tracked
            visibs = visibs.numpy()
        else:
            coords, visibs_logits = tracked
            visibs = (torch.sigmoid(visibs_logits) >=
                      self.args.vis_threshold).numpy()
            keep_t = None
            drop_reasons = None

        # --- save: one folder per prompt (its own query columns only) --------
        # Prompt blocks are contiguous in column order both in the full
        # layout and — survivors only — in the filtered arrays, so the
        # prompt cursor (col) walks keep_t while a kept-column cursor
        # (kept_col) walks the filtered coords/visibs. Filtered runs cap
        # each prompt's output at MAX_KEPT_KEYPOINTS survivors (a 128-seed
        # manipulator prompt can outlive the filters with more).
        col = 0
        kept_col = 0
        written["passes"] += 1
        for (p, d, rows, px_kf, px_dep) in per_prompt:
            n = len(rows) if rows is not None else 0
            start = col
            col += n
            if n == 0:
                self._save_empty(p, role, d,
                                 "no usable keypoints on the pass anchor "
                                 f"key-frame {anchor_abs}", written)
                continue
            rows = np.asarray(rows)
            px_kf = np.asarray(px_kf)
            if keep_t is not None:
                # run() returns CPU tensors (see Tapip3DStreamPT2.run), so
                # the column mask and coords slice on cpu directly
                sel_t = keep_t[start:col]                # (n,) bool
                sel = sel_t.numpy()                      # (n,) survivors
                sel, over = _cap_keypoint_survivors(sel, MAX_KEPT_KEYPOINTS)
                for i in over:
                    drop_reasons[start + int(i)] = (
                        f"beyond the {MAX_KEPT_KEYPOINTS}-keypoint output "
                        f"cap after filtering")
                n = int(sel.sum())
                coords_save = coords[:, kept_col:kept_col + n].numpy()
                visibs_save = visibs[:, kept_col:kept_col + n]
                queries_save = queries[start:col][sel_t].numpy()
                kept_col += n
                rows_keep = rows[sel]
                px_keep = px_kf[sel]
            else:
                coords_save = coords[:, start:col].numpy()
                visibs_save = visibs[:, start:col]
                queries_save = queries[start:col].numpy()
                rows_keep, px_keep = rows, px_kf
            out_dir = Path(d)
            out_dir.mkdir(parents=True, exist_ok=True)
            np.save(out_dir / "coords.npy", coords_save)
            np.save(out_dir / "visibs.npy", visibs_save)
            np.save(out_dir / "queries.npy", queries_save)
            entry = {
                "episode": int(self.ep_idx), "subtask": int(k),
                "role": role, "prompt": p["prompt"],
                "prompt_slug": p["slug"],
                "camera_key": self.cam_subdir,
                "status": "ok",
                "anchor_frame": int(anchor_abs),
                "steps": [int(s) for s in steps],
                "num_steps": int(len(steps)),
                "num_queries": int(n),
                "pass_queries": int(queries.shape[0]),
                "query_keypoint_rows":
                    [int(i) for i in rows_keep],
                "pixels": [[float(x), float(y)] for x, y in px_keep],
                "model": {"encoder": str(Path(_DEFAULT_ENCODER).absolute()),
                          "iteration":
                              str(Path(_DEFAULT_ITERATION).absolute()),
                          "num_iters": self._pt2.num_iters},
                "image_size": list(self._pt2.image_size),
                "vis_threshold": self.args.vis_threshold,
                "seed": self.args.seed,
                "inputs": {"init_points_dir": str(p["dir"]),
                           "depth_dir": str(self.seg_depth_dir)},
            }
            if keep_t is not None:
                dropped = np.nonzero(~sel)[0]
                entry["filter"] = {
                    "num_kept": int(n),
                    "num_dropped": int(len(dropped)),
                    "dropped_keypoint_rows": [int(i) for i in rows[dropped]],
                    "dropped_reasons": [drop_reasons[start + int(i)]
                                        for i in dropped],
                }
                if self.args.filter_visible:
                    entry["filter"]["visible"] = {
                        "max_invisible_stems": self.args.max_invisible_stems,
                        "max_reappear_ratio":
                            self.args.max_reappear_ratio,
                    }
                if self.args.filter_static_pixel:
                    entry["filter"]["static_pixel"] = {
                        "min_motion_pixels": self.args.min_motion_pixels,
                    }
            with open(out_dir / "metadata.json", "w") as f:
                json.dump(entry, f, indent=2)
            print(f"    [{p['slug']}] {n} keypoints -> {d}")
            written["prompts"] += 1

    def _save_empty(self, prompt: dict, role: str | None, out_dir: str,
                    reason: str, written: dict) -> None:
        """Prompt metadata only (status empty) — mirrors the Step-3b
        init_points.json convention."""
        entry = {
            "episode": int(self.ep_idx), "subtask": int(self.k),
            "role": role, "prompt": prompt["prompt"],
            "prompt_slug": prompt["slug"],
            "camera_key": self.cam_subdir,
            "status": "empty", "empty_reason": reason,
            "inputs": {"init_points_dir": str(prompt["dir"]),
                       "depth_dir": str(self.seg_depth_dir)},
        }
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(out_dir) / "metadata.json", "w") as f:
            json.dump(entry, f, indent=2)
        written["prompts"] += 1

    # --- tracking ------------------------------------------------------------

    def _track(self, steps: list[int], queries: torch.Tensor):
        """Streamed TAPIP3D over the sequence steps: batches of seq_len
        frames decoded online + Step-2 geometry resized to the inference
        resolution (the same math as load_resized_batch). With all
        filters off returns (coords (T, Q, 3), visibs_logits (T, Q));
        with --filter-visible and/or --filter-static-pixel on, returns
        the stream's filtered 4-tuple (coords, visibs bool, keep (Q,)
        bool, reasons) instead. All returned tensors are CPU — the
        stream moves its outputs back from the GPU."""
        pt2 = self._ensure_pt2()
        inf_h, inf_w = pt2.image_size
        steps = [int(t) for t in steps]  # absolute container frame indices

        def batches():
            # both legs of a batch walk the same traced steps: the frames
            # decoded online from the dataset and the geometry read by
            # absolute frame index from the depth_pose camera folder
            for s in range(0, len(steps), pt2.seq_len):
                end = min(s + pt2.seq_len, len(steps))
                video = self._frames_u8(steps[s:end])   # (T, 3, H0, W0)
                geo = load_npz_batch(str(self.seg_depth_dir), steps, s, end)
                yield resize_batch_to_inference(video, geo, inf_h, inf_w)

        depth_roi = compute_global_depth_roi(str(self.seg_depth_dir), steps,
                                             inf_h, inf_w)
        si = Tapip3DStreamPT2(pt2, queries, depth_roi=depth_roi,
                              vis_threshold=self.args.vis_threshold,
                              max_invisible_stems=self.args.max_invisible_stems,
                              max_reappear_ratio=self.args.max_reappear_ratio,
                              min_motion_pixels=self.args.min_motion_pixels)
        with torch.inference_mode():
            out = si.run(batches(), len(steps),
                         filter_visible=self.args.filter_visible,
                         filter_static_pixel=self.args.filter_static_pixel)
        return out


def main() -> None:
    args = parse_args()
    # the TAPIP3D path runs validated numerics (fp32, TF32 off) — same
    # convention as infer_tapip3d.py
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    SubtaskTraceExtract(args).run()


if __name__ == "__main__":
    main()
