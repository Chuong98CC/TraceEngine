"""Step 4 online — per-sub-task 3D point tracking with the TAPIP3D
torch.export programs, straight from the Step-2/Step-3 results and the
LeRobotDataset (nothing extracted to disk).

For every sub-task of the selected episodes it loads the per-prompt init
points of Step 3b (SAM3 masks + RoMAv2 keypoints under the Step-3
sampling_points root — <data-root>/eps_data/sampling_points/init_points),
anchors them on the Step-2 depth + pose outputs (<out>/depth_pose/), and
tracks the 3D positions of the points with TAPIP3D over the sub-task's
streamed frames — the RGB frames are decoded **online** from the dataset,
the geometry comes from the saved depth_pose npz/lz4 files:

    Step-2 stems + Step-3 keypoints (per prompt)
               │
               ▼  one TAPIP3D pass per role (object / manipulator)
    ┌──────────────────────────────┐
    │  coords + visibs per prompt  │
    └──────────────────────────────┘

Each role pass traces its prompts over the Step-2 stems inside a window
bounded by the prompts' Step-3 key-frames (span_stems): from the last
stem at-or-before the earliest first key-frame to the first stem
at-or-after the latest last key-frame.

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
each pass tracks up to 64 role keypoints + a full-frame support grid
trimmed/padded deterministically to reach exactly 1088 (random padding
points are sampled among the anchor frame's valid-depth pixels,
np.random.default_rng(seed + role_index)).

Every camera of a sub-task is tracked separately over its own Step-2
depth_<camera> outputs — the cameras come from the per-camera init-points
subtrees of Step 3b (init_points/<episode>/subtask_XX/<camera>/), the
role labels from that camera's Step-3a detections JSON — the Step-3
inputs are read from the sampling_points root, while --out-dir holds only
the depth_pose read and the traces write. Output, per camera under
<out>/traces/<episode>/subtask_XX/<camera>/ and per prompt
under <prompt_slug>/: coords.npy (T, Q, 3) world-space traces, visibs.npy
(T, Q) visibility flags, queries.npy (Q, 4) query points (home frame, x,
y, z) and metadata.json — plus a camera-level metadata.json summarizing
the roles/passes.
Visualization: tools/astribot/visualize_step4_traces.py renders
per-camera videos of the traces.

Examples
--------
    # Track every sub-task of episode 0 (object + manipulator roles)
    python tools/astribot/run_step4_traces.py
        --repo-id Kronze157/astri_making_coffee_vlva
        --data-root /data/astri_making_coffee --episode-idxes 0
"""

from __future__ import annotations

import argparse
import json
import os
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
from utils.depth_utils import load_depth_lz4
from utils.file_io.image_io import to_image_tensor
from utils.keyframe_utils import load_subtask_meta, sampling_points_root, span_stems
from utils.streaming_utils import (
    compute_global_depth_roi,
    load_npz_batch,
    resize_batch_to_inference,
    unproject_xy_queries,
)
from utils.visualize.visualize_mask import to_pil

#: role output order of the sub-task annotations (meta/subtasks.csv columns).
ROLE_ORDER = ("object", "manipulator")
#: max tracked role keypoints per pass (the shipped iteration graph is
#: 64 object-query slots + 32x32 = 1024 support-grid slots = 1088 total).
MAX_OBJECT_QUERIES = 64
SUPPORT_GRID_SIZE = 32

_EP_RE = re.compile(r"^ep(\d{6})$")
_SUB_RE = re.compile(r"^subtask_(\d+)$")


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
                        help="output root (default: <data-root>/eps_data); "
                             "Step-2 results are read under "
                             "<out-dir>/depth_pose, traces are saved under "
                             "<out-dir>/traces; the Step-3 inputs are read "
                             "from the sampling_points root "
                             "(<data-root>/eps_data/sampling_points/"
                             "{detections,init_points})")
    parser.add_argument("--vis-threshold", type=float, default=0.5,
                        help="sigmoid visibility threshold for visibs "
                             "(default: %(default)s)")
    parser.add_argument("--filter-visible", action="store_true",
                        help="drop the tracked columns that are not always "
                             "visible, or whose invisible runs are long or "
                             "reappear far from where they were last "
                             "visible (see --max-invisible-stems / "
                             "--max-reappear-displacement)")
    parser.add_argument("--filter-static", action="store_true",
                        help="additionally drop the columns that never "
                             "move: their max displacement from the first "
                             "visible (first-appear) stem to any later "
                             "visible stem is at most "
                             "--min-motion-displacement. Applied after "
                             "--filter-visible when both are on")
    parser.add_argument("--max-invisible-stems", type=int, default=4,
                        help="an invisible run of this many consecutive "
                             "trace stems or more drops the column (the "
                             "threshold is strict; default: %(default)s)")
    parser.add_argument("--max-reappear-displacement", type=float,
                        default=0.01,
                        help="max 3D displacement in metres between the "
                             "last visible stem and the reappearance of a "
                             "bounded invisible run (default: %(default)s)")
    parser.add_argument("--min-motion-displacement", type=float,
                        default=0.02,
                        help="a column whose max displacement from its "
                             "first-appear stem is at most this many "
                             "metres is static and dropped by "
                             "--filter-static (the criterion is strict; "
                             "default: %(default)s m)")
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


def _geometry_at(depth_dir: Path, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Saved Step-2 geometry of one step: (depth (H, W) float32 metres,
    intrinsics (3, 3), extrinsics padded to (4, 4)) — the same pair the
    npz/lz4 hold (see load_stream_data / load_npz_batch)."""
    lz4_path = depth_dir / f"frame_{idx:06d}.lz4"
    pose_path = depth_dir / f"frame_{idx:06d}.npz"
    if not lz4_path.is_file() or not pose_path.is_file():
        raise FileNotFoundError(f"missing depth/pose pair: {lz4_path} / "
                                f"{pose_path}")
    with np.load(pose_path) as data:
        depth = load_depth_lz4(lz4_path, tuple(int(v) for v in data["shape"]))
        extr = data["extrinsics"] if "extrinsics" in data else data["extrinsic"]
        if extr.shape == (3, 4):
            extr = np.vstack([extr, [0.0, 0.0, 0.0, 1.0]])
        intrs = data["intrinsics"] if "intrinsics" in data else data["intrinsic"]
    return (depth.astype(np.float32), intrs.astype(np.float32),
            extr.astype(np.float32))


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


def _support_queries(depth: np.ndarray, intrs: np.ndarray, extr: np.ndarray,
                     n_points: int, rng: np.random.Generator) -> torch.Tensor:
    """n_points support queries at the anchor frame: the full-frame
    SUPPORT_GRID_SIZE^2 grid (valid-depth pixels only), trimmed to
    n_points; a shortfall is padded with uniformly sampled valid-depth
    pixels. Returns (n_points, 4) with home frame 0."""
    dh, dw = depth.shape
    depth_t = torch.from_numpy(depth).float()
    intrs_t = torch.from_numpy(intrs).float()
    extr_t = torch.from_numpy(extr).float()
    grid = get_grid_queries(SUPPORT_GRID_SIZE, depth_t[None], intrs_t[None],
                            extr_t[None]).squeeze(0)          # (G, 4)
    if grid.shape[0] >= n_points:
        # even spread when the grid outgrows its share (never with the
        # shipped 1088 graph: 1024 grid slots >= 1088 - 64)
        pick = np.linspace(0, grid.shape[0] - 1, n_points).round().astype(int)
        return grid[pick].contiguous()
    parts = [grid]
    ys, xs = np.nonzero(depth > 0)
    if xs.size == 0:
        raise ValueError("anchor depth has no valid pixels (> 0)")
    idx = rng.choice(xs.size, size=n_points - grid.shape[0], replace=False)
    xy = np.stack([xs[idx], ys[idx]], axis=-1).astype(np.float32)
    pad = unproject_xy_queries(xy, depth, intrs, extr)
    if pad is None or pad.shape[0] < n_points - grid.shape[0]:
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
        # Step-3 inputs live under the Step-3 sampling_points root (see
        # sampling_points_root) — fixed, not relocated by --out-dir, which
        # only locates the Step-2 depth_pose read and the traces write.
        self.det_root = str(sampling_points_root(args.data_root) / "detections")
        self.init_root = str(sampling_points_root(args.data_root) / "init_points")
        self.depth_pose_root = os.path.join(self.out_dir, "depth_pose")
        self.trace_root = os.path.join(self.out_dir, "traces")
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
        self.seg_labels: dict[int, int] = {}  # per (episode, camera)
        self.args = args

    # --- model setup --------------------------------------------------------

    def _ensure_pt2(self) -> Tapip3D_PT2:
        if self._pt2 is None:
            print("Loading TAPIP3D .pt2 artifacts...")
            # the shipped artifacts and their fixed graph config (encoder
            # 480x640, 1088-query iteration graph, 6 window iterations) —
            # see _DEFAULT_ENCODER / _DEFAULT_ITERATION in tapip3d.py
            self._pt2 = Tapip3D_PT2()
            if self._pt2.num_queries != SUPPORT_GRID_SIZE ** 2 + MAX_OBJECT_QUERIES:
                raise SystemExit(
                    f"iteration graph has {self._pt2.num_queries} fixed "
                    f"queries, expected {SUPPORT_GRID_SIZE ** 2 + MAX_OBJECT_QUERIES} "
                    f"({SUPPORT_GRID_SIZE}x{SUPPORT_GRID_SIZE} support grid + "
                    f"{MAX_OBJECT_QUERIES} object slots) — re-export the "
                    f"iteration program for this query count")
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

    def _detection_labels(self, ep_idx: int, cam: str) -> dict[int, int]:
        """{segment ordinal: canonical subtask label} that Step 3a recorded
        in the (episode, camera) detections JSON — the label resolves the
        sub-task annotation row of every segment's role split ({} when the
        JSON is missing or carries no labels: the prompts then degrade to
        unlabelled tracking, never to segment-ordinal rows)."""
        path = (Path(self.det_root) / f"ep{ep_idx:06d}" / f"{cam}.json")
        if not path.is_file():
            return {}
        data = json.loads(path.read_text())
        out = {}
        for k, seg in (data.get("subtasks") or {}).items():
            label = seg.get("subtask_index") if isinstance(seg, dict) else None
            if label is not None:
                out[int(k)] = int(label)
        return out

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

    def _usable_at(self, prompt: dict, kf_abs: int, depth: np.ndarray):
        """(rows, px_keyframe, px_depth) of a prompt's keypoints usable on
        the Step-2 stem kf_abs: the prompt's own key-frame column when the
        stem is one of its key-frames, else the first column's keypoints
        tested against kf_abs's depth — valid for the window's leading stem
        (at-or-before the first key-frame), where the object is still
        static. Empty rows when nothing is usable there."""
        kfs = list(prompt["frame_indices"])
        j = kfs.index(kf_abs) if kf_abs in kfs else 0
        return _row_pixels(prompt["keypoints"], prompt["masks"], j, depth,
                           MAX_OBJECT_QUERIES)

    # --- orchestration ------------------------------------------------------

    def run(self) -> None:
        init_root = Path(self.init_root)
        depth_root = Path(self.depth_pose_root)
        if not init_root.is_dir() or not depth_root.is_dir():
            raise FileNotFoundError(
                f"need Step-3 init points ({init_root}) and Step-2 depth + "
                f"pose ({depth_root}): run run_step3_init_points.py and "
                f"run_step2_depth_stream.py first")
        discovered = sorted(int(m.group(1)) for p in init_root.iterdir()
                            if p.is_dir() and (m := _EP_RE.match(p.name)))
        eps = [e for e in self.ep_idxes if e in discovered]
        if self.args.episode_idxes is not None:
            missing = sorted(set(self.ep_idxes) - set(discovered))
            if missing:
                print(f"skip episode(s) {missing}: no Step-3 init points on disk")
        print(f"\n{len(eps)} episode(s) selected: {eps}")
        for ep_idx in tqdm(eps, desc="episodes"):
            self._process_episode(ep_idx)
        print(f"\ndone: {len(eps)} episode(s) -> {self.trace_root}")

    def _process_episode(self, ep_idx: int) -> None:
        self.ep_idx = ep_idx
        tid = self._episode_task_id(ep_idx)
        print(f"\nepisode {ep_idx}: task {tid} "
              f"({self._task_description(tid)})")
        ep_init = Path(self.init_root) / f"ep{ep_idx:06d}"
        subtasks = sorted(int(m.group(1)) for p in ep_init.iterdir()
                          if p.is_dir() and (m := _SUB_RE.match(p.name))
                          and any(x.is_dir() for x in p.iterdir()))
        for k in subtasks:
            self._process_segment(k)

    def _process_segment(self, k: int) -> None:
        """Track every camera of the sub-task: each camera owns an
        init-points subtree (init_points/.../subtask_XX/<camera>/, one per
        camera from Step 3b) and is tracked over its own Step-2
        depth_<camera> outputs into traces/.../subtask_XX/<camera>/."""
        self.k = k
        seg_init = (Path(self.init_root) / f"ep{self.ep_idx:06d}"
                    / f"subtask_{k:02d}")
        cameras = sorted(p.name for p in seg_init.iterdir() if p.is_dir())
        for cam in cameras:
            self._process_camera(cam)

    def _process_camera(self, cam: str) -> None:
        """TAPIP3D tracking of one (sub-task, camera): prompts come from the
        camera's init-points subtree, the geometry from the same camera's
        Step-2 depth_<camera> folder, the role labels from that camera's
        Step-3a detections JSON."""
        k = self.k
        seg_init = (Path(self.init_root) / f"ep{self.ep_idx:06d}"
                    / f"subtask_{k:02d}" / cam)
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
        self.seg_labels = self._detection_labels(self.ep_idx, cam)
        print(f"  [subtask {k:02d}] camera {cam}")

        seg_depth = (Path(self.depth_pose_root) / f"ep{self.ep_idx:06d}"
                     / f"subtask_{k:02d}")
        self.seg_depth_dir = seg_depth / f"depth_{cam}"
        if not self.seg_depth_dir.is_dir():
            print(f"  [subtask {k:02d}] {cam}: skip, no Step-2 outputs "
                  f"under {self.seg_depth_dir} (run "
                  f"run_step2_depth_stream.py for this camera)")
            return
        self.seg_stems = sorted(int(p.stem.rsplit("_", 1)[-1])
                                for p in self.seg_depth_dir.glob("*.npz"))
        if not self.seg_stems:
            print(f"  [subtask {k:02d}] {cam}: skip, no Step-2 steps in "
                  f"{self.seg_depth_dir}")
            return

        label = self.seg_labels.get(k)
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
        # object and manipulator first, then any unlabelled prompts
        seg_report = {"episode": int(self.ep_idx), "subtask": int(k),
                      "camera_key": cam,
                      "depth_dir": str(self.seg_depth_dir),
                      "prompts": [], "passes": []}
        order = [r for r in list(ROLE_ORDER) + [None] if r in roles]
        for role in order:
            self._process_role(role, roles[role], seg_report)
        if not seg_report["prompts"] and not seg_report["passes"]:
            if self.args.skip_done:
                print(f"  [subtask {k:02d}] {cam}: all prompts already "
                      f"tracked (--skip-done)")
            else:
                print(f"  [subtask {k:02d}] {cam}: skip, nothing to track")
            return
        out_dir = Path(self.trace_root) / f"ep{self.ep_idx:06d}" \
            / f"subtask_{k:02d}" / cam
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "metadata.json", "w") as f:
            json.dump(seg_report, f, indent=2)

    def _process_role(self, role: str | None, prompts: list[dict],
                      seg_report: dict) -> None:
        """One TAPIP3D pass for a role: pick the anchor key-frame, assemble
        the exact-N queries and track the role's keypoints over the trace
        window's stems from the anchor on. Records per-prompt results into
        seg_report."""
        k = self.k
        label = role or "unlabelled"
        pdirs = [str(Path(self.trace_root) / f"ep{self.ep_idx:06d}"
                     / f"subtask_{k:02d}" / self.cam_subdir
                     / p["slug"]) for p in prompts]
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
        depth = None
        anchor_abs = None
        usable: dict[str, tuple] = {}
        for cand in self._candidate_frames(prompts, window):
            depth, intrs, extr = _geometry_at(self.seg_depth_dir, cand)
            for p in prompts:
                usable[p["slug"]] = self._usable_at(p, cand, depth)
            if any(len(u[0]) for u in usable.values()):
                anchor_abs = cand
                break
        if anchor_abs is None:
            reason = "no usable keypoints on the pass's candidate frames " \
                     f"(window {window[0]}..{window[-1]})"
            print(f"  [subtask {k:02d}] {label} pass: empty ({reason})")
            for p, d in zip(prompts, pdirs):
                self._save_empty(p, role, d, reason, seg_report)
            return

        # --- object queries: rows per prompt, rank order, <= 64 total --------
        per_prompt = []          # (prompt, rows, px_keyframe, px_depth)
        n_obj = 0
        for p, d in zip(prompts, pdirs):
            rows, px_kf, px_dep = usable[p["slug"]]
            n_here = len(rows)
            if n_here and n_obj < MAX_OBJECT_QUERIES:
                n_here = min(n_here, MAX_OBJECT_QUERIES - n_obj)
                rows, px_kf, px_dep = rows[:n_here], px_kf[:n_here], px_dep[:n_here]
                n_obj += n_here
            else:
                rows, px_kf, px_dep = None, None, None
            per_prompt.append((p, d, rows, px_kf, px_dep))
        n_obj = sum(len(r[2]) if r[2] is not None else 0 for r in per_prompt)
        if n_obj == 0:
            reason = f"pass query cap of {MAX_OBJECT_QUERIES} reached by " \
                     "the earlier prompts"
            print(f"  [subtask {k:02d}] {label} pass: empty ({reason})")
            for p, d in zip(prompts, pdirs):
                self._save_empty(p, role, d, reason, seg_report)
            return

        # --- exact-N query assembly ------------------------------------------
        xy_blocks = [r[4] for r in per_prompt if r[4] is not None]
        init = unproject_xy_queries(np.concatenate(xy_blocks), depth,
                                    intrs, extr)
        assert init is not None, "anchor rows already depth-filtered"
        need = self._ensure_pt2().num_queries - init.shape[0]
        pass_seed = self.args.seed + (ROLE_ORDER.index(role)
                                      if role in ROLE_ORDER else 2)
        rng = np.random.default_rng(pass_seed)
        support = _support_queries(depth, intrs, extr, need, rng)
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
        if self.args.filter_visible or self.args.filter_static:
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
        # (kept_col) walks the filtered coords/visibs.
        col = 0
        kept_col = 0
        pass_meta = {"role": role, "anchor_frame": int(anchor_abs),
                     "num_steps": int(len(steps)),
                     "num_queries": int(queries.shape[0]),
                     "num_object_queries": int(n_obj)}
        if keep_t is not None:
            pass_meta["num_kept_queries"] = int(keep_t.sum())
        seg_report["passes"].append(pass_meta)
        for (p, d, rows, px_kf, px_dep) in per_prompt:
            n = len(rows) if rows is not None else 0
            start = col
            col += n
            if n == 0:
                self._save_empty(p, role, d,
                                 "no usable keypoints on the pass anchor "
                                 f"key-frame {anchor_abs}", seg_report)
                continue
            rows = np.asarray(rows)
            px_kf = np.asarray(px_kf)
            if keep_t is not None:
                # run() returns CPU tensors (see Tapip3DStreamPT2.run), so
                # the column mask and coords slice on cpu directly
                sel_t = keep_t[start:col]                # (n,) bool
                sel = sel_t.numpy()                      # (n,) survivors
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
                        "max_reappear_displacement":
                            self.args.max_reappear_displacement,
                    }
                if self.args.filter_static:
                    entry["filter"]["static"] = {
                        "min_motion_displacement":
                            self.args.min_motion_displacement,
                    }
            with open(out_dir / "metadata.json", "w") as f:
                json.dump(entry, f, indent=2)
            print(f"    [{p['slug']}] {n} keypoints -> {d}")
            seg_report["prompts"].append(entry)

    def _save_empty(self, prompt: dict, role: str | None, out_dir: str,
                    reason: str, seg_report: dict) -> None:
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
        seg_report["prompts"].append(entry)

    # --- tracking ------------------------------------------------------------

    def _track(self, steps: list[int], queries: torch.Tensor):
        """Streamed TAPIP3D over the sequence steps: batches of seq_len
        frames decoded online + Step-2 geometry resized to the inference
        resolution (the same math as load_resized_batch). With both
        filters off returns (coords (T, Q, 3), visibs_logits (T, Q));
        with --filter-visible and/or --filter-static on, returns the
        stream's filtered 4-tuple (coords, visibs bool, keep (Q,) bool,
        reasons) instead. All returned tensors are CPU — the stream moves
        its outputs back from the GPU."""
        pt2 = self._ensure_pt2()
        inf_h, inf_w = pt2.image_size
        file_list = [(int(t), None) for t in steps]

        def batches():
            for s in range(0, len(steps), pt2.seq_len):
                end = min(s + pt2.seq_len, len(steps))
                video = self._frames_u8(steps[s:end])   # (T, 3, H0, W0)
                geo = load_npz_batch(str(self.seg_depth_dir), file_list, s, end)
                yield resize_batch_to_inference(video, geo, inf_h, inf_w)

        depth_roi = compute_global_depth_roi(str(self.seg_depth_dir),
                                             file_list, inf_h, inf_w)
        si = Tapip3DStreamPT2(pt2, queries, depth_roi=depth_roi,
                              vis_threshold=self.args.vis_threshold,
                              max_invisible_stems=self.args.max_invisible_stems,
                              max_reappear_displacement=(
                                  self.args.max_reappear_displacement),
                              min_motion_displacement=(
                                  self.args.min_motion_displacement))
        with torch.inference_mode():
            out = si.run(batches(), len(steps),
                         filter_visible=self.args.filter_visible,
                         filter_static=self.args.filter_static)
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
