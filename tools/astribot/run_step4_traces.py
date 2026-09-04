"""Step 4 online — per-sub-task 3D point tracking with the TAPIP3D
torch.export programs, straight from the Step-2/Step-3 results and the
LeRobotDataset (nothing extracted to disk).

For every sub-task of the selected episodes it loads the per-prompt init
points of Step 3b (SAM3 masks + RoMAv2 keypoints under <out>/init_points/),
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

The two roles anchor differently:

- the **manipulator** keypoints are tracked from the first frame of the
  sub-task (its first Step-2 stem);
- the **object** keypoints are tracked from the sub-task's first key-frame
  that carries usable object keypoints (leading stems are skipped when the
  sub-task start has none, and the role is skipped when no key-frame of the
  sub-task has any).

A keypoint is *usable* on a key-frame when it is a surviving Step-3
keypoint lying inside that key-frame's SAM3 mask (masks[j].any() missing
-> unconstrained) with valid depth at its pixel. Roles come from the
dataset annotations meta/subtasks.csv ([object, manipulator] of the
sub-task's row): Step 3a recorded the segment's canonical sub-task label
(subtask_index) in the detections JSON, and that label resolves the row —
a segment without a recorded label is tracked unlabelled (anchored like
the object), never role-matched by the segment ordinal.

The shipped TAPIP3D iteration graph has a fixed query count (1088), so
each pass tracks up to 64 role keypoints + a full-frame support grid
trimmed/padded deterministically to reach exactly 1088 (random padding
points are sampled among the anchor frame's valid-depth pixels,
np.random.default_rng(seed + role_index)).

Output, per sub-task under <out>/traces/<episode>/subtask_XX/ and per
prompt under <prompt_slug>/: coords.npy (T, Q, 3) world-space traces,
visibs.npy (T, Q) visibility flags, queries.npy (Q, 4) query points
(home frame, x, y, z) and metadata.json — plus a sub-task-level
metadata.json summarizing the roles/passes. No visualization (v1).

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
from utils.keyframe_utils import load_subtask_meta
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
#: default inference resolution (H W) of the shipped encoder graph.
DEFAULT_IMAGE_SIZE = (480, 640)

_EP_RE = re.compile(r"^ep(\d{6})$")
_SUB_RE = re.compile(r"^subtask_(\d+)$")


def _slugify(text: str) -> str:
    """Prompt folder slug — mirrors run_object_init_points.py."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Step 4 online: track the Step-3 keypoints with TAPIP3D "
                    "over the Step-2 depth + pose outputs — one pass per role "
                    "(object anchored at the sub-task's first usable key-frame, "
                    "manipulator at the sub-task's first frame), RGB frames "
                    "decoded online from the dataset."
    )
    parser.add_argument("--repo-id", "-id", required=True,
                        help="dataset repo id as seen by LeRobotDataset")
    parser.add_argument("--data-root", "-d", required=True,
                        help="root of the local dataset copy")
    parser.add_argument("--camera-idxes", "-c", nargs="+", type=int,
                        default=None,
                        help="dataset camera indices eligible for tracking "
                             "(default: all; the tracked camera of each "
                             "sub-task is the one its Step-3 init points "
                             "recorded)")
    select = parser.add_mutually_exclusive_group()
    select.add_argument("--episode-idxes", "-e", nargs="*", type=int,
                        default=None,
                        help="only process these episode indices (default: "
                             "all episodes with Step-3 init points on disk)")
    select.add_argument("--one-per-task", action="store_true",
                        help="select the first episode of each task")
    parser.add_argument("--max-episodes", "-x", type=int, default=None,
                        help="cap the number of processed episodes")
    parser.add_argument("--out-dir", "-o", default=None,
                        help="output root (default: <data-root>/eps_data); "
                             "Step-2 results are read under <out-dir>/depth_pose, "
                             "Step-3 under <out-dir>/init_points, traces are "
                             "saved under <out-dir>/traces")
    parser.add_argument("--image-size", nargs=2, type=int,
                        default=list(DEFAULT_IMAGE_SIZE),
                        help="inference resolution (H W), must match the "
                             "encoder graph (default: %(default)s)")
    parser.add_argument("--encoder", default=_DEFAULT_ENCODER,
                        help="TAPIP3D encoder .pt2 artifact")
    parser.add_argument("--iteration", default=_DEFAULT_ITERATION,
                        help="TAPIP3D fused corr+updater .pt2 artifact "
                             "(query count auto-detected from the graph)")
    parser.add_argument("--num-iters", type=int, default=6,
                        help="fused corr+updater iterations inside each "
                             "window (default: %(default)s)")
    parser.add_argument("--vis-threshold", type=float, default=0.5,
                        help="sigmoid visibility threshold for visibs "
                             "(default: %(default)s)")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for the support-grid padding "
                             "(per role: seed + 0/1) (default: %(default)s)")
    parser.add_argument("--device", default=None, choices=["cuda", "cpu"],
                        help="device (default: auto; TAPIP3D is GPU-only)")
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
        if args.camera_idxes is None:
            keys = LeRobotDatasetMetadata(repo_id=args.repo_id,
                                          root=args.data_root).camera_keys
            # every camera stays selectable: the tracked camera of each
            # sub-task comes from its Step-3 init_points.json
            args.camera_idxes = list(range(len(keys)))
        super().__init__(args)
        self.init_root = os.path.join(self.out_dir, "init_points")
        self.depth_pose_root = os.path.join(self.out_dir, "depth_pose")
        self.trace_root = os.path.join(self.out_dir, "traces")
        #: {subtask_index: annotation row}; role resolution degrades to
        #: unlabelled prompts (anchored like the object) when missing.
        try:
            self.subtask_meta = load_subtask_meta(args.data_root)
        except FileNotFoundError:
            self.subtask_meta = {}
        self._device = args.device or ("cuda" if torch.cuda.is_available()
                                       else "cpu")
        self._pt2 = None
        # per-segment state, set by _process_segment()
        self.ep_idx = self.k = self.cam_key = None
        self.seg_depth_dir = None
        self.seg_stems: list[int] = []
        self.seg_labels: dict[int, int] = {}  # per-episode, _process_episode
        self.args = args

    # --- model setup --------------------------------------------------------

    def _ensure_pt2(self) -> Tapip3D_PT2:
        if self._pt2 is None:
            print(f"Loading TAPIP3D .pt2 artifacts...")
            print(f"  Encoder: {self.args.encoder}")
            print(f"  Iteration: {self.args.iteration}")
            self._pt2 = Tapip3D_PT2(
                self.args.encoder, self.args.iteration,
                image_size=tuple(self.args.image_size),
                num_iters=self.args.num_iters)
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

    def _detection_labels(self, ep_idx: int) -> dict[int, int]:
        """{segment ordinal: canonical subtask label} that Step 3a recorded
        in the episode's detections JSON — the label resolves the sub-task
        annotation row of every segment's role split ({} when the JSON is
        missing or carries no labels: the prompts then degrade to unlabelled
        tracking, never to segment-ordinal rows)."""
        path = Path(self.out_dir) / "detections" / f"ep{ep_idx:06d}.json"
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

    def _candidate_frames(self, prompts: list[dict]) -> list[int]:
        """Chronological anchor candidates of one role pass: the sub-task's
        first Step-2 stem first (the manipulator's rule — the object pass
        finds the same frame when its keypoints are usable there), then the
        role's key-frames that are Step-2 stems."""
        stems = self.seg_stems
        cands = [stems[0]]
        for t in sorted({int(i) for p in prompts
                         for i in p["frame_indices"]}):
            if t in stems and t not in cands:
                cands.append(t)
        return cands

    def _usable_at(self, prompt: dict, kf_abs: int, depth: np.ndarray):
        """(rows, px_keyframe, px_depth) of a prompt's keypoints usable on
        the key-frame (Step-2 stem) kf_abs; empty rows when the key-frame
        is not among the prompt's key-frames or nothing is usable there."""
        kfs = list(prompt["frame_indices"])
        if kf_abs not in kfs:
            return np.empty(0, dtype=np.int64), None, None
        j = kfs.index(kf_abs)
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
        self.seg_labels = self._detection_labels(ep_idx)
        ep_init = Path(self.init_root) / f"ep{ep_idx:06d}"
        subtasks = sorted(int(m.group(1)) for p in ep_init.iterdir()
                          if p.is_dir() and (m := _SUB_RE.match(p.name))
                          and any(x.is_dir() for x in p.iterdir()))
        for k in subtasks:
            self._process_segment(k)

    def _process_segment(self, k: int) -> None:
        self.k = k
        seg_init = (Path(self.init_root) / f"ep{self.ep_idx:06d}"
                    / f"subtask_{k:02d}")
        prompts = []
        for pdir in sorted(seg_init.iterdir()):
            if not pdir.is_dir() or not (pdir / "init_points.npz").is_file():
                continue
            try:
                prompt = _read_prompt_dir(pdir)
            except FileNotFoundError as e:
                print(f"  [subtask {k:02d}] skip {pdir.name}: {e}")
                continue
            if prompt["num_keypoints"] <= 0:
                print(f"  [subtask {k:02d}] skip {pdir.name}: no Step-3 "
                      f"keypoints ({prompt['empty_reason'] or 'empty'})")
                continue
            prompts.append(prompt)
        if not prompts:
            print(f"  [subtask {k:02d}] skip: no usable Step-3 prompts")
            return

        # camera: the one Step 3b recorded; every prompt must agree.
        keep = [p for p in prompts if p["camera_key"]]
        if len(keep) < len(prompts):
            print(f"  [subtask {k:02d}] skip {len(prompts) - len(keep)} "
                  f"prompt(s) without a camera_key in their init_points.json")
        prompts = keep
        if not prompts:
            return
        cameras = {p["camera_key"] for p in prompts}
        if len(cameras) > 1:
            print(f"  [subtask {k:02d}] skip: prompts of different cameras "
                  f"{sorted(cameras)} (one camera per sub-task required)")
            return
        self.cam_key = self._cam_key_for_subdir(cameras.pop())
        print(f"  [subtask {k:02d}] camera {self._camera_subdir(self.cam_key)}")

        seg_depth = (Path(self.depth_pose_root) / f"ep{self.ep_idx:06d}"
                     / f"subtask_{k:02d}")
        self.seg_depth_dir = seg_depth / f"depth_{self._camera_subdir(self.cam_key)}"
        if not self.seg_depth_dir.is_dir():
            print(f"  [subtask {k:02d}] skip: no Step-2 outputs under "
                  f"{self.seg_depth_dir} (run run_step2_depth_stream.py for "
                  f"this camera)")
            return
        self.seg_stems = sorted(int(p.stem.rsplit("_", 1)[-1])
                                for p in self.seg_depth_dir.glob("*.npz"))
        if not self.seg_stems:
            print(f"  [subtask {k:02d}] skip: no Step-2 steps in "
                  f"{self.seg_depth_dir}")
            return

        label = self.seg_labels.get(k)
        if label is None:
            print(f"  [subtask {k:02d}] no ground-truth label recorded for "
                  f"it in the Step-3a detections JSON — tracking its "
                  f"prompts unlabelled (re-run Step 3 to regenerate the "
                  f"JSON with labels)")
        row = self.subtask_meta.get(label) if label is not None else None
        roles: dict[str | None, list[dict]] = {}
        for p in prompts:
            role = _prompt_role(row, p["prompt"])
            if role is None:
                print(f"    [{p['slug']}] warning: prompt {p['prompt']!r} "
                      f"matches no object/manipulator entry of sub-task {k}; "
                      f"anchoring it like the object")
            roles.setdefault(role, []).append(p)
        # object and manipulator first, then any unlabelled prompts
        seg_report = {"episode": int(self.ep_idx), "subtask": int(k),
                      "camera_key": self._camera_subdir(self.cam_key),
                      "depth_dir": str(self.seg_depth_dir),
                      "prompts": [], "passes": []}
        order = [r for r in list(ROLE_ORDER) + [None] if r in roles]
        for role in order:
            self._process_role(role, roles[role], seg_report)
        if not seg_report["prompts"] and not seg_report["passes"]:
            if self.args.skip_done:
                print(f"  [subtask {k:02d}] all prompts already tracked "
                      f"(--skip-done)")
            else:
                print(f"  [subtask {k:02d}] skip: nothing to track")
            return
        out_dir = Path(self.trace_root) / f"ep{self.ep_idx:06d}" \
            / f"subtask_{k:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "metadata.json", "w") as f:
            json.dump(seg_report, f, indent=2)

    def _process_role(self, role: str | None, prompts: list[dict],
                      seg_report: dict) -> None:
        """One TAPIP3D pass for a role: pick the anchor key-frame, assemble
        the exact-N queries and track the role's keypoints over the stems
        from the anchor on. Records per-prompt results into seg_report."""
        k = self.k
        label = role or "unlabelled"
        pdirs = [str(Path(self.trace_root) / f"ep{self.ep_idx:06d}"
                     / f"subtask_{k:02d}" / p["slug"]) for p in prompts]
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

        # --- anchor: first candidate frame with usable keypoints ------------
        depth = None
        anchor_abs = None
        usable: dict[str, tuple] = {}
        for cand in self._candidate_frames(prompts):
            depth, intrs, extr = _geometry_at(self.seg_depth_dir, cand)
            for p in prompts:
                usable[p["slug"]] = self._usable_at(p, cand, depth)
            if any(len(u[0]) for u in usable.values()):
                anchor_abs = cand
                break
        if anchor_abs is None:
            reason = "no usable keypoints on any key-frame of the " \
                     "sub-task's Step-2 steps"
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

        # --- sequence: Step-2 stems from the anchor on ------------------------
        steps = self.seg_stems[self.seg_stems.index(anchor_abs):]
        if len(steps) < self._pt2.seq_len:
            print(f"    warning: {len(steps)} steps < the {self._pt2.seq_len}-"
                  f"frame window; the trace will stay at the anchor points "
                  f"(all visibs false)")
        t0 = time.perf_counter()
        coords, visibs_logits = self._track(steps, queries)
        print(f"    tracked {len(steps)} steps in "
              f"{time.perf_counter() - t0:.1f}s")
        visibs = (torch.sigmoid(visibs_logits) >=
                  self.args.vis_threshold).cpu().numpy()

        # --- save: one folder per prompt (its own query columns only) --------
        col = 0
        pass_meta = {"role": role, "anchor_frame": int(anchor_abs),
                     "num_steps": int(len(steps)),
                     "num_queries": int(queries.shape[0]),
                     "num_object_queries": int(n_obj)}
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
            out_dir = Path(d)
            out_dir.mkdir(parents=True, exist_ok=True)
            np.save(out_dir / "coords.npy",
                    coords[:, start:col].cpu().numpy())
            np.save(out_dir / "visibs.npy", visibs[:, start:col])
            np.save(out_dir / "queries.npy",
                    queries[start:col].cpu().numpy())
            entry = {
                "episode": int(self.ep_idx), "subtask": int(k),
                "role": role, "prompt": p["prompt"],
                "prompt_slug": p["slug"],
                "camera_key": self._camera_subdir(self.cam_key),
                "status": "ok",
                "anchor_frame": int(anchor_abs),
                "steps": [int(s) for s in steps],
                "num_steps": int(len(steps)),
                "num_queries": int(n),
                "pass_queries": int(queries.shape[0]),
                "query_keypoint_rows":
                    [int(i) for i in rows],
                "pixels": [[float(x), float(y)] for x, y in
                           (px_kf if px_kf is not None else [])],
                "model": {"encoder": str(Path(self.args.encoder).absolute()),
                          "iteration":
                              str(Path(self.args.iteration).absolute()),
                          "num_iters": self.args.num_iters},
                "image_size": list(self.args.image_size),
                "vis_threshold": self.args.vis_threshold,
                "seed": self.args.seed,
                "inputs": {"init_points_dir": str(p["dir"]),
                           "depth_dir": str(self.seg_depth_dir)},
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
            "camera_key": self._camera_subdir(self.cam_key),
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
        resolution (the same math as load_resized_batch). Returns
        (coords (T, Q, 3), visibs_logits (T, Q)) CPU."""
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
        si = Tapip3DStreamPT2(pt2, queries, depth_roi=depth_roi)
        with torch.inference_mode():
            coords, visibs = si.run(batches(), len(steps))
        return coords, visibs


def main() -> None:
    args = parse_args()
    # the TAPIP3D path runs validated numerics (fp32, TF32 off) — same
    # convention as infer_tapip3d.py
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    SubtaskTraceExtract(args).run()


if __name__ == "__main__":
    main()
