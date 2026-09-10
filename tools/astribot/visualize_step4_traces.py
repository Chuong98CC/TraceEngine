"""Per-sub-task 3D-trace videos, online — the visualization counterpart of
run_step4_traces.py.

For every tracked camera of the selected episodes it renders two videos
under the shared visualization tree, with NO model inference and nothing
extracted to disk — RGB frames decoded online from the LeRobotDataset,
geometry from the Step-2 depth_pose npzs, traces from the Step-4
coords/visibs:

    <out>/visualization/<episode>/subtask_XX/<camera>/
        trace2d.mp4  — the world-space keypoint traces projected back onto
                       the RGB frames (a 2D overlay: role-colored keypoint
                       markers + trails, invisible keypoints red, abs-step
                       HUD);
        trace3d.mp4  — the 3D point-cloud scene (per-step cloud + frustum,
                       as visualize_step2_depth_pose.py renders) with the
                       growing world-space trace curves overlaid in role
                       colors.

Different role prompts cover different step windows (object = the
close..open transport only, manipulator = the whole sub-task), so each
video renders over the sorted union of the camera's prompt steps — every
union step is a Step-2 stem, so its pose npz exists.

Colour legend: object green, manipulator orange, unlabelled cyan,
occluded red.

Episodes, sub-task segments, cameras and rendered steps are discovered
from the saved Step-4 trace outputs themselves — <out-dir>/traces/ep*/
subtask_XX/<camera>/<prompt>/coords.npy — plus the Step-2 depth_pose
geometry: no dataset split inference, no subtask_splits.json, no
selection flag that must match the run_step4_traces.py run; ``-e``/
``-c`` only filter what exists on disk.

Examples
--------
    # Render every tracked camera of every sub-task of episode 0
    python tools/astribot/visualize_step4_traces.py
        --repo-id Kronze157/astribot_making_coffee_vlva_full
        --data-root /data/astri_making_coffee --episode-idxes 0
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import cv2
import imageio
import numpy as np
import open3d as o3d
import trimesh
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from tqdm import tqdm

from tools.general_test.pipeline.visualize_stream import render_stream_video
from utils.visualize.visualize_mask import to_pil

#: role output order of the sub-task annotations (meta/subtasks.csv
#: columns); a prompt whose text matches no role entry is unlabelled
#: (its metadata role key is None).
ROLE_ORDER = ("object", "manipulator")
#: trace-role colours in BGR (the cv2 drawing space): object green,
#: manipulator orange, unlabelled cyan; occluded keypoints draw red.
ROLE_COLORS = {"object": (0, 255, 0), "manipulator": (0, 140, 255),
               None: (255, 255, 0)}
OCCLUDED_COLOR = (0, 0, 255)
#: current-point marker radius (px) of the 2D overlay, occluded ones draw
#: at half that size (the render_tracks conventions of visualize_tapip3d).
POINT_SIZE = 2
#: base brightness of the oldest trail segment (brightness grows towards
#: the current point, like draw_tracks' trail_alpha).
TRAIL_ALPHA = 0.4

_EP_RE = re.compile(r"^ep(\d{6})$")
_SUB_RE = re.compile(r"^subtask_(\d+)$")


def _role_color_rgb01(role: str | None) -> np.ndarray:
    """Role colour as 0..1 RGB floats (open3d's colour space)."""
    bgr = np.asarray(ROLE_COLORS[role], dtype=np.float64)
    return bgr[::-1] / 255.0


def _project_points(world: np.ndarray, intr: np.ndarray,
                    extr: np.ndarray) -> np.ndarray:
    """World (Q, 3) -> pixels (Q, 2) at the intrinsics' resolution: extr is
    the w2c (4, 4) pose, px = K @ (extr @ [x, y, z, 1]) dehomogenized —
    the same math as flow_models.tapip3d.utils._common.batch_project, kept
    numpy so the tapip3d runtimes stay out of this module."""
    n = world.shape[0]
    cam = (extr @ np.hstack([world, np.ones((n, 1))]).T).T     # (Q, 4)
    cam = cam[:, :3] / cam[:, 3:]
    img = (intr @ cam.T).T                                     # (Q, 3)
    return img[:, :2] / img[:, 2:]


def _hud_text(canvas: np.ndarray, text: str, org: tuple[int, int],
              color: tuple[int, int, int], scale: float = 0.6) -> None:
    """Text with a black outline under a coloured fill — the
    _label_viewport style of tools/general_test/pipeline/visualize_stream.py."""
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, 1, cv2.LINE_AA)


def _draw_legend(canvas: np.ndarray, roles_present: list, y: int) -> None:
    """One-line legend at ``y`` (baseline): a colour chip + name per role
    present (in ROLE_ORDER, unlabelled last), then the occluded chip."""
    entries = []
    for role in list(ROLE_ORDER) + [None]:
        if role in roles_present:
            entries.append((role if role is not None else "unlabelled",
                            ROLE_COLORS[role]))
    entries.append(("occluded", OCCLUDED_COLOR))
    x = 8
    for name, color in entries:
        cv2.circle(canvas, (x + 5, y - 4), 4, color, -1, cv2.LINE_AA)
        _hud_text(canvas, name, (x + 14, y), color, scale=0.5)
        (w, _), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        x += 18 + w


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Render the Step-4 3D-trace videos (trace2d.mp4 / "
                    "trace3d.mp4) of every selected camera of each sub-task "
                    "segment of an episode, online (RGB frames decoded from "
                    "the LeRobotDataset; geometry from the Step-2 depth_pose "
                    "NPZs; traces from the Step-4 coords/visibs). Episodes, "
                    "sub-task segments, cameras and rendered steps are "
                    "discovered from the saved Step-4 trace outputs on disk "
                    "— no split inference, no subtask_splits.json, no "
                    "selection flag that must match the run_step4_traces.py "
                    "run; -e/-c only filter what exists."
    )
    parser.add_argument("--repo-id", "-id", required=True,
                        help="dataset repo id as seen by LeRobotDataset")
    parser.add_argument("--data-root", "-d", required=True,
                        help="root passed to LeRobotDataset; for chunked datasets give the "
                             "full chunk path, e.g. /data/x/chunk-0000/part-0000")
    parser.add_argument("--camera-idxes", "-c", nargs="+", type=int, default=None,
                        help="indices into the dataset's camera_keys to "
                             "visualize (default: every camera that has "
                             "Step-4 trace outputs on disk)")
    parser.add_argument("--episode-idxes", "-e", nargs="*", type=int, default=None,
                        help="only visualize these episode indices (default: "
                             "every episode with Step-4 traces under "
                             "<out-dir>/traces); an episode with no traces "
                             "on disk is warned about and skipped")
    parser.add_argument("--max-episodes", "-x", type=int, default=None,
                        help="cap the number of discovered episodes (first N, "
                             "after any -e filter)")
    parser.add_argument("--out-dir", "-o", default=None,
                        help="output root (default: <data-root>/eps_data); "
                             "Step-2 geometry is read under <out-dir>/"
                             "depth_pose, Step-4 traces under <out-dir>/"
                             "traces, videos written under <out-dir>/"
                             "visualization")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--trail-len", type=int, default=30,
                        help="2D-overlay trail window: trailing projected "
                             "steps drawn per keypoint (default: %(default)s)")
    parser.add_argument("--size", default="960x540",
                        help="video size WxH, e.g. 960x540 (default: 960x540)")
    parser.add_argument("--max-points", type=int, default=100_000,
                        help="max point-cloud points rendered per video frame "
                             "(default: 100000)")
    parser.add_argument(
        "--views", type=int, choices=[1, 4], default=4,
        help="viewpoints per frame: 1 (center only) or 4 (2x2 grid of "
             "center/down/left/right) (default: 4)",
    )
    parser.add_argument(
        "--view-distance", type=float, default=0.3,
        help="eye distance behind the first camera, in scene-extent units "
             "(default: 0.3)",
    )
    parser.add_argument(
        "--view-angle", type=float, default=45.0,
        help="side-view swing for the left/right viewpoints, in degrees off "
             "the center view around the scene's vertical axis (clamped to "
             "85; default: 45)",
    )
    parser.add_argument(
        "--view-lower", type=float, default=0.1,
        help="downward shift of the left/right viewpoints below the center "
             "eye, in scene-extent units (default: 0.1)",
    )
    parser.add_argument(
        "--view-raise", type=float, default=0.1,
        help="elevation of the down viewpoint, in scene-extent units "
             "(default: 0.1)",
    )
    parser.add_argument(
        "--view-back", type=float, default=0.3,
        help="backward pull of the down viewpoint, in scene-extent units "
             "(default: 0.3)",
    )
    parser.add_argument(
        "--view-fov", type=float, default=None,
        help="override the auto-fitted vertical field of view in degrees "
             "(default: auto-fit each viewport to the scene)",
    )
    parser.add_argument("--render", choices=["2d", "3d", "both"],
                        default="both",
                        help="which videos to render per camera "
                             "(default: %(default)s)")
    return parser.parse_args(argv)


class SubtaskTraceVisualize:
    """Render the Step-4 trace videos of every tracked camera, online.

    Standalone: the episodes, sub-task segments, cameras and rendered
    steps are all discovered from the saved Step-4 trace outputs on disk
    (traces/<ep>/subtask_XX/<camera>/<prompt>) — never recomputed from
    dataset splits, so no selection flag has to match the
    run_step4_traces.py run. The Step-2 depth_pose tree supplies the
    geometry behind trace3d.mp4 (every union step is a Step-2 stem); the
    dataset is only consulted for camera_keys (decoding the -c indices
    and mapping the on-disk camera subdirs back to decode keys) and for
    the online frame decode. The videos are written under the shared
    visualization tree (viz_root), next to the step-2 depth_pose.mp4
    videos.
    """

    def __init__(self, args):
        self.args = args
        # dataset metadata for camera_keys only — episodes/segments come
        # from the disk (self.trace_root), not from the dataset
        self.ds_meta = LeRobotDatasetMetadata(repo_id=args.repo_id,
                                              root=args.data_root)
        self.dataset = None  # LeRobotDataset handle, created lazily
        self.cam_keys = self._select_cameras()
        self.out_dir = args.out_dir or os.path.join(args.data_root, "eps_data")
        # Step-4 trace outputs read from <out-dir>/traces/<episode>/
        # subtask_XX/<camera>/, Step-2 geometry from <out-dir>/depth_pose/
        # <episode>/subtask_XX/ — the videos go to the sibling
        # visualization/ tree
        self.trace_root = Path(self.out_dir) / "traces"
        self.depth_pose_root = Path(self.out_dir) / "depth_pose"
        self.viz_root = Path(self.out_dir) / "visualization"
        # per-camera state, set by _process_camera()
        self.ep_idx = self.k = self.cam_key = self.cam_subdir = None
        self.seg_dir: Path | None = None      # depth_pose/<ep>/subtask_XX
        self.seg_depth_dir: Path | None = None  # .../subtask_XX/depth_<cam>
        self.n_rendered = 0

    # --- dataset access -----------------------------------------------------

    def _ensure_dataset(self):
        """LeRobotDataset handle (videos on disk), created lazily: only
        the per-step frame decode opens it."""
        if self.dataset is None:
            self.dataset = LeRobotDataset(repo_id=self.args.repo_id,
                                          root=self.args.data_root,
                                          download_videos=False)
        return self.dataset

    @staticmethod
    def _camera_subdir(cam_key):
        """Subdir per camera, named after the dataset's camera key (the
        key_frames naming, mirrors run_step4_traces.py)."""
        return cam_key.rsplit(".", 1)[-1]

    def _select_cameras(self):
        """-c indices -> dataset camera keys, validated against the
        dataset's camera_keys (None: every camera with Step-4 trace
        outputs on disk is rendered)."""
        idxes = self.args.camera_idxes
        if idxes is None:
            return None
        for idx in idxes:
            if not 0 <= idx < len(self.ds_meta.camera_keys):
                raise ValueError(f"camera_idx {idx} out of range "
                                 f"(dataset has {len(self.ds_meta.camera_keys)} cameras)")
        return {idx: self.ds_meta.camera_keys[idx] for idx in idxes}

    def _cam_key_for_subdir(self, subdir: str) -> str:
        """Dataset camera key whose subdir (key_frames naming) matches a
        Step-4 trace camera dir — the key decoding that camera's frames
        online. With -c only the requested cameras are candidates (a
        trace camera outside them is skipped); without, every dataset
        camera key is a candidate."""
        candidates = (self.cam_keys.values() if self.cam_keys is not None
                      else self.ds_meta.camera_keys)
        for key in candidates:
            if self._camera_subdir(key) == subdir:
                return key
        raise ValueError(
            f"camera {subdir!r} of the Step-4 traces has no dataset "
            f"camera among "
            f"{sorted({self._camera_subdir(k) for k in candidates})}")

    # --- frame access -------------------------------------------------------

    def _frame_bgr(self, abs_idx: int) -> np.ndarray:
        """One frame of the current trace camera at an absolute dataset
        index, decoded online as BGR uint8 at its native resolution."""
        frame = self._ensure_dataset()[abs_idx]
        return np.asarray(to_pil(frame[self.cam_key]), dtype=np.uint8)[:, :, ::-1]

    def _frame_loader(self, h: int, w: int):
        """Per-step RGB loader for render_stream_video of the current
        camera: decodes the step's dataset frame and resizes it to the
        depth resolution, matching load_pair's image contract (RGB —
        the renderer wants (N, H, W, 3) uint8)."""
        def load(stem: str) -> np.ndarray:
            abs_idx = int(stem.rsplit("_", 1)[-1])
            rgb = self._frame_bgr(abs_idx)[:, :, ::-1]
            if rgb.shape[:2] != (h, w):
                rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
            return rgb[None]

        return load

    # --- disk discovery -----------------------------------------------------

    def _episode_dir(self, ep_idx: int) -> str:
        """The on-disk episode dir name of a dataset episode index."""
        return f"ep{ep_idx:06d}"

    def _discover_episodes(self) -> list[int]:
        """Sorted dataset indices of the episode dirs under traces/
        (a dir counts when its name is ep<int>)."""
        eps = []
        for p in Path(self.trace_root).iterdir():
            if not p.is_dir() or (m := _EP_RE.match(p.name)) is None:
                continue
            eps.append(int(m.group(1)))
        return sorted(eps)

    def _segment_dirs(self, ep_idx: int) -> list[int]:
        """Sorted subtask ints of an episode, as present under
        traces/ep%06d/ — the Step-2 segmentation the traces were tracked
        over, never recomputed from dataset splits."""
        ep_dir = Path(self.trace_root) / self._episode_dir(ep_idx)
        if not ep_dir.is_dir():
            return []
        return sorted(int(m.group(1)) for p in ep_dir.iterdir()
                      if p.is_dir() and (m := _SUB_RE.match(p.name)))

    def _select_episodes(self, discovered: list[int]) -> list[int]:
        """Episodes to process: the discovered ones, filtered by -e (a
        requested episode with no Step-4 traces on disk is warned about
        and skipped — never a failure) and capped by --max-episodes."""
        if self.args.episode_idxes is not None:
            requested = set(self.args.episode_idxes)
            for ep in sorted(requested - set(discovered)):
                print(f"  episode {ep}: no Step-4 traces under "
                      f"{self.trace_root} — skipped")
            eps = [ep for ep in discovered if ep in requested]
        else:
            eps = discovered
        if self.args.max_episodes is not None:
            eps = eps[: self.args.max_episodes]
        return eps

    # --- geometry -----------------------------------------------------------

    def _pose_at(self, depth_dir: Path, abs_idx: int):
        """Step-2 pose of one step: (intrinsics (3, 3), extrinsics padded
        to (4, 4), depth shape (H, W)) — mirrors run_step4_traces.
        _geometry_at minus the depth .lz4 read."""
        pose_path = depth_dir / f"frame_{abs_idx:06d}.npz"
        if not pose_path.is_file():
            raise FileNotFoundError(
                f"Step-4 steps were tracked over existing Step-2 stems, "
                f"yet the pose npz is missing: {pose_path}")
        with np.load(pose_path) as data:
            extr = data["extrinsics"] if "extrinsics" in data else data["extrinsic"]
            if extr.shape == (3, 4):
                extr = np.vstack([extr, [0.0, 0.0, 0.0, 1.0]])
            intr = data["intrinsics"] if "intrinsics" in data else data["intrinsic"]
            shape = (tuple(int(v) for v in data["shape"])
                     if "shape" in data else None)
        return (intr.astype(np.float32), extr.astype(np.float32), shape)

    # --- orchestration ------------------------------------------------------

    def run(self) -> None:
        if not self.trace_root.is_dir() or not self.depth_pose_root.is_dir():
            raise FileNotFoundError(
                f"need Step-4 traces ({self.trace_root}) and Step-2 depth + "
                f"pose ({self.depth_pose_root}): run run_step4_traces.py and "
                f"run_step2_depth_stream.py first")
        discovered = self._discover_episodes()
        if not discovered:
            print(f"\nno Step-4 traces under {self.trace_root} — run "
                  f"run_step4_traces.py first")
            return
        eps = self._select_episodes(discovered)
        print(f"\n{len(eps)} episode(s) selected:")
        for ep in eps:
            print(f"  episode {ep}: {len(self._segment_dirs(ep))} "
                  f"sub-task segment(s)")
        for ep_idx in tqdm(eps, desc="episodes"):
            self._process_episode(ep_idx)
        print(f"\ndone: {self.n_rendered} camera(s) -> {self.viz_root}")

    def _process_episode(self, ep_idx: int) -> None:
        self.ep_idx = ep_idx
        for k in self._segment_dirs(ep_idx):
            self._process_segment(k)

    def _process_segment(self, k: int) -> None:
        """Render the trace videos of every tracked camera of the sub-task
        — the cameras come from the Step-4 trace dirs (traces/.../subtask_
        XX/<camera>/), each with its own per-prompt coords/visibs."""
        self.k = k
        seg_trace = (Path(self.trace_root) / f"ep{self.ep_idx:06d}"
                     / f"subtask_{k:02d}")
        cameras = sorted(p.name for p in seg_trace.iterdir()
                         if p.is_dir()
                         and any(x.is_dir() for x in p.iterdir()))
        for cam in cameras:
            self._process_camera(cam)

    def _process_camera(self, cam: str) -> None:
        """Both videos of one (sub-task, camera): the renderable prompts
        are the camera's Step-4 prompt subdirs with coords.npy and a
        status-ok metadata.json; their union of abs steps drives both
        videos."""
        k = self.k
        cam_trace = (Path(self.trace_root) / f"ep{self.ep_idx:06d}"
                     / f"subtask_{k:02d}" / cam)
        prompts = []
        for pdir in sorted(cam_trace.iterdir()):
            if not pdir.is_dir() or not (pdir / "coords.npy").is_file():
                continue
            meta_path = pdir / "metadata.json"
            if not meta_path.is_file() or \
                    json.loads(meta_path.read_text()).get("status") != "ok":
                print(f"  [subtask {k:02d}] camera {cam}: skip {pdir.name}: "
                      f"no status-ok metadata.json (run_step4_traces.py "
                      f"outputs a metadata.json next to every coords.npy)")
                continue
            prompts.append(self._read_prompt(pdir))
        if not prompts:
            print(f"  [subtask {k:02d}] camera {cam}: skip, no Step-4 "
                  f"trace prompts under {cam_trace}")
            return
        self.cam_subdir = cam
        try:
            self.cam_key = self._cam_key_for_subdir(cam)
        except ValueError as e:
            print(f"  [subtask {k:02d}] camera {cam}: skip, {e}")
            return
        self.seg_dir = (Path(self.depth_pose_root) / f"ep{self.ep_idx:06d}"
                        / f"subtask_{k:02d}")
        self.seg_depth_dir = self.seg_dir / f"depth_{cam}"
        if not self.seg_depth_dir.is_dir():
            print(f"  [subtask {k:02d}] camera {cam}: skip, no Step-2 "
                  f"outputs under {self.seg_depth_dir} (run "
                  f"run_step2_depth_stream.py for this camera)")
            return

        for p in prompts:
            print(f"    [{p['slug']}] {p['role'] or 'unlabelled'} role, "
                  f"{p['num_steps']} steps")
        union_abs = sorted({int(s) for p in prompts for s in p["steps"]})
        # union steps were tracked over the camera's Step-2 stems, so
        # every one has geometry (enforced again per frame by _pose_at)
        print(f"  [subtask {k:02d}] camera {cam}: {len(prompts)} prompt(s) "
              f"-> {len(union_abs)} union steps {union_abs[0]}.."
              f"{union_abs[-1]} (from {cam_trace})")
        viz_dir = (self.viz_root / f"ep{self.ep_idx:06d}"
                   / f"subtask_{k:02d}" / cam)
        viz_dir.mkdir(parents=True, exist_ok=True)
        if self.args.render in ("2d", "both"):
            self._render_2d(prompts, union_abs, viz_dir / "trace2d.mp4")
        if self.args.render in ("3d", "both"):
            self._render_3d(prompts, union_abs, viz_dir / "trace3d.mp4")
        self.n_rendered += 1

    def _read_prompt(self, pdir: Path) -> dict:
        """A Step-4 prompt output: coords (T, Q, 3) world positions, visibs
        (T, Q) and the metadata role/steps (row i of coords corresponds to
        the absolute dataset step steps[i])."""
        meta = json.loads((pdir / "metadata.json").read_text())
        coords = np.load(pdir / "coords.npy")
        visibs = np.load(pdir / "visibs.npy")
        steps = [int(s) for s in meta["steps"]]
        if not (len(coords) == len(visibs) == len(steps)):
            raise ValueError(
                f"{pdir}: coords {coords.shape} / visibs {visibs.shape} "
                f"rows disagree with {len(steps)} metadata steps")
        return {
            "slug": pdir.name,
            "prompt": meta.get("prompt", pdir.name),
            "role": meta.get("role"),
            "steps": steps,
            "num_steps": int(meta.get("num_steps", len(steps))),
            "coords": coords.astype(np.float32),   # (T, Q, 3) world
            "visibs": np.asarray(visibs, dtype=bool),   # (T, Q)
            "row_of": {step: r for r, step in enumerate(steps)},
            # recent projected 2D history of the prompt's own rendered
            # steps (filled by _render_2d, capped at --trail-len)
            "trail_px": [],
            "trail_vis": [],
        }

    # --- 2D overlay ---------------------------------------------------------

    def _render_2d(self, prompts: list[dict], union_abs: list[int],
                   out_path: Path) -> Path:
        """trace2d.mp4: the world-space keypoints of every union step
        projected back onto that step's RGB frame (intrinsics rescaled when
        the frame resolution differs from the npz's), with per-prompt
        age-faded trails and role-coloured markers (occluded red) plus the
        abs-step HUD and legend — the render_tracks conventions of
        utils/visualize/visualize_tapip3d.py."""
        writer = imageio.get_writer(str(out_path), fps=self.args.fps,
                                    codec="libx264", quality=8,
                                    macro_block_size=1)
        try:
            for i, abs_step in enumerate(union_abs):
                frame = self._frame_bgr(abs_step)
                orig_h, orig_w = frame.shape[:2]
                intr, extr, shape = self._pose_at(self.seg_depth_dir,
                                                  abs_step)
                dh, dw = shape if shape is not None else (orig_h, orig_w)
                # intrinsics sit at the npz depth resolution; scale them
                # row-wise onto the (possibly different) native frame
                # resolution — the exact convention of render_tracks
                if (orig_h, orig_w) != (dh, dw):
                    intr[0, :] *= (orig_w - 1) / (dw - 1)
                    intr[1, :] *= (orig_h - 1) / (dh - 1)
                overlay = frame.copy()
                for p in prompts:
                    r = p["row_of"].get(abs_step)
                    if r is None:
                        continue
                    px = _project_points(p["coords"][r], intr, extr)
                    p["trail_px"].append(px)
                    p["trail_vis"].append(p["visibs"][r])
                    if self.args.trail_len > 0:
                        del p["trail_px"][:-self.args.trail_len]
                        del p["trail_vis"][:-self.args.trail_len]
                    self._draw_trail(overlay, p)
                    self._draw_points(overlay, px, p["visibs"][r],
                                      ROLE_COLORS[p["role"]])
                frame = cv2.addWeighted(frame, 0.3, overlay, 0.7, 0)
                roles_present = [p["role"] for p in prompts]
                self._draw_hud(frame, abs_step, roles_present)
                # libx264/yuv420p wants even dimensions; pad odd native
                # sizes with a black bar (the render_tracks convention)
                out_h, out_w = orig_h - orig_h % 2, orig_w - orig_w % 2
                if (out_h, out_w) != (orig_h, orig_w):
                    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
                    canvas[:orig_h, :orig_w] = frame
                    frame = canvas
                writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                print(f"  rendered {i + 1}/{len(union_abs)}: "
                      f"abs {abs_step:04d}")
        finally:
            writer.close()
        print(f"    -> {out_path}")
        return out_path

    def _draw_trail(self, overlay: np.ndarray, prompt: dict) -> None:
        """Age-faded trail segments of one prompt's keypoints on its
        overlay: each keypoint's recent projected positions (its own
        rendered steps, up to --trail-len) are connected through the
        visible-in-bounds entries, segment brightness growing towards the
        current point (draw_tracks' scheme, recoloured per role)."""
        h, w = overlay.shape[:2]
        color = ROLE_COLORS[prompt["role"]]
        for q in range(prompt["coords"].shape[1]):
            pts = []
            for tpx, tvis in zip(prompt["trail_px"], prompt["trail_vis"]):
                if not tvis[q]:
                    continue
                x, y = tpx[q]
                if 0 <= x < w and 0 <= y < h:
                    pts.append((int(x), int(y)))
            for j in range(1, len(pts)):
                alpha = TRAIL_ALPHA * (j / max(len(pts), 1))
                seg = tuple(int(c * alpha) for c in color)
                cv2.line(overlay, pts[j - 1], pts[j], seg, 1, cv2.LINE_AA)

    @staticmethod
    def _draw_points(overlay: np.ndarray, px: np.ndarray, vis: np.ndarray,
                     color: tuple[int, int, int]) -> None:
        """Current keypoints on the overlay: role-coloured filled circles
        where visible, small red circles where occluded (all roles)."""
        h, w = overlay.shape[:2]
        for q, (x, y) in enumerate(px):
            if not (0 <= x < w and 0 <= y < h):
                continue
            radius = POINT_SIZE if vis[q] else POINT_SIZE // 2
            col = color if vis[q] else OCCLUDED_COLOR
            cv2.circle(overlay, (int(x), int(y)), radius, col, -1,
                       cv2.LINE_AA)

    def _draw_hud(self, frame: np.ndarray, abs_step: int,
                  roles_present: list) -> None:
        """Abs-step identity line and the role-colour legend, top-left."""
        label = (f"ep{self.ep_idx:06d} subtask_{self.k:02d} "
                 f"{self.cam_subdir} | abs {abs_step:04d}")
        _hud_text(frame, label, (8, 22), (255, 255, 255))
        _draw_legend(frame, roles_present, 46)

    # --- 3D scene -----------------------------------------------------------

    def _render_3d(self, prompts: list[dict], union_abs: list[int],
                   out_path: Path) -> Path:
        """trace3d.mp4 via render_stream_video (visualize_step2_depth_pose
        renders the same scene): the per-step clouds + frustum get the
        growing world-space trace curves of every prompt overlaid through
        its trace_geoms_fn hook."""
        stems = [f"frame_{s:06d}" for s in union_abs]
        # fit points: the prompts' tracked positions (rows where anything
        # is visible — an all-occluded row's estimates can wander far and
        # would skew the auto-fitted view away from the action)
        rows = [p["coords"][p["visibs"].any(axis=1)].reshape(-1, 3)
                for p in prompts]
        extra = np.concatenate(rows) if rows else np.empty((0, 3))
        extra = extra[np.isfinite(extra).all(axis=1)]
        if len(extra) > 20_000:
            extra = extra[::2]
        w_vid, h_vid = (int(x) for x in self.args.size.lower().split("x"))

        def trace_geoms_fn(t: int, alignment: np.ndarray):
            """Per-union-step overlay: current visible keypoints as
            role-coloured points + per-keypoint trail polylines through
            their own visible history up to row r (consecutive visible
            rows only), all in the aligned glTF frame."""
            abs_step = union_abs[t]
            geoms = []
            for p in prompts:
                r = p["row_of"].get(abs_step)
                if r is None:
                    continue
                color01 = _role_color_rgb01(p["role"])
                vis = p["visibs"]
                if vis[r].any():
                    pts = trimesh.transform_points(
                        p["coords"][r][vis[r]], alignment)
                    pcd = o3d.geometry.PointCloud(
                        o3d.utility.Vector3dVector(pts))
                    pcd.colors = o3d.utility.Vector3dVector(
                        np.tile(color01, (len(pts), 1)))
                    geoms.append(pcd)
                ls_pts: list[np.ndarray] = []
                ls_lines: list[np.ndarray] = []
                for q in range(p["coords"].shape[1]):
                    rows_q = np.flatnonzero(vis[:r + 1, q])
                    if len(rows_q) < 2:
                        continue
                    # break across invisible rows: only consecutive
                    # visible rows connect
                    breaks = np.flatnonzero(np.diff(rows_q) != 1) + 1
                    for group in np.split(rows_q, breaks):
                        if len(group) < 2:
                            continue
                        base = sum(len(a) for a in ls_pts)
                        ls_pts.append(p["coords"][group, q])
                        ls_lines.append(np.stack(
                            [np.arange(base, base + len(group) - 1),
                             np.arange(base + 1, base + len(group))],
                            axis=1))
                if ls_pts:
                    pts = trimesh.transform_points(
                        np.concatenate(ls_pts, axis=0), alignment)
                    line_set = o3d.geometry.LineSet()
                    line_set.points = o3d.utility.Vector3dVector(pts)
                    line_set.lines = o3d.utility.Vector2iVector(
                        np.concatenate(ls_lines, axis=0))
                    line_set.colors = o3d.utility.Vector3dVector(
                        np.tile(color01, (len(line_set.lines), 1)))
                    geoms.append(line_set)
            return geoms or None

        print(f"  [subtask {self.k:02d}] camera {self.cam_subdir}: 3D "
              f"scene over {len(stems)} steps -> {out_path}")
        # view-fit stride derived from the union steps' dataset spacing:
        # only a spatial thinning of each depth map for the auto view-fit
        # (visualize_stream._union_scene_points) — unrelated to any producer
        # frame stride (fallback 1: single-step segment)
        fit_stride = (union_abs[1] - union_abs[0] if len(union_abs) > 1 else 1)
        render_stream_video(
            stems,
            [self.cam_subdir],
            str(self.seg_dir),
            str(out_path),
            fps=self.args.fps,
            size=(w_vid, h_vid),
            max_points_per_frame=self.args.max_points,
            stride=fit_stride,
            view_distance=self.args.view_distance,
            views=self.args.views,
            view_angle=self.args.view_angle,
            view_lower=self.args.view_lower,
            view_raise=self.args.view_raise,
            view_back=self.args.view_back,
            view_fov=self.args.view_fov,
            frame_loader=self._frame_loader(*self._depth_shape(union_abs[0])),
            extra_fit_points=extra,
            trace_geoms_fn=trace_geoms_fn,
        )
        return out_path

    def _depth_shape(self, abs_idx: int) -> tuple[int, int]:
        """(H, W) depth resolution of the camera's Step-2 outputs, probed
        from the npz's shape entry — the pose npz records the depth shape
        its .lz4 buffer is stored at (see load_stream_data)."""
        _, _, shape = self._pose_at(self.seg_depth_dir, abs_idx)
        if shape is None:
            raise FileNotFoundError(
                f"no depth shape recorded in {self.seg_depth_dir}/"
                f"frame_{abs_idx:06d}.npz")
        return shape


def main():
    SubtaskTraceVisualize(parse_args()).run()


if __name__ == "__main__":
    main()
