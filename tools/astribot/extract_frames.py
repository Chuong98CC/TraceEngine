"""Extract sub-task splits, key frames and/or per-subtask frames and videos
from a local LeRobotDataset copy.

Four independent modes:

- detect_subtask: detects the gripper key frames and infers the sub-task
  split frames from the tabular observation.state only — no video is read, so
  it also runs on datasets whose videos are not on disk. Saves the gripper
  plot (raw + filtered state, key frames, split frames) and a
  subtask_splits.json recording the key and split frames under
  <out_dir>/subtask/<episode>/.
- key_frames: saves one jpg per selected camera of the episode's first and
  last frame, the key frames detected by detect_subtask, and the start and
  end frame of every sub-task segment, each under the sub-task segment it
  belongs to:
  <out_dir>/key_frames/<episode>/subtask_XX/<camera>/.
- videos: cuts one mp4 per sub-task segment (the episode split at the split
  frames) per selected camera from the source videos with ffmpeg: each
  segment is re-encoded with x264 at its exact split frames (no per-frame
  Python decode; a decode+encode pass is unavoidable for frame-accurate
  H.264 cuts), under <out_dir>/subtask_videos/<episode>/<camera>/.
- frames: saves one jpg per selected camera of every --interval-th frame of
  each sub-task segment (capped at --max-frames per sub-task), under
  <out_dir>/subtask_frames/<episode>/subtask_XX/<camera>/. When a camera has
  a paired depth feature stored as uint16 (mm), the raw depth array is saved
  alongside as frame_<idx>.lz4 (see utils.astribot_dataloader.save_depth_lz4,
  loadable by load_depth_lz4) under
  <out_dir>/subtask_frames/<episode>/subtask_XX/depth_<camera>/.

Key frames are the first and last frame of an episode plus every frame where
the gripper state crosses the KEY_FRAME_CHANGE open/close threshold, detected
on the low-pass filtered signal; runs of closed state shorter than
KEY_FRAME_MIN_CLOSE_S seconds are dropped first, so a noisy blip does not
produce a key-frame pair. The sub-task split frames are the re-grasp
midpoints of the combined 1 -> 0 -> 1 gripper pattern (see
_subtask_split_idxes). The videos, frames and key_frames modes prefer the
dataset's ground-truth subtask_index column when it exists — the split
frames are then the first frame of each new sub-task segment (see
_split_frames_from_ground_truth) — and fall back to subtask_splits.json only
when the episode has fewer than two sub-tasks. The ground-truth labels are
sometimes wrong; pass --use-inferred-splits to always prefer the split
frames inferred by detect_subtask instead. The key_frames mode always uses
the gripper-detected key frames of detect_subtask (ground truth has no key
frames) and additionally saves the start and end frame of every sub-task
segment (its boundaries under the chosen split source); each saved jpg
lands in the sub-task segment its frame index belongs to.

Depth pairing: a camera observation.images.cam_X is paired with the
observation.depth.cam_X feature (uint16 mm) when present, else with the
legacy <cam_key>_depth video. The video is only trusted when the dataset
metadata flags it as a real depth map (video.is_depth_map) — the
astri_making_coffee recording is flagged false and its depth is unusable
(see the re-recorded astri_making_coffee_v1 dataset). The dataset videos
must be present on disk for the key_frames, videos and frames modes
(LeRobotDataset is opened with download_videos=False).

Examples
--------
    # 1) infer the key and sub-task split frames of every episode (no videos)
    python tools/astribot/extract_frames.py \
        --repo-id Kronze157/astri_making_coffee_vlva \
        --data-root /data/astri_making_coffee_v1 --mode detect_subtask

    # 2) save the first/last/key frames as jpgs for the first two cameras
    python tools/astribot/extract_frames.py \
        --repo-id Kronze157/astri_making_coffee_vlva \
        --data-root /data/astri_making_coffee_v1 --mode key_frames \
        --camera-idxes 0 1

    # 3) one mp4 per sub-task segment, first two cameras
    python tools/astribot/extract_frames.py \
        --repo-id Kronze157/astri_making_coffee_vlva \
        --data-root /data/astri_making_coffee_v1 --mode videos \
        --camera-idxes 0 1

    # 4) every 4th frame of each sub-task (capped at 50 per sub-task), head
    #    camera; the paired uint16 depth lands in depth_cam_head/ as .lz4
    python tools/astribot/extract_frames.py \
        --repo-id Kronze157/astri_making_coffee_vlva \
        --data-root /data/astri_making_coffee_v1 --mode frames \
        --camera-idxes 0 --interval 4 --max-frames 50
"""

import argparse
import json
import os
import subprocess
import warnings

import cv2
import numpy as np
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from tqdm import tqdm

from utils.astribot_dataloader import save_depth_lz4
from utils.visualize.visualize_mask import to_pil

# gripper values above this threshold count as 1 (closed), otherwise 0
# (open); a change between the two levels marks a key frame
KEY_FRAME_CHANGE = 20
# moving-average window (frames) applied to the gripper state before the
# binarization, to suppress sensor noise
KEY_FRAME_SMOOTH_WIN = 5
# runs of closed (1) shorter than this many seconds are sensor noise (a
# gripper that blips closed and re-opens) and are zeroed out before the
# key-frame change detection; default for --min-close-seconds
KEY_FRAME_MIN_CLOSE_S = 1.5
# output roots per mode, under <out_dir>: detect_subtask writes the
# subtask_splits.json + gripper plot, key_frames the first/last/key-frame
# jpgs, videos the per-subtask mp4s, frames the sampled per-subtask frames
# (+ a uint16 depth .lz4 per frame for cameras with a paired depth feature)
MODE_ROOTS = {"detect_subtask": "subtask",
              "key_frames": "key_frames",
              "videos": "subtask_videos",
              "frames": "subtask_frames"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract sub-task splits, key frames and/or videos from a "
                    "local LeRobotDataset copy."
    )
    parser.add_argument("--repo-id", "-id", required=True,
                        help="dataset repo id as seen by LeRobotDataset")
    parser.add_argument("--data-root", "-d", required=True,
                        help="root passed to LeRobotDataset; for chunked datasets give the "
                             "full chunk path, e.g. /data/x/chunk-0000/part-0000")
    parser.add_argument("--camera-idxes", "-c", nargs="+", type=int, default=[2, 1],
                        help="indices into the dataset's camera_keys")
    parser.add_argument("--mode", "-m", required=True,
                        choices=("detect_subtask", "key_frames", "videos", "frames"),
                        help="detect_subtask: detect the gripper key frames and infer the "
                             "sub-task split frames from observation.state only (no video "
                             "access), saving the gripper plot and subtask_splits.json; "
                             "key_frames: save one jpg per camera of the episode's first, "
                             "last and key frames plus the start/end frame of every "
                             "sub-task segment, each under the sub-task segment it "
                             "belongs to (from subtask_splits.json); "
                             "videos: one mp4 per sub-task segment per camera, "
                             "always frame-accurate (from subtask_splits.json); "
                             "frames: one jpg per --interval-th frame of each sub-task "
                             "(capped at --max-frames), plus a uint16 depth .lz4 per "
                             "frame for cameras with a paired uint16 depth feature")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="cap the sampled frames per sub-task "
                             "(default: all frames at --interval)")
    parser.add_argument("--interval", type=int, default=4,
                        help="sample every N-th frame of each sub-task segment "
                             "(default: %(default)s)")
    select = parser.add_mutually_exclusive_group()
    select.add_argument("--episode-idxes", "-e", nargs="*", type=int, default=None,
                        help="only process these episode indices (default: all)")
    select.add_argument("--one-per-task", action="store_true",
                        help="select the first episode of each task")
    parser.add_argument("--max-episodes", "-x", type=int, default=None,
                        help="cap the number of processed episodes")
    parser.add_argument("--out-dir", "-o", default=None,
                        help="output root (default: <data-root>/eps_data, with "
                             "subtask/, key_frames/ and subtask_videos/ "
                             "sub-folders)")
    parser.add_argument("--dedup-tasks", action="store_true",
                        help="skip episodes whose task already produced output")
    parser.add_argument("--use-inferred-splits", action="store_true",
                        help="prefer the sub-task split frames inferred by "
                             "detect_subtask (subtask_splits.json) over the "
                             "dataset's ground-truth subtask_index column "
                             "when both exist")
    parser.add_argument("--min-close-seconds", type=float,
                        default=KEY_FRAME_MIN_CLOSE_S,
                        help="runs of closed gripper state shorter than this "
                             "(seconds) are dropped as noise before key-frame "
                             "detection (default: %(default)s)")
    return parser.parse_args()


class DataExtract:
    """Extract sub-task splits, key-frame jpgs and/or per-subtask videos from
    a local LeRobotDataset copy.
    """

    def __init__(self, args):
        self.args = args
        self.ds_meta = LeRobotDatasetMetadata(repo_id=args.repo_id, root=args.data_root)
        self.dataset = None  # LeRobotDataset handle, created lazily by the modes that decode videos
        self.tasks = getattr(self.ds_meta, "tasks", None)
        self.subtasks = self._load_subtasks()
        self.out_dir = args.out_dir or (args.data_root + "/eps_data")
        self._features = getattr(self.ds_meta.info, "features", None)
        if self._features is None:
            self._features = getattr(self.ds_meta, "features", {})

        self.cam_keys = self._select_cameras()
        self.ep_idxes = self._select_episodes()
        self.cam_dirs = self._camera_dirs()
        self.fps = int(getattr(self.ds_meta, "fps", 30) or 30)
        self.gripper_idxes = None
        if args.mode == "detect_subtask":
            self._setup_gripper()
        self.done_tasks = set()

        # per-episode state, set by _begin_episode()
        self.ep_idx = self.from_idx = self.to_idx = self.task_id = None
        # print the meta data of the current extraction state
        self._print_meta_info()

    # --- dataset introspection -------------------------------------------------

    @staticmethod
    def _camera_subdir(cam_key):
        """Subdir per camera, named after the dataset's camera key."""
        return cam_key.rsplit(".", 1)[-1]

    @staticmethod
    def _to_gray_pil(img):
        """Grayscale (2D) PIL image of a frame. Depth streams are stored as
        color RGB; drop them to luminance (same conversion for jpgs and mp4s)."""
        arr = np.asarray(to_pil(img))
        if arr.ndim == 2:
            return to_pil(arr)
        return to_pil(cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY))

    @staticmethod
    def _state_feature(ds_meta):
        """The observation.state feature spec (dict-like), or None."""
        features = getattr(ds_meta.info, "features", None)
        if features is None and hasattr(ds_meta.info, "get"):
            features = ds_meta.info.get("features", {})
        return features.get("observation.state") if features else None

    @staticmethod
    def _state_feature_names(ds_meta):
        """Flat list of observation.state column names ([] when unavailable).
        Handles the dict spec ({'dtype', 'shape', 'names': [...]}), the
        plain-list layout, and 1-D states whose names are stored nested
        per-dimension: [['arm_left_j0', ..., 'gripper_right']]."""
        spec = DataExtract._state_feature(ds_meta)
        if spec is None:
            return []
        names = spec.get("names") if isinstance(spec, dict) else spec
        if names is None:
            return []
        names = list(names)
        # 1-D states may keep names as one inner list (len(names) == 1).
        if names and not isinstance(names[0], str):
            names = names[0] if len(names) == 1 else [n for sub in names for n in sub]
        return [str(n) for n in names]

    def _select_cameras(self):
        """Validate camera indices and map them to the dataset's camera keys."""
        idxes = self.args.camera_idxes
        for idx in idxes:
            if not 0 <= idx < len(self.ds_meta.camera_keys):
                raise ValueError(f"camera_idx {idx} out of range "
                                 f"(dataset has {len(self.ds_meta.camera_keys)} cameras)")
        return {idx: self.ds_meta.camera_keys[idx] for idx in idxes}

    def _select_episodes(self):
        """Episode indices to process: one per task, an explicit list, or all."""
        if self.args.one_per_task:
            ep_idxes, seen = [], set()
            for ep in range(self.ds_meta.total_episodes):
                tid = self._episode_task_id(ep)
                if tid not in seen:
                    seen.add(tid)
                    ep_idxes.append(ep)
        else:
            ep_idxes = self.args.episode_idxes
            if ep_idxes is None:
                ep_idxes = list(range(self.ds_meta.total_episodes))
        if self.args.max_episodes is not None:
            ep_idxes = ep_idxes[: self.args.max_episodes]
        return ep_idxes

    def _episode_col(self, ep_idx, col):
        """Value of an episode-table column for an episode, tolerating both
        the pandas-DataFrame and dict-of-lists episode tables."""
        eps = self.ds_meta.episodes
        if hasattr(eps, "columns"):  # pandas DataFrame
            return int(eps[col].iloc[ep_idx])
        return int(eps[col][ep_idx])  # dict of lists/arrays

    def _episode_bounds(self, ep_idx):
        """(dataset_from_index, dataset_to_index) of an episode; the to-index
        is exclusive (from + episode length), so an episode owns exactly
        [from, to)."""
        return (self._episode_col(ep_idx, "dataset_from_index"),
                self._episode_col(ep_idx, "dataset_to_index"))

    def _episode_task_id(self, ep_idx):
        """Task id of an episode: from the episode table when it has a
        task_index column, else from the per-frame data parquet, else from
        the episode's first frame (video decode)."""
        eps = self.ds_meta.episodes
        has_task = ("task_index" in eps.columns
                    if hasattr(eps, "columns") else "task_index" in eps)
        if has_task:
            return self._episode_col(ep_idx, "task_index")
        tid = self._task_index_from_parquet(ep_idx)
        if tid is not None:
            return tid
        from_idx, _ = self._episode_bounds(ep_idx)
        return int(self._ensure_dataset()[from_idx]["task_index"].item())

    def _load_subtasks(self):
        """Dataset subtask mapping {subtask_index: description} from
        meta/subtasks.parquet, or None when the dataset has none. The parquet
        index holds the descriptions and the subtask_index column the index,
        mirroring the tasks.parquet layout."""
        path = os.path.join(self.ds_meta.root, "meta", "subtasks.parquet")
        if not os.path.isfile(path):
            return None
        import pandas as pd

        df = pd.read_parquet(path)
        if not len(df) or "subtask_index" not in df.columns:
            return None
        return {int(idx): str(desc) for desc, idx in df["subtask_index"].items()}

    def _task_description(self, tid):
        """Task description for a task id, or '?' when unknown."""
        tasks = self.tasks
        if tasks is None:
            return "?"
        try:
            desc = tasks[tid]
            if isinstance(desc, str):
                return desc
        except (KeyError, IndexError, TypeError):
            pass
        # tasks may be a pandas DataFrame with one row per task; the task
        # name may live in a column or in the row index alone
        if hasattr(tasks, "columns"):
            try:
                rows = (tasks[tasks["index"] == tid]
                        if "index" in tasks.columns else tasks.iloc[[tid]])
                for col in ("task", "language_instruction"):
                    if col in rows.columns and len(rows):
                        desc = rows[col].iloc[0]
                        if isinstance(desc, str) and desc:
                            return desc
            except Exception:
                pass
            try:
                label = tasks.index[tid]
                if isinstance(label, str) and label:
                    return label
            except Exception:
                pass
        return "?"

    def _camera_dirs(self):
        """Per-camera subdir names, named after the dataset camera keys; the
        per-mode output layout lives under out_dir/<mode_root>/ep{ep}/ (see
        MODE_ROOTS and _episode_dir)."""
        cam_dirs = {idx: self._camera_subdir(cam_key)
                    for idx, cam_key in self.cam_keys.items()}
        os.makedirs(self.out_dir, exist_ok=True)
        return cam_dirs

    def _print_meta_info(self):
        """Print the dataset meta info: summary lines, the camera table, the
        observation.state table and the subtask table."""
        meta = self.ds_meta
        print(f"repo: {self.args.repo_id}")
        line = f"episodes: {meta.total_episodes} | frames: {meta.total_frames}"
        if getattr(meta, "fps", None):
            line += f" | fps: {meta.fps}"
        print(line)
        if self.tasks is not None:
            print(f"tasks: {len(self.tasks)}")
        if getattr(meta, "video_width", None) and getattr(meta, "video_height", None):
            print(f"video: {meta.video_width}x{meta.video_height}")
        self._print_camera_table()
        self._print_state_table()
        self._print_subtask_table()

    def _rich_note(self, text):
        """Print a yellow note line via rich (imported lazily)."""
        from rich.console import Console

        Console().print(f"[yellow]{text}[/yellow]")

    def _rich_table(self, title):
        """Console and a titled table for the meta tables (rich imported
        lazily)."""
        from rich.console import Console
        from rich.table import Table

        return Console(), Table(title=title, title_style="bold cyan")

    def _print_camera_table(self):
        """Available cameras as a rich table, marking the selected ones."""
        console, table = self._rich_table(
            f"cameras: {len(self.ds_meta.camera_keys)} available, "
            f"{len(self.cam_keys)} selected")
        selected = set(self.args.camera_idxes)
        table.add_column("index", justify="right", style="cyan")
        table.add_column("name", style="white")
        table.add_column("selected", justify="center")
        for i, key in enumerate(self.ds_meta.camera_keys):
            mark = "[green]yes[/green]" if i in selected else "no"
            table.add_row(str(i), key, mark)
        console.print(table)

    def _print_state_table(self):
        """Print the observation.state layout as a rich terminal table, one
        row per state column."""
        spec = self._state_feature(self.ds_meta)
        if spec is None:
            self._rich_note("note: no observation.state feature in dataset")
            return
        names = self._state_feature_names(self.ds_meta)
        shape = spec.get("shape") if isinstance(spec, dict) else None
        dtype = (spec.get("dtype", "") if isinstance(spec, dict) else "")

        title = f"observation.state: {len(names)} dims"
        if shape:
            title += f", shape {list(shape)}"
        if dtype:
            title += f", {dtype}"
        console, table = self._rich_table(title)
        table.add_column("index", justify="right", style="cyan")
        table.add_column("name", style="white")
        table.add_column("dtype", style="magenta")
        for i, name in enumerate(names):
            table.add_row(str(i), name, str(dtype) if dtype else "-")
        console.print(table)

    def _print_subtask_table(self):
        """Print the subtask index -> description mapping as a rich table."""
        if not self.subtasks:
            self._rich_note("note: no subtasks.parquet in dataset")
            return
        console, table = self._rich_table(f"subtasks: {len(self.subtasks)}")
        table.add_column("index", justify="right", style="cyan")
        table.add_column("subtask", style="white")
        for idx in sorted(self.subtasks):
            table.add_row(str(idx), self.subtasks[idx])
        console.print(table)

    def _setup_gripper(self):
        """Resolve the gripper columns of observation.state."""
        names = self._state_feature_names(self.ds_meta)
        self.gripper_idxes = [i for i, name in enumerate(names) if "gripper" in name.lower()]
        if not self.gripper_idxes:
            print("warning: no gripper columns in observation.state; "
                  "gripper plot skipped (split frames cannot be inferred)")

    def _ensure_dataset(self):
        """LeRobotDataset handle (videos on disk), created lazily: only the
        key_frames and frames modes decode videos (the videos mode cuts them
        with ffmpeg straight from disk)."""
        if self.dataset is None:
            self.dataset = LeRobotDataset(repo_id=self.args.repo_id,
                                          root=self.args.data_root,
                                          download_videos=False)
        return self.dataset

    # --- tabular (video-independent) episode data -------------------------------

    def _tabular_path(self, ep_idx):
        """Absolute path of the tabular parquet holding an episode's frames."""
        return os.path.join(self.ds_meta.root,
                            self.ds_meta.get_data_file_path(ep_idx))

    def _task_index_from_parquet(self, ep_idx):
        """The episode's task_index from the data parquet, or None when the
        dataset has no such column. Reads no video."""
        import pandas as pd
        import pyarrow.parquet as pq

        path = self._tabular_path(ep_idx)
        if "task_index" not in set(pq.read_schema(path).names):
            return None
        from_idx, to_idx = self._episode_bounds(ep_idx)
        df = pd.read_parquet(path, columns=["index", "task_index"])
        rows = df[(df["index"] >= from_idx) & (df["index"] < to_idx)]
        return int(rows["task_index"].iloc[0]) if len(rows) else None

    def _load_episode_table(self):
        """The current episode's rows from the tabular parquet (absolute index,
        task_index and subtask_index, plus observation.state when grippers are
        in play). Reads no video."""
        import pandas as pd
        import pyarrow.parquet as pq

        path = self._tabular_path(self.ep_idx)
        cols = ["index", "task_index", "subtask_index"]
        if self.gripper_idxes:
            cols += ["observation.state"]
        # datasets may lack optional columns (e.g. subtask_index); read only
        # the columns the schema actually has
        cols = [c for c in cols if c in set(pq.read_schema(path).names)]
        df = pd.read_parquet(path, columns=cols)
        return df[(df["index"] >= self.from_idx) & (df["index"] < self.to_idx)]

    def _episode_split_data(self, df):
        """(frame idxes, gripper values (T, n_grippers), subtask idxes) of the
        current episode; the values are None when the dataset has no gripper
        columns."""
        frames = df["index"].to_numpy()
        if not self.gripper_idxes:
            return frames, None, None
        states = np.stack(df["observation.state"].to_numpy())  # (T, n_dims)
        vals = states[:, self.gripper_idxes]
        subtasks = df["subtask_index"].to_numpy() if "subtask_index" in df.columns else None
        return frames, vals, subtasks

    # --- key-frame / split-frame detection --------------------------------------

    @staticmethod
    def _low_pass(vals):
        """Centered moving-average low-pass filter (window
        KEY_FRAME_SMOOTH_WIN) applied along the time axis of (T, n) values.
        The window shrinks at the borders and is renormalized, so a flat
        trace stays flat (a naive mode='same' convolution would attenuate
        the edges and invent key frames there)."""
        half = KEY_FRAME_SMOOTH_WIN // 2
        n = len(vals)
        smooth = np.empty_like(vals, dtype=np.float64)
        for t in range(n):
            smooth[t] = vals[max(0, t - half): min(n, t + half + 1)].mean(axis=0)
        return smooth

    def _suppress_short_closes(self, binary):
        """Zero out runs of closed (1) shorter than --min-close-seconds,
        per gripper column: a gripper that blips closed for a few frames
        and re-opens is sensor noise, not a real grasp, and would otherwise
        create a spurious key-frame pair (0 -> 1, 1 -> 0)."""
        min_frames = max(1, round(self.args.min_close_seconds * self.fps))
        out = binary.copy()
        for col in range(out.shape[1]):
            b = out[:, col]
            # run boundaries: diff of the zero-padded column is +1 at a run
            # start and -1 at its end; the pairs are (start, end)
            bounds = np.flatnonzero(np.diff(
                np.concatenate(([False], b, [False]))))
            for s, e in bounds.reshape(-1, 2):
                if e - s < min_frames:
                    b[s:e] = False
        return out

    def _key_frame_idxes(self, frames, vals, smooth=None):
        """Dataset indices of the episode's key frames: the first frame, the
        last frame, plus every frame where the (smoothed) gripper state
        crosses the KEY_FRAME_CHANGE threshold (0 -> 1 or 1 -> 0). Runs of
        closed state shorter than KEY_FRAME_MIN_CLOSE_S seconds are dropped
        first (see _suppress_short_closes), so a noisy blip does not produce
        a key-frame pair. Without gripper data, falls back to the start /
        middle / end frames. The low-passed signal may be passed in to avoid
        recomputing it."""
        if vals is None or not len(frames):
            return [self.from_idx, (self.from_idx + self.to_idx - 1) // 2,
                    self.to_idx - 1]
        if smooth is None:
            smooth = self._low_pass(vals)
        binary = smooth > KEY_FRAME_CHANGE  # (T, n_grippers) 0/1
        binary = self._suppress_short_closes(binary)
        key_idxes = [frames[0]]
        # a change from 0 to 1 or 1 to 0 marks the first frame of the new
        # level as a key frame. The smoothed crossing flips up to half a
        # window before the raw step (ramp), so pin the key frame to the
        # first raw frame that crosses the threshold, scanning within one
        # filter window of the transition (a slow drift cannot overshoot
        # into the next transition)
        changed = np.any(binary[1:] != binary[:-1], axis=1)
        for t in range(len(frames) - 1):
            if not changed[t]:
                continue
            start = max(1, t - KEY_FRAME_SMOOTH_WIN // 2)
            end = min(len(frames), t + 1 + KEY_FRAME_SMOOTH_WIN)
            prev = vals[start - 1] > KEY_FRAME_CHANGE
            t0 = None
            for k in range(start, end):
                cur = vals[k] > KEY_FRAME_CHANGE
                if np.any(cur != prev):
                    t0 = k
                    break
                prev = cur
            key_idxes.append(frames[t0 if t0 is not None else t + 1])
        if key_idxes[-1] != frames[-1]:
            key_idxes.append(frames[-1])
        return key_idxes

    def _subtask_split_idxes(self, frames, vals, smooth, key_idxes):
        """Split-frame indices between sub-tasks, inferred from the
        combined gripper state only (the dataset's subtask_index labels are
        ground truth for reference and are never used here).

        The combined binary state is the OR of the two gripper columns. A
        split is placed at the middle of the key-frame pair bounding an
        open phase of a complete 1 -> 0 -> 1 pattern (closed, released,
        re-closed): (k0 + k1) // 2, with k0 the 1 -> 0 toggle key frame and
        k1 the 0 -> 1 toggle key frame. The open phase at the episode start
        (no preceding 1) or end (no return to 1) yields no split. ``smooth``
        and ``key_idxes`` are the caller's already-computed analysis."""
        if vals is None or not len(frames):
            return []
        combined = np.any(vals > KEY_FRAME_CHANGE, axis=1)
        pos = np.searchsorted(frames, key_idxes)  # key frames are drawn from frames
        splits = []
        for p0, p1 in zip(pos, pos[1:]):
            # k0 opens (1 -> 0) and k1 re-closes (0 -> 1); the key frames
            # are pinned to the first raw crossing, so test the raw state
            if (p0 > 0 and combined[p0] == 0 and combined[p0 - 1] == 1
                    and combined[p1 - 1] == 0 and combined[p1] == 1):
                splits.append((int(frames[p0]) + int(frames[p1])) // 2)
        return splits

    # --- outputs ---------------------------------------------------------------

    def _episode_dir(self):
        """Per-episode output root of the current mode:
        out_dir/<mode_root>/ep{ep_idx:06d} (see MODE_ROOTS)."""
        return os.path.join(self.out_dir, MODE_ROOTS[self.args.mode],
                            f"ep{self.ep_idx:06d}")

    def _split_file_path(self):
        """subtask_splits.json of the current episode (written by
        detect_subtask, read by key_frames, videos and frames). Always lives
        under the subtask root, so the reading modes can find it without
        knowing the detect_subtask layout."""
        return os.path.join(self.out_dir, MODE_ROOTS["detect_subtask"],
                            f"ep{self.ep_idx:06d}", "subtask_splits.json")

    def _save_splits(self, key_idxes, split_idxes):
        """Record the detected key frames and the inferred sub-task split
        frames (absolute dataset indices) as subtask_splits.json."""
        data = {
            "episode": self.ep_idx,
            "task_id": self.task_id,
            "task": self._task_description(self.task_id),
            "from_idx": self.from_idx,
            "to_idx": self.to_idx,
            "key_frames": [int(k) for k in key_idxes],
            "split_frames": [int(s) for s in split_idxes],
        }
        path = self._split_file_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _split_frames_from_ground_truth(self):
        """Split frames from the dataset's ground-truth subtask_index
        column: the first frame of each new sub-task segment (every frame
        where the value changes). Returns None when the dataset has no
        subtask_index column, or when the episode has fewer than two
        sub-tasks (nothing to split)."""
        df = self._load_episode_table()
        if "subtask_index" not in df.columns:
            return None
        subtasks = df["subtask_index"].to_numpy()
        if len(np.unique(subtasks)) < 2:
            return None
        frames = df["index"].to_numpy()
        changes = np.flatnonzero(subtasks[1:] != subtasks[:-1])
        return [int(frames[t + 1]) for t in changes]

    def _load_splits_json(self):
        """The current episode's subtask_splits.json dict, raising a
        helpful error when the detect_subtask mode has not run yet."""
        path = self._split_file_path()
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"{path} missing: run --mode detect_subtask first "
                "(--dedup-tasks skips episodes whose task already has output)")
        with open(path) as f:
            return json.load(f)

    def _load_splits(self):
        """Split frames of the current episode, ground truth first: when
        the dataset stores a per-frame subtask_index column, they are
        collected at every frame where the subtask changes (see
        _split_frames_from_ground_truth). An episode must have at least two
        sub-tasks; otherwise the split frames are loaded from the
        subtask_splits.json written by the detect_subtask mode. With
        --use-inferred-splits the detect_subtask split frames always win —
        the ground-truth subtask_index labels are sometimes wrong."""
        if not self.args.use_inferred_splits:
            splits = self._split_frames_from_ground_truth()
            if splits is not None:
                return {"split_frames": splits}
        return self._load_splits_json()

    def _plot_gripper_state(self, frames, vals, subtask_idxes, smooth,
                            key_idxes, split_idxes):
        """Plot the per-frame gripper state columns, raw and low-pass
        filtered, mark the detected key frames with dashed vertical lines
        annotated with their frame index and the inferred sub-task split
        frames with solid red lines, and save one png. When the subtask
        index is available it is drawn as a second subplot below the gripper
        state: a step line at the raw integer values, not rescaled. ``smooth``,
        ``key_idxes`` and ``split_idxes`` are the caller's already-computed
        analysis."""
        import matplotlib
        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt

        names = [self._state_feature_names(self.ds_meta)[i]
                 for i in self.gripper_idxes]
        if subtask_idxes is not None:
            # two stacked subplots sharing the x axis: gripper state on top,
            # the subtask index below
            fig, (ax, ax_sub) = plt.subplots(
                2, 1, figsize=(10, 5.5), sharex=True,
                gridspec_kw={"height_ratios": [3, 1], "hspace": 0.12})
        else:
            fig, ax = plt.subplots(figsize=(10, 4))
        for i, name in enumerate(names):
            ax.plot(frames, vals[:, i], "--", color=f"C{i}", alpha=0.35,
                    label=f"{name} (raw)")
            ax.plot(frames, smooth[:, i], color=f"C{i}",
                    label=f"{name} (filtered)")
        # detected key frames: dashed vertical line, frame index at the top
        # edge of the axes (hanging down from ymax, ending at the line)
        _, ymax = ax.get_ylim()
        for k in key_idxes:
            ax.axvline(x=k, linestyle="--", color="gray", alpha=0.6,
                       linewidth=1)
            ax.text(k, ymax, str(k), rotation=90, ha="right", va="top",
                    fontsize=8, color="gray")
        # inferred sub-task split frames: solid red vertical line, split
        # position at the bottom edge (distinct from the key-frame lines)
        if split_idxes:
            ymin, _ = ax.get_ylim()
            for s in split_idxes:
                ax.axvline(x=s, color="red", linewidth=1.1)
                ax.text(s, ymin, f"{s:g}", rotation=90, ha="right",
                        va="bottom", fontsize=8, color="red")
        ax.set_ylabel("gripper state")
        ax.set_title(f"episode {self.ep_idx:06d} gripper state")
        if subtask_idxes is not None:
            # raw subtask index below: one step per frame (where="post"),
            # integer ticks and a tight y range so the values are shown
            # as-is instead of being rescaled
            ax_sub.step(frames, subtask_idxes, where="post", color="C2",
                        linewidth=1.2)
            ax_sub.set_ylabel("subtask index")
            ax_sub.set_yticks(sorted(set(subtask_idxes)))
            lo, hi = min(subtask_idxes), max(subtask_idxes)
            ax_sub.set_ylim(lo - 0.5, hi + 0.5)
            ax_sub.set_xlabel("frame index")
            # the inferred split frames, mirrored on the subtask subplot
            for s in split_idxes:
                ax_sub.axvline(x=s, color="red", linewidth=1.1)
        else:
            ax.set_xlabel("frame index")
        # legend outside the axes: an in-axes legend can collide with the
        # key-frame labels at the top edge
        ax.legend(bbox_to_anchor=(1.01, 1.0), loc="upper left",
                  borderaxespad=0.0)
        # the legend sits outside the axes (bbox_to_anchor), which tight_layout
        # cannot account for and warns about; the savefig below crops with
        # bbox_inches="tight", so the output includes the legend correctly
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r".*tight_layout.*")
            plt.tight_layout()
        edir = self._episode_dir()
        os.makedirs(edir, exist_ok=True)
        plt.savefig(os.path.join(edir, f"split_graph_{self.ep_idx:06d}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()

    def _load_key_frames(self):
        """Key frames of the current episode, detected by the detect_subtask
        mode (ground truth has no key frames — the gripper analysis is the
        only source)."""
        return self._load_splits_json()["key_frames"]

    def _save_frame_jpg(self, frame, cam_key, path):
        """Save one camera frame of a dataset row as a jpg. Depth streams
        are stored as color RGB in the dataset; they are saved grayscale
        (the same conversion as the mp4 cut)."""
        img = to_pil(frame[cam_key])
        if "depth" in cam_key:
            img = self._to_gray_pil(img)
        img.save(path)

    def _save_frames(self, frame_idxes, split_idxes):
        """Save one jpg per dataset index per selected camera, each under the
        sub-task segment the frame belongs to:
        <out_dir>/key_frames/<episode>/subtask_XX/<camera>/ (an episode
        without splits lands everything in subtask_00)."""
        base = self._episode_dir()
        # map each key-frame dataset index to its sub-task segment folder
        seg_of = {}
        for k, (lo, hi) in enumerate(self._segment_bounds(split_idxes)):
            for idx in frame_idxes:
                if lo <= idx < hi:
                    seg_of[idx] = f"subtask_{k:02d}"
        for seg in set(seg_of.values()):
            for ci, sub in self.cam_dirs.items():
                os.makedirs(os.path.join(base, seg, sub), exist_ok=True)
        ds = self._ensure_dataset()
        for frame_idx in frame_idxes:
            frame = ds[frame_idx]
            for ci, sub in self.cam_dirs.items():
                self._save_frame_jpg(frame, self.cam_keys[ci],
                                     os.path.join(base, seg_of[frame_idx],
                                                  sub,
                                                  f"frame_{frame_idx:06d}.jpg"))

    def _save_subtask_frames(self, split_idxes):
        """Sample every --interval-th frame of each sub-task segment (capped
        at --max-frames per sub-task) and save one jpg per selected camera
        under <out_dir>/subtask_frames/<episode>/subtask_XX/<camera>/. When a
        camera has a paired depth feature stored as uint16 (mm, see
        _depth_key_for), the raw depth array is saved alongside as
        frame_<idx>.lz4 (loadable by load_depth_lz4) under
        <out_dir>/subtask_frames/<episode>/subtask_XX/depth_<camera>/."""
        ds = self._ensure_dataset()
        # the depth pairing depends only on the camera key + dataset
        # features; resolve it once per camera, not per segment
        depth_keys = {ci: dkey for ci, key in self.cam_keys.items()
                      if (dkey := self._depth_key_for(key)) is not None
                      and self._feature(dkey).get("dtype") == "uint16"}
        for k, (lo, hi) in enumerate(self._segment_bounds(split_idxes)):
            end = hi if self.args.max_frames is None else min(
                hi, lo + self.args.max_frames * self.args.interval)
            steps = list(range(lo, end, self.args.interval))
            if not steps:
                print(f"  [subtask {k:02d}] skip: {hi - lo} frames at interval "
                      f"{self.args.interval} yield no steps")
                continue
            seg_dir = os.path.join(self._episode_dir(), f"subtask_{k:02d}")
            cam_dirs = {ci: os.path.join(seg_dir, sub)
                        for ci, sub in self.cam_dirs.items()}
            depth_dirs = {ci: os.path.join(seg_dir, f"depth_{self.cam_dirs[ci]}")
                          for ci in depth_keys}
            for d in list(cam_dirs.values()) + list(depth_dirs.values()):
                os.makedirs(d, exist_ok=True)
            for t in steps:
                frame = ds[t]
                for ci, sub in self.cam_dirs.items():
                    self._save_frame_jpg(frame, self.cam_keys[ci],
                                         os.path.join(cam_dirs[ci],
                                                      f"frame_{t:06d}.jpg"))
                for ci, dkey in depth_keys.items():
                    # raw uint16 mm depth as .lz4 (see save_depth_lz4)
                    save_depth_lz4(
                        np.asarray(frame[dkey]).astype(np.uint16),
                        os.path.join(depth_dirs[ci], f"frame_{t:06d}.lz4"),
                    )

    def _feature(self, key):
        """Feature spec dict for a dataset key ({} when the dataset has no
        such key)."""
        return self._features.get(key) or {}

    def _depth_key_for(self, cam_key):
        """Paired depth feature key for a camera key, or None.

        Prefers the raw observation.depth.<name> feature (uint16 mm, the
        astri_making_coffee_v1 layout). The legacy <cam_key>_depth video is
        only trusted when the dataset metadata flags it as a real depth map
        (video.is_depth_map) — the astri_making_coffee recording is flagged
        false and its video depth is unusable."""
        name = self._camera_subdir(cam_key)
        raw = f"observation.depth.{name}"
        if self._feature(raw):
            return raw
        video = f"{cam_key}_depth"
        if self._feature(video):
            info = self._feature(video).get("info") or {}
            if info.get("video.is_depth_map"):
                return video
            print(f"warning: {video} is flagged video.is_depth_map=false — "
                  "recorded depth unusable, depth output skipped")
        return None

    def _segment_bounds(self, split_idxes):
        """[(lo, hi)] segment bounds of the episode: the range split at the
        split frames ([from_idx] + split_idxes + [to_idx])."""
        bounds = [self.from_idx] + [int(s) for s in split_idxes] + [self.to_idx]
        return list(zip(bounds, bounds[1:]))

    def _write_subtask_videos(self, split_idxes):
        """One mp4 per sub-task segment per selected camera, cut straight
        from the episode's source videos with ffmpeg (no per-frame Python
        decode). Saved under <out_dir>/subtask_videos/<episode>/<camera>/;
        the dataset videos must be on disk (see _ensure_dataset).

        Always frame-accurate: each segment is cut with its own x264
        re-encode at the exact split frames — a decode+encode pass is
        unavoidable for exact H.264 cuts (stream copy can only snap to
        keyframes). Depth cameras are written as monochrome streams (the
        same grayscale convention as the jpg outputs).
        """
        vdir = self._episode_dir()
        eps = self.ds_meta.episodes
        for ci, sub in self.cam_dirs.items():
            os.makedirs(os.path.join(vdir, sub), exist_ok=True)
            cam = self.cam_keys[ci]
            prefix = f"videos/{cam}/"
            ch = int(eps[prefix + "chunk_index"][self.ep_idx])
            fi = int(eps[prefix + "file_index"][self.ep_idx])
            t0 = float(eps[prefix + "from_timestamp"][self.ep_idx])
            src = os.path.join(self.ds_meta.root, "videos", cam,
                               f"chunk-{ch:03d}", f"file-{fi:03d}.mp4")
            if not os.path.isfile(src):
                raise FileNotFoundError(
                    f"source video {src} missing "
                    "(download the dataset videos first)")
            vf = ["-vf", "format=gray"] if "depth" in cam else []
            for seg_i, (lo, hi) in enumerate(self._segment_bounds(split_idxes)):
                # frame i of the episode sits at from_timestamp + i/fps
                # (the source videos are CFR at the dataset fps); the input
                # seek decodes from the keyframe before the cut and discards
                # up to the exact frame
                start = t0 + (lo - self.from_idx) / self.fps
                dur = (hi - lo) / self.fps
                out = os.path.join(vdir, sub, f"subtask_{seg_i:02d}.mp4")
                r = subprocess.run(
                    ["ffmpeg", "-y", "-loglevel", "error",
                     "-ss", f"{start:.6f}", "-i", src,
                     "-t", f"{dur:.6f}", *vf, "-c:v", "libx264",
                     "-preset", "fast", "-crf", "18", "-an", out],
                    capture_output=True, text=True)
                if r.returncode != 0:
                    raise RuntimeError(
                        f"ffmpeg failed for {out}:\n{r.stderr.strip()}")

    # --- orchestration -----------------------------------------------------------

    def run(self):
        print(f"\n{len(self.ep_idxes)} episode(s) selected:")
        for ep in self.ep_idxes:
            tid = self._episode_task_id(ep)
            print(f"  episode {ep}: task {tid} ({self._task_description(tid)})")

        for ep_idx in tqdm(self.ep_idxes, desc="episodes"):
            self._process_episode(ep_idx)

        print(f"done: {len(self.ep_idxes)} episodes -> {self.out_dir}")

    def _begin_episode(self, ep_idx):
        """Set the per-episode state (ep_idx, dataset bounds, task id) and
        return the episode's tabular frame table."""
        self.ep_idx = ep_idx
        self.from_idx, self.to_idx = self._episode_bounds(ep_idx)
        df = self._load_episode_table()
        self.task_id = (int(df["task_index"].iloc[0])
                        if "task_index" in df.columns
                        else self._episode_task_id(ep_idx))
        return df

    def _process_episode(self, ep_idx):
        df = self._begin_episode(ep_idx)
        if self.args.dedup_tasks and self.task_id in self.done_tasks:
            return

        if self.args.mode == "detect_subtask":
            frames, vals, subtasks = self._episode_split_data(df)
            smooth = self._low_pass(vals) if vals is not None else None
            key_idxes = self._key_frame_idxes(frames, vals, smooth)
            split_idxes = self._subtask_split_idxes(frames, vals, smooth,
                                                    key_idxes)
            if self.gripper_idxes:
                self._plot_gripper_state(frames, vals, subtasks, smooth,
                                         key_idxes, split_idxes)
            self._save_splits(key_idxes, split_idxes)
        elif self.args.mode == "key_frames":
            # the start and end frame of every sub-task segment (the
            # boundaries under the chosen split source, ground-truth unless
            # --use-inferred-splits) plus the key frames detected by
            # detect_subtask, each saved under the sub-task segment it
            # belongs to (see _save_frames)
            splits = self._load_splits()["split_frames"]
            bounds = self._segment_bounds(splits)
            seg_frames = [lo for lo, _ in bounds] + [hi - 1 for _, hi in bounds]
            frame_idxes = sorted(set(seg_frames + self._load_key_frames()))
            self._save_frames(frame_idxes, splits)
        elif self.args.mode == "videos":
            self._write_subtask_videos(self._load_splits()["split_frames"])
        elif self.args.mode == "frames":
            self._save_subtask_frames(self._load_splits()["split_frames"])
        else:
            raise ValueError(f"unknown mode: {self.args.mode}")

        self.done_tasks.add(self.task_id)


def main():
    DataExtract(parse_args()).run()


if __name__ == "__main__":
    main()
