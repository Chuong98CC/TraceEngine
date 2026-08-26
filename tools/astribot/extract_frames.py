"""Extract key frames, videos and/or gripper-state plots from a local LeRobotDataset copy.

For each episode: saves key frames (the first and last frames, plus every
frame where the gripper state crosses the KEY_FRAME_CHANGE open/close
threshold, detected on the low-pass filtered signal), writes a re-encoded
mp4 per selected camera, and/or saves a gripper_left/right state plot that
also marks the detected key frames. The dataset videos must already be
present on disk (LeRobotDataset is opened with download_videos=False).

Examples
--------
    # key frames + videos + gripper plots for the first two cameras of every
    # episode; outputs go under <data-root>/eps_data/ep{ep}_task{task}/,
    # each with key_frames/<camera>/, videos/<camera>/ and gripper/
    python tools/astribot/extract_frames.py \
        --repo-id Kronze157/astri_making_coffee_vlva \
        --data-root /data/astri_making_coffee --camera-idxes 0 1 --mode all

    # videos only, for specific episodes of a chunked dataset
    # (--data-root is then the full chunk path)
    python tools/astribot/extract_frames.py \
        --repo-id simple-world-lab/HiFi-UMI-2K \
        --data-root /data/HiFi-UMI-2K/chunk-0000/part-0000 \
        --episode-idxes 3 7 --mode video

    # gripper plot only (no videos, no key frames)
    python tools/astribot/extract_frames.py \
        --repo-id Kronze157/astri_making_coffee_vlva \
        --data-root /data/astri_making_coffee --mode plot_gripper_state

    # one episode per task (first episode of each task)
    python tools/astribot/extract_frames.py \
        --repo-id Kronze157/astri_making_coffee_vlva \
        --data-root /data/astri_making_coffee --one-per-task
"""

import argparse
import os

import cv2
import numpy as np
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from tqdm import tqdm

from utils.visualize_mask import to_pil

# gripper values above this threshold count as 1 (closed), otherwise 0
# (open); a change between the two levels marks a key frame
KEY_FRAME_CHANGE = 20
# moving-average window (frames) applied to the gripper state before the
# binarization, to suppress sensor noise
KEY_FRAME_SMOOTH_WIN = 5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract key frames and/or videos from a local LeRobotDataset."
    )
    parser.add_argument("--repo-id", "-id",required=True,
                        help="dataset repo id as seen by LeRobotDataset")
    parser.add_argument("--data-root", "-d",required=True,
                        help="root passed to LeRobotDataset; for chunked datasets give the "
                             "full chunk path, e.g. /data/x/chunk-0000/part-0000")
    parser.add_argument("--camera-idxes", "-c", nargs="+", type=int, default=[2, 1],
                        help="indices into the dataset's camera_keys (default: 0 1)")
    parser.add_argument("--mode", "-m", choices=("frames", "video", "plot_gripper_state", "all"),
                        default="all",
                        help="frames: save the key-frame jpgs (gripper state "
                             "crossing the KEY_FRAME_CHANGE open/close "
                             "threshold) plus the gripper plot; "
                             "video: one mp4 per episode and camera; "
                             "plot_gripper_state: gripper state plot per episode; "
                             "all (default): video + frames + plot")
    select = parser.add_mutually_exclusive_group()
    select.add_argument("--episode-idxes", "-e", nargs="*", type=int, default=None,
                        help="only process these episode indices (default: all)")
    select.add_argument("--one-per-task", action="store_true",
                        help="select the first episode of each task")
    parser.add_argument("--max-episodes", "-x", type=int, default=None,
                        help="cap the number of processed episodes")
    parser.add_argument("--out-dir", "-o", default=None,
                        help="output root (default: <data-root>/eps_data, with "
                             "key_frames/ and videos/ sub-folders)")
    parser.add_argument("--dedup-tasks", action="store_true",
                        help="skip episodes whose task already produced output")
    return parser.parse_args()


class DataExtract:
    """Extract key frames, videos and/or gripper-state plots from a local
    LeRobotDataset copy.
    """

    def __init__(self, args):
        self.args = args
        self.ds_meta = LeRobotDatasetMetadata(repo_id=args.repo_id, root=args.data_root)
        self.dataset = LeRobotDataset(repo_id=args.repo_id, root=args.data_root,
                                      download_videos=False)
        self.tasks = getattr(self.ds_meta, "tasks", None)
        self.out_dir = args.out_dir or (args.data_root + "/eps_data")

        self.cam_keys = self._select_cameras()
        self.ep_idxes = self._select_episodes()
        self.frame_dirs, self.video_dirs = self._output_dirs()
        self.fps = int(getattr(self.ds_meta, "fps", 30) or 30)
        self.gripper_idxes = self.gripper_names = None
        if self._want_plot():
            self._setup_gripper()
        self.done_tasks = set()

        # per-episode state, set by _process_episode()
        self.ep_idx = self.from_idx = self.to_idx = self.middle_idx = self.task_id = None
        # print the meta data of the current extraction state
        self._print_meta_info()

    # --- dataset introspection -------------------------------------------------

    @staticmethod
    def _camera_subdir(cam_key):
        """Subdir per camera, named after the dataset's camera key."""
        return cam_key.rsplit(".", 1)[-1]

    @staticmethod
    def _to_bgr(img):
        # PIL is RGB; cv2.VideoWriter expects BGR.
        return np.asarray(to_pil(img))[:, :, ::-1]

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

    def _episode_bounds(self, ep_idx):
        """(dataset_from_index, dataset_to_index) of an episode; tolerates both
        the pandas-DataFrame and dict episode tables."""
        eps = self.ds_meta.episodes
        if hasattr(eps, "columns"):  # pandas DataFrame
            from_idx = int(eps["dataset_from_index"].iloc[ep_idx])
            to_idx = int(eps["dataset_to_index"].iloc[ep_idx])
        else:  # dict of lists/arrays
            from_idx = int(eps["dataset_from_index"][ep_idx])
            to_idx = int(eps["dataset_to_index"][ep_idx])
        return from_idx, to_idx

    def _episode_task_id(self, ep_idx):
        """Task id of an episode; falls back to the episode's first frame when
        the episode table has no task_index column."""
        eps = self.ds_meta.episodes
        if hasattr(eps, "columns"):  # pandas DataFrame
            if "task_index" in eps.columns:
                return int(eps["task_index"].iloc[ep_idx])
            from_idx = int(eps["dataset_from_index"].iloc[ep_idx])
        else:  # dict of lists/arrays
            if "task_index" in eps:
                return int(eps["task_index"][ep_idx])
            from_idx = int(eps["dataset_from_index"][ep_idx])
        return int(self.dataset[from_idx]["task_index"].item())

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
        # tasks may be a pandas DataFrame with one row per task
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
        return "?"

    def _output_dirs(self):
        """Per-camera subdir names for key frames and videos. Both live under
        the per-episode dir out_dir/ep{ep}_task{task}/ (see _episode_dir)."""
        frame_dirs, video_dirs = {}, {}
        for idx, cam_key in self.cam_keys.items():
            sub = self._camera_subdir(cam_key)
            if self.args.mode in ("frames", "all"):
                frame_dirs[idx] = sub
            if self.args.mode in ("video", "all"):
                video_dirs[idx] = sub
        os.makedirs(self.out_dir, exist_ok=True)
        return frame_dirs, video_dirs

    def _want_plot(self):
        return self.args.mode in ("plot_gripper_state", "frames", "all")

    def _print_meta_info(self):
        """Print the dataset meta info: summary lines, the camera table and
        the observation.state table."""
        meta = self.ds_meta
        print(f"repo: {self.args.repo_id}")
        print(f"episodes: {meta.total_episodes} | frames: {meta.total_frames}"
              + (f" | fps: {meta.fps}" if getattr(meta, "fps", None) else ""))
        if self.tasks is not None:
            print(f"tasks: {len(self.tasks)}")
        if getattr(meta, "video_width", None) and getattr(meta, "video_height", None):
            print(f"video: {meta.video_width}x{meta.video_height}")
        self._print_camera_table()
        self._print_state_table()

    def _print_camera_table(self):
        """Available cameras as a rich table, marking the selected ones."""
        from rich.console import Console
        from rich.table import Table

        console = Console()
        selected = set(self.args.camera_idxes)
        table = Table(title=f"cameras: {len(self.ds_meta.camera_keys)} available, "
                            f"{len(self.cam_keys)} selected", title_style="bold cyan")
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
        from rich.console import Console
        from rich.table import Table

        console = Console()
        spec = self._state_feature(self.ds_meta)
        if spec is None:
            console.print("[yellow]note: no observation.state feature in dataset[/yellow]")
            return
        names = self._state_feature_names(self.ds_meta)
        shape = spec.get("shape") if isinstance(spec, dict) else None
        dtype = (spec.get("dtype", "") if isinstance(spec, dict) else "")

        title = f"observation.state: {len(names)} dims"
        if shape:
            title += f", shape {list(shape)}"
        if dtype:
            title += f", {dtype}"
        table = Table(title=title, title_style="bold cyan")
        table.add_column("index", justify="right", style="cyan")
        table.add_column("name", style="white")
        table.add_column("dtype", style="magenta")
        for i, name in enumerate(names):
            table.add_row(str(i), name, str(dtype) if dtype else "-")
        console.print(table)

    def _setup_gripper(self):
        """Resolve the gripper columns of observation.state."""
        names = self._state_feature_names(self.ds_meta)
        self.gripper_idxes = [i for i, name in enumerate(names) if "gripper" in name.lower()]
        self.gripper_names = [names[i] for i in self.gripper_idxes]
        if not self.gripper_idxes:
            print("warning: no gripper columns in observation.state; "
                  "gripper plot skipped")

    def _get_gripper_state(self, frame):
        """Gripper state columns of one frame (left/right in dataset order)."""
        return frame["observation.state"][self.gripper_idxes]

    # --- output helpers ---------------------------------------------------------

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

    def _key_frame_idxes(self, states):
        """Dataset indices of the episode's key frames: the first frame, the
        last frame, plus every frame where the (smoothed) gripper state
        crosses the KEY_FRAME_CHANGE threshold (0 -> 1 or 1 -> 0). Without
        gripper data, falls back to the start / middle / end frames."""
        if not self.gripper_idxes or not states:
            return [self.from_idx, self.middle_idx, self.to_idx]
        frames = [f for f, _ in states]
        vals = np.stack([s.detach().cpu().numpy() if hasattr(s, "detach") else np.asarray(s)
                         for _, s in states])  # (T, n_grippers)
        smooth = self._low_pass(vals)
        binary = smooth > KEY_FRAME_CHANGE  # (T, n_grippers) 0/1
        key_idxes = [frames[0]]
        # a change from 0 to 1 or 1 to 0 marks the first frame of the new
        # level as a key frame; the smoothed ramp crosses the threshold up
        # to half a window before the raw step, so pin the key frame to the
        # first raw frame on the new level, bounded by the window so a slow
        # drift cannot overshoot into the next transition
        changed = np.any(binary[1:] != binary[:-1], axis=1)
        for t in range(len(frames) - 1):
            if not changed[t]:
                continue
            # pin the key frame to the first raw frame that crosses the
            # threshold: the smoothed binary may flip before the raw step
            # (ramp), so scan the raw within one filter window around the
            # transition and emit the first crossing frame
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

    def _episode_dir(self):
        """Per-episode output root: out_dir/ep{ep_idx}_task{task_id}."""
        return os.path.join(self.out_dir, f"ep{self.ep_idx:06d}_task{self.task_id}")

    def _save_key_frames(self, key_idxes):
        """Save the key frames as one jpg per selected camera under
        <episode_dir>/key_frames/<camera>/."""
        base = os.path.join(self._episode_dir(), "key_frames")
        for ci, sub in self.frame_dirs.items():
            os.makedirs(os.path.join(base, sub), exist_ok=True)
        for frame_idx in key_idxes:
            frame = self.dataset[frame_idx]
            for ci, sub in self.frame_dirs.items():
                name = f"frame_{frame_idx:06d}.jpg"
                to_pil(frame[self.cam_keys[ci]]).save(os.path.join(base, sub, name))

    def _stream_episode(self, first_frame):
        """Single pass: stream the episode into the video writers, collect
        the gripper states, then save the key frames and the plot from them."""
        first_cam = next(iter(self.cam_keys.values()))
        h, w = first_frame[first_cam].shape[-2:]
        vdir = os.path.join(self._episode_dir(), "videos")
        writers = {}
        for ci, sub in self.video_dirs.items():
            os.makedirs(os.path.join(vdir, sub), exist_ok=True)
            path = os.path.join(vdir, sub, f"episode_{self.ep_idx:06d}.mp4")
            writers[ci] = cv2.VideoWriter(
                path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
        states = []
        for frame_idx in range(self.from_idx, self.to_idx + 1):
            frame = self.dataset[frame_idx]
            for ci, writer in writers.items():
                writer.write(self._to_bgr(frame[self.cam_keys[ci]]))
            if self.gripper_idxes:
                states.append((frame_idx, self._get_gripper_state(frame)))
        for writer in writers.values():
            writer.release()
        if self.frame_dirs:
            self._save_key_frames(self._key_frame_idxes(states))
        if self.gripper_idxes and states:
            self._plot_gripper_state(states)

    def _collect_gripper_states(self):
        """Per-frame gripper states of the current episode."""
        return [(i, self._get_gripper_state(self.dataset[i]))
                for i in range(self.from_idx, self.to_idx + 1)]

    def _plot_gripper_state(self, states):
        """Plot the per-frame gripper state columns, raw and low-pass
        filtered, mark the detected key frames with dashed vertical lines
        annotated with their frame index, and save one png."""
        import matplotlib
        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt

        frames = [f for f, _ in states]
        arrays = [s.detach().cpu().numpy() if hasattr(s, "detach") else np.asarray(s)
                  for _, s in states]
        stacked = np.stack(arrays)  # (T, n_grippers)
        smooth = self._low_pass(stacked)
        plt.figure(figsize=(10, 4))
        for i, name in enumerate(self.gripper_names or range(stacked.shape[1])):
            plt.plot(frames, stacked[:, i], "--", color=f"C{i}", alpha=0.35,
                     label=f"{name} (raw)")
            plt.plot(frames, smooth[:, i], color=f"C{i}",
                     label=f"{name} (filtered)")
        # detected key frames: dashed vertical line, frame index at the top
        # edge of the axes (hanging down from ymax, ending at the line)
        ymin, ymax = plt.ylim()
        for k in self._key_frame_idxes(states):
            plt.axvline(x=k, linestyle="--", color="gray", alpha=0.6,
                        linewidth=1)
            plt.text(k, ymax, str(k), rotation=90, ha="right", va="top",
                     fontsize=8, color="gray")
        plt.xlabel("frame index")
        plt.ylabel("gripper state")
        plt.title(f"episode {self.ep_idx:06d} gripper state")
        # legend outside the axes: an in-axes legend can collide with the
        # key-frame labels at the top edge
        plt.legend(bbox_to_anchor=(1.01, 1.0), loc="upper left",
                   borderaxespad=0.0)
        plt.tight_layout()
        gdir = os.path.join(self._episode_dir(), "gripper")
        os.makedirs(gdir, exist_ok=True)
        plt.savefig(os.path.join(gdir, f"episode_{self.ep_idx:06d}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()

    # --- orchestration -----------------------------------------------------------

    def run(self):
        print(f"\n{len(self.ep_idxes)} episode(s) selected:")
        for ep in self.ep_idxes:
            tid = self._episode_task_id(ep)
            print(f"  episode {ep}: task {tid} ({self._task_description(tid)})")

        for ep_idx in tqdm(self.ep_idxes, desc="episodes"):
            self._process_episode(ep_idx)

        print(f"done: {len(self.ep_idxes)} episodes -> {self.out_dir}")

    def _process_episode(self, ep_idx):
        self.ep_idx = ep_idx
        self.from_idx, self.to_idx = self._episode_bounds(ep_idx)
        self.middle_idx = (self.from_idx + self.to_idx) // 2
        first_frame = self.dataset[self.from_idx]
        self.task_id = int(first_frame["task_index"].item())
        if self.args.dedup_tasks and self.task_id in self.done_tasks:
            return

        if self.video_dirs:
            self._stream_episode(first_frame)
        else:
            # Frames and/or plot: one pass over the episode collects the
            # gripper states the key frames and the plot are derived from.
            states = self._collect_gripper_states() if self.gripper_idxes else []
            if self.frame_dirs:
                self._save_key_frames(self._key_frame_idxes(states))
            if self.gripper_idxes:
                self._plot_gripper_state(states)

        self.done_tasks.add(self.task_id)


def main():
    DataExtract(parse_args()).run()


if __name__ == "__main__":
    main()
