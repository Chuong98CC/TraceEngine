"""Extract key frames and/or videos from a local LeRobotDataset copy.

For each episode: saves key frames (start / middle / last) and/or writes a
re-encoded mp4 per selected camera. The dataset videos must already be present
on disk (LeRobotDataset is opened with download_videos=False).

Examples
--------
    # key frames + videos for the first two cameras of every episode;
    # outputs go to <data-root>/eps_frames/ and <data-root>/eps_videos/
    python tools/astribot/extract_frames.py \
        --repo-id Kronze157/astri_making_coffee_vlva \
        --data-root /data/astri_making_coffee --camera-idxes 0 1 --mode both

    # videos only, for specific episodes of a chunked dataset
    # (--data-root is then the full chunk path)
    python tools/astribot/extract_frames.py \
        --repo-id simple-world-lab/HiFi-UMI-2K \
        --data-root /data/HiFi-UMI-2K/chunk-0000/part-0000 \
        --episode-idxes 3 7 --mode video

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
    parser.add_argument("--mode", "-m", choices=("frames", "video", "both"), default="both",
                        help="frames: save start/middle/last key-frame jpgs; "
                             "video: one mp4 per episode and camera")
    select = parser.add_mutually_exclusive_group()
    select.add_argument("--episode-idxes", "-e", nargs="*", type=int, default=None,
                        help="only process these episode indices (default: all)")
    select.add_argument("--one-per-task", action="store_true",
                        help="select the first episode of each task")
    parser.add_argument("--max-episodes", "-x", type=int, default=None,
                        help="cap the number of processed episodes")
    parser.add_argument("--out-dir", "-o", default=None,
                        help="output root (default: <data-root>, with eps_frames/ and "
                             "eps_videos/ sub-folders)")
    parser.add_argument("--dedup-tasks", action="store_true",
                        help="skip episodes whose task already produced output")
    return parser.parse_args()


def _camera_subdir(cam_key):
    """Subdir per camera, named after the dataset's camera key."""
    return cam_key.rsplit(".", 1)[-1]


def _to_bgr(img):
    # PIL is RGB; cv2.VideoWriter expects BGR.
    return np.asarray(to_pil(img))[:, :, ::-1]


def main():
    args = parse_args()

    ds_meta = LeRobotDatasetMetadata(repo_id=args.repo_id, root=args.data_root)
    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.data_root,
                             download_videos=False)

    print(f"repo: {args.repo_id}")
    print(f"episodes: {ds_meta.total_episodes} | frames: {ds_meta.total_frames}"
          + (f" | fps: {ds_meta.fps}" if getattr(ds_meta, "fps", None) else ""))
    tasks = getattr(ds_meta, "tasks", None)
    if tasks is not None:
        print(f"tasks: {len(tasks)}")
    if getattr(ds_meta, "video_width", None) and getattr(ds_meta, "video_height", None):
        print(f"video: {ds_meta.video_width}x{ds_meta.video_height}")
    print("available cameras:")
    for i, key in enumerate(ds_meta.camera_keys):
        print(f"  [{i}] {key}")
    print("selected cameras:",
          [ds_meta.camera_keys[i] for i in args.camera_idxes])

    for idx in args.camera_idxes:
        if not 0 <= idx < len(ds_meta.camera_keys):
            raise ValueError(f"camera_idx {idx} out of range "
                             f"(dataset has {len(ds_meta.camera_keys)} cameras)")
    cam_keys = {idx: ds_meta.camera_keys[idx] for idx in args.camera_idxes}

    def episode_task_id(ep_idx):
        eps = ds_meta.episodes
        if hasattr(eps, "columns"):  # pandas DataFrame
            if "task_index" in eps.columns:
                return int(eps["task_index"].iloc[ep_idx])
            from_idx = int(eps["dataset_from_index"].iloc[ep_idx])
        else:  # dict of lists/arrays
            if "task_index" in eps:
                return int(eps["task_index"][ep_idx])
            from_idx = int(eps["dataset_from_index"][ep_idx])
        return int(dataset[from_idx]["task_index"].item())

    def task_description(tid):
        """Task description for a task id, or '?' when unknown."""
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

    if args.one_per_task:
        ep_idxes, seen = [], set()
        for ep in range(ds_meta.total_episodes):
            tid = episode_task_id(ep)
            if tid not in seen:
                seen.add(tid)
                ep_idxes.append(ep)
    else:
        ep_idxes = args.episode_idxes
        if ep_idxes is None:
            ep_idxes = list(range(ds_meta.total_episodes))
    if args.max_episodes is not None:
        ep_idxes = ep_idxes[: args.max_episodes]

    print(f"\n{len(ep_idxes)} episode(s) selected:")
    for ep in ep_idxes:
        tid = episode_task_id(ep)
        print(f"  episode {ep}: task {tid} ({task_description(tid)})")

    want_frames = args.mode in ("frames", "both")
    want_video = args.mode in ("video", "both")

    out_dir = args.out_dir or args.data_root
    frame_dirs = {}
    video_dirs = {}
    for idx, cam_key in cam_keys.items():
        sub = _camera_subdir(cam_key)
        if want_frames:
            frame_dirs[idx] = os.path.join(out_dir, "eps_frames", sub)
        if want_video:
            video_dirs[idx] = os.path.join(out_dir, "eps_videos", sub)
    for d in list(frame_dirs.values()) + list(video_dirs.values()):
        os.makedirs(d, exist_ok=True)

    fps = int(getattr(ds_meta, "fps", 30) or 30)
    done_tasks = set()

    for ep_idx in tqdm(ep_idxes, desc="episodes"):
        eps = ds_meta.episodes
        if hasattr(eps, "columns"):  # pandas DataFrame
            from_idx = int(eps["dataset_from_index"].iloc[ep_idx])
            to_idx = int(eps["dataset_to_index"].iloc[ep_idx])
        else:  # dict of lists/arrays
            from_idx = int(eps["dataset_from_index"][ep_idx])
            to_idx = int(eps["dataset_to_index"][ep_idx])
        middle_idx = (from_idx + to_idx) // 2

        first_frame = dataset[from_idx]
        task_id = int(first_frame["task_index"].item())
        if args.dedup_tasks and task_id in done_tasks:
            continue

        def save_key_frame(frame, frame_idx, pos):
            for ci, d in frame_dirs.items():
                name = f"ep{ep_idx:06d}_task{task_id}_frame_{frame_idx:06d}_{pos}.jpg"
                to_pil(frame[cam_keys[ci]]).save(os.path.join(d, name))

        if video_dirs:
            # Single pass: stream the episode into the writers and pick the
            # key frames along the way.
            h, w = first_frame[cam_keys[args.camera_idxes[0]]].shape[-2:]
            writers = {}
            for ci, d in video_dirs.items():
                path = os.path.join(d, f"episode_{ep_idx:06d}.mp4")
                writers[ci] = cv2.VideoWriter(
                    path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            for frame_idx in range(from_idx, to_idx + 1):
                frame = dataset[frame_idx]
                for ci, writer in writers.items():
                    writer.write(_to_bgr(frame[cam_keys[ci]]))
                if frame_dirs and frame_idx in (from_idx, middle_idx, to_idx):
                    pos = {from_idx: "start", middle_idx: "middle", to_idx: "end"}[frame_idx]
                    save_key_frame(frame, frame_idx, pos)
            for writer in writers.values():
                writer.release()
        else:
            # Frames only: fetch just the three key frames.
            for frame_idx, pos in ((from_idx, "start"),
                                   (middle_idx, "middle"),
                                   (to_idx, "end")):
                save_key_frame(dataset[frame_idx], frame_idx, pos)

        done_tasks.add(task_id)

    print(f"done: {len(ep_idxes)} episodes -> {out_dir}")


if __name__ == "__main__":
    main()
