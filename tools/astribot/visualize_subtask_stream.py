"""Render the trajectory video of every sub-task segment of an episode,
online — the visualization counterpart of ``run_subtask_stream.py``.

Reuses ``DataExtract`` (tools/astribot/extract_frames.py) for dataset
introspection, episode/camera selection and sub-task split-frame inference
(identical flags to ``run_subtask_stream.py``), then for each sub-task
segment renders the streaming trajectory video from the saved pipeline
outputs: per-frame ``depth``/``extrinsics``/``intrinsics`` npz files under
``<out-dir>/pipeline/<episode>/subtask_XX/`` (the same contract as
``tools/general_test/visualize_stream.py``, which is invoked under the
hood) — but the colour images are **decoded online from the dataset** (one
frame per rendered step) instead of read from frame folders, so no extracted
frames or videos are needed on disk.

Each video frame shows that step's coloured point cloud with its camera
frustums and the growing camera path; the view is fixed per segment
(aligned to the first camera, fitted to the union of the segment's clouds).
Output: ``<seg_dir>/trajectory.mp4`` per segment.

Examples
--------
    # Render every sub-task of episode 0 (single camera, default fps/size)
    python tools/astribot/visualize_subtask_stream.py \
        --repo-id Kronze157/astri_making_coffee_vlva \
        --data-root /data/astri_making_coffee --episode-idxes 0

    # Stereo pair at 30 fps
    python tools/astribot/visualize_subtask_stream.py \
        --repo-id Kronze157/astri_making_coffee_vlva \
        --data-root /data/astri_making_coffee --episode-idxes 0 \
        --camera-idxes 4 5 --fps 30
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from lerobot.datasets import LeRobotDatasetMetadata
from tqdm import tqdm

from tools.astribot.extract_frames import DataExtract
from tools.general_test.visualize_stream import render_stream_video
from utils.streaming_utils import load_stream_data
from utils.visualize_mask import to_pil


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Render the trajectory video of every sub-task segment "
                    "of an episode, online (images decoded from the "
                    "LeRobotDataset; geometry from the saved pipeline NPZs)."
    )
    parser.add_argument("--repo-id", "-id", required=True,
                        help="dataset repo id as seen by LeRobotDataset")
    parser.add_argument("--data-root", "-d", required=True,
                        help="root passed to LeRobotDataset; for chunked datasets give the "
                             "full chunk path, e.g. /data/x/chunk-0000/part-0000")
    parser.add_argument("--camera-idxes", "-c", nargs="+", type=int, default=None,
                        help="indices into the dataset's camera_keys to visualize "
                             "(default: the first camera whose key does not contain "
                             "'depth'; must match the cameras used by run_subtask_stream)")
    select = parser.add_mutually_exclusive_group()
    select.add_argument("--episode-idxes", "-e", nargs="*", type=int, default=None,
                        help="only process these episode indices (default: all)")
    select.add_argument("--one-per-task", action="store_true",
                        help="select the first episode of each task")
    parser.add_argument("--max-episodes", "-x", type=int, default=None,
                        help="cap the number of processed episodes")
    parser.add_argument("--out-dir", "-o", default=None,
                        help="output root (default: <data-root>/eps_data); per-sub-task "
                             "results are read from <out-dir>/pipeline/<episode>/subtask_XX/")
    parser.add_argument("--stride", type=int, default=4,
                        help="subsample the sequence for the view fitting only; the "
                             "rendered steps are the npz stems actually saved "
                             "(default: %(default)s)")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--size", default="960x540",
                        help="video size WxH, e.g. 960x540 (default: 960x540)")
    parser.add_argument("--max-points", type=int, default=100_000,
                        help="max point-cloud points rendered per video frame "
                             "(default: 100000)")
    parser.add_argument(
        "--view-distance", type=float, default=0.3,
        help="eye distance behind the first camera, in scene-extent units "
             "(default: 0.3)",
    )
    parser.add_argument(
        "--views", type=int, choices=[1, 4], default=4,
        help="viewpoints per frame: 1 (center only) or 4 (2x2 grid of "
             "center/down/left/right) (default: 4)",
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
        help="backward pull of the down viewpoint, in scene-extent units — "
             "the camera pose falls outside the auto-fitted viewport when "
             "the eye is straight above it, so the eye is pulled back to "
             "bring the frustums/path into view (default: 0.3)",
    )
    parser.add_argument(
        "--view-fov", type=float, default=None,
        help="override the auto-fitted vertical field of view in degrees "
             "(default: auto-fit each viewport to the scene)",
    )
    return parser.parse_args(argv)


class SubtaskStreamVisualize(DataExtract):
    """Render one trajectory video per sub-task segment, online.

    Reuses DataExtract for dataset introspection, episode/camera selection
    and split-frame inference (same selection as ``run_subtask_stream.py``);
    the frames shown in the point clouds are decoded from the LeRobotDataset
    on the fly, the geometry comes from the saved pipeline npz files.
    """

    def __init__(self, args):
        if args.camera_idxes is None:
            keys = LeRobotDatasetMetadata(repo_id=args.repo_id,
                                          root=args.data_root).camera_keys
            args.camera_idxes = [i for i, key in enumerate(keys) if "depth" not in key][:1]
        args.mode = "videos"  # DataExtract needs one of its modes; only the
                             # output-dir/camera/episode machinery is reused
        super().__init__(args)
        self.pipeline_dir = os.path.join(self.out_dir, "pipeline")
        self.input_dirs = [self._camera_subdir(k) for k in self.cam_keys.values()]

    # --- frame access ---------------------------------------------------------

    def _dataset_step(self, t: int) -> dict:
        """Online decode of dataset frame t for all selected cameras, as BGR
        uint8 arrays (the visualizer flips to RGB, matching load_pair)."""
        frame = self._ensure_dataset()[t]
        return {key: np.asarray(to_pil(frame[key]), dtype=np.uint8)[:, :, ::-1]
                for key in self.cam_keys.values()}

    def _frame_loader(self, h: int, w: int):
        """Per-step RGB loader for render_stream_video: decodes the step's
        dataset frames (one per camera) and resizes them to the depth
        resolution, matching load_pair's image contract."""
        cam_keys = list(self.cam_keys.values())

        def load(stem: str) -> np.ndarray:
            t = int(stem.rsplit("_", 1)[-1])
            step = self._dataset_step(t)
            imgs = []
            for key in cam_keys:
                rgb = step[key][:, :, ::-1]
                if rgb.shape[:2] != (h, w):
                    rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
                imgs.append(rgb)
            return np.stack(imgs, axis=0)

        return load

    # --- orchestration ---------------------------------------------------------

    def run(self) -> None:
        print(f"\n{len(self.ep_idxes)} episode(s) selected:")
        for ep in self.ep_idxes:
            tid = self._episode_task_id(ep)
            print(f"  episode {ep}: task {tid} ({self._task_description(tid)})")
        for ep_idx in tqdm(self.ep_idxes, desc="episodes"):
            self._process_episode(ep_idx)
        print(f"\ndone: {len(self.ep_idxes)} episode(s) -> {self.pipeline_dir}")

    def _process_episode(self, ep_idx: int) -> None:
        self.ep_idx = ep_idx
        self.from_idx, self.to_idx = self._episode_bounds(ep_idx)
        splits = self._load_splits()["split_frames"]
        bounds = [self.from_idx] + [int(s) for s in splits] + [self.to_idx]
        print(f"\nepisode {ep_idx}: {len(bounds) - 1} sub-task segment(s)")
        for k, (lo, hi) in enumerate(zip(bounds, bounds[1:])):
            self._process_segment(k, lo, hi)

    def _process_segment(self, k: int, lo: int, hi: int) -> None:
        seg_dir = os.path.join(self.pipeline_dir, f"ep{self.ep_idx:06d}",
                               f"subtask_{k:02d}")
        npz_dir = os.path.join(seg_dir, f"depth_{Path(self.input_dirs[0]).name}")
        stems = sorted(p.stem for p in Path(npz_dir).glob("*.npz"))
        if not stems:
            print(f"  [subtask {k:02d}] skip: no npz results in {npz_dir}")
            return
        out_path = os.path.join(seg_dir, "trajectory.mp4")
        depth0, _, _ = load_stream_data(self.input_dirs[0], seg_dir, stems[0])
        h, w = depth0.shape
        print(f"  [subtask {k:02d}] frames [{lo}, {hi}) -> {len(stems)} steps "
              f"({npz_dir}) -> {out_path}")
        w_vid, h_vid = (int(x) for x in self.args.size.lower().split("x"))
        render_stream_video(
            stems,
            self.input_dirs,
            seg_dir,
            out_path,
            fps=self.args.fps,
            size=(w_vid, h_vid),
            max_points_per_frame=self.args.max_points,
            stride=self.args.stride,
            view_distance=self.args.view_distance,
            views=self.args.views,
            view_angle=self.args.view_angle,
            view_lower=self.args.view_lower,
            view_raise=self.args.view_raise,
            view_back=self.args.view_back,
            view_fov=self.args.view_fov,
            frame_loader=self._frame_loader(h, w),
        )


def main():
    SubtaskStreamVisualize(parse_args()).run()


if __name__ == "__main__":
    main()
