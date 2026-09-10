"""Per-(sub-task, camera) depth+pose videos, online — the visualization
counterpart of run_step2_depth_stream.py.

Episodes, sub-task segments, cameras and rendered steps are discovered
from the saved Step-2 outputs themselves — <out-dir>/depth_pose/ep*/
subtask_XX/depth_<camera>/frame_<abs-idx>.npz — the on-disk segmentation:
no dataset split inference, no subtask_splits.json, no selection flag
that must match the Step-2 run; ``-e``/``-c`` only filter what exists on
disk. For each (sub-task, selected camera) with outputs it renders one
``depth_pose.mp4`` (per step the coloured point cloud with the camera's
frustum and growing path, the view fixed per segment), reusing
render_stream_video of tools/general_test/pipeline/visualize_stream.py
(geometry from the saved depth_pose npz files; colour frames decoded
online from the LeRobotDataset — no extracted frames or videos needed on
disk). Output:
<out-dir>/visualization/<episode>/subtask_XX/<camera>/depth_pose.mp4.

Examples
--------
    # Render every sub-task of episode 0 (every camera with Step-2 outputs)
    python tools/astribot/visualize_step2_depth_pose.py
        --repo-id Kronze157/astribot_making_coffee_vlva_full
        --data-root /data/astri_making_coffee --episode-idxes 0

    # Stereo pair at 30 fps (one depth_pose.mp4 per camera)
    python tools/astribot/visualize_step2_depth_pose.py
        --repo-id Kronze157/astribot_making_coffee_vlva_full
        --data-root /data/astri_making_coffee --episode-idxes 0
        --camera-idxes 4 5 --fps 30
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from tqdm import tqdm

from tools.general_test.pipeline.visualize_stream import render_stream_video
from utils.streaming_utils import load_stream_data
from utils.visualize.visualize_mask import to_pil


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Render the depth+pose video (depth_pose.mp4) of every "
                    "selected camera of each sub-task segment of an episode, "
                    "online (images decoded from the LeRobotDataset; "
                    "geometry from the saved depth_pose NPZs). Episodes, "
                    "sub-task segments, cameras and steps are discovered "
                    "from the Step-2 outputs on disk — no split inference, "
                    "no selection flag that must match the Step-2 run; "
                    "-e/-c only filter what exists."
    )
    parser.add_argument("--repo-id", "-id", required=True,
                        help="dataset repo id as seen by LeRobotDataset")
    parser.add_argument("--data-root", "-d", required=True,
                        help="root passed to LeRobotDataset; for chunked datasets give the "
                             "full chunk path, e.g. /data/x/chunk-0000/part-0000")
    parser.add_argument("--camera-idxes", "-c", nargs="+", type=int, default=None,
                        help="indices into the dataset's camera_keys to visualize "
                             "(default: every camera that has Step-2 outputs on disk)")
    parser.add_argument("--episode-idxes", "-e", nargs="*", type=int, default=None,
                        help="only visualize these episode indices (default: every "
                             "episode with Step-2 outputs under <out-dir>/depth_pose)")
    parser.add_argument("--max-episodes", "-x", type=int, default=None,
                        help="cap the number of discovered episodes (first N, after "
                             "any -e filter)")
    parser.add_argument("--out-dir", "-o", default=None,
                        help="output root (default: <data-root>/eps_data); per-sub-task "
                             "results are read from <out-dir>/depth_pose/<episode>/subtask_XX/ "
                             "and the videos are written under <out-dir>/visualization/")
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


class SubtaskStreamVisualize:
    """Render one depth_pose.mp4 per (sub-task, selected camera), online.

    Standalone: the episodes, sub-task segments, cameras and rendered
    steps are all discovered from the saved Step-2 outputs on disk
    (depth_pose/<ep>/subtask_XX/depth_<cam>) — never recomputed from
    dataset splits, so nothing has to match the Step-2 run. The dataset
    is only consulted for camera_keys (decoding the -c indices and
    mapping the on-disk camera subdirs back to decode keys) and for the
    online frame decode; the geometry comes from the saved depth_pose
    npz files, each camera's video rendered over its own saved stems.
    """

    def __init__(self, args):
        self.args = args
        # dataset metadata for camera_keys only — episodes/segments come
        # from the disk (self.depth_pose_dir), not from the dataset
        self.ds_meta = LeRobotDatasetMetadata(repo_id=args.repo_id,
                                              root=args.data_root)
        self.dataset = None  # LeRobotDataset handle, created lazily
        self.cam_keys = self._select_cameras()
        self.out_dir = args.out_dir or os.path.join(args.data_root, "eps_data")
        # Step-2 outputs read from <out-dir>/depth_pose/<episode>/subtask_XX
        # — the videos go to the sibling visualization/ tree
        self.depth_pose_dir = os.path.join(self.out_dir, "depth_pose")
        self.viz_dir = os.path.join(self.out_dir, "visualization")
        self.n_rendered = 0

    # --- dataset access -----------------------------------------------------

    def _ensure_dataset(self):
        """LeRobotDataset handle (videos on disk), created lazily: only
        the per-step frame decode opens it (mirrors
        DataExtract._ensure_dataset)."""
        if self.dataset is None:
            self.dataset = LeRobotDataset(repo_id=self.args.repo_id,
                                          root=self.args.data_root,
                                          download_videos=False)
        return self.dataset

    @staticmethod
    def _camera_subdir(cam_key):
        """Subdir per camera, named after the dataset's camera key (the
        Step-2 npz folders are named depth_<subdir>)."""
        return cam_key.rsplit(".", 1)[-1]

    def _cam_key_for_subdir(self, subdir: str) -> str:
        """Dataset camera key whose subdir matches an on-disk depth_<cam>
        folder — the key decoding that camera's frames online."""
        for key in self.ds_meta.camera_keys:
            if self._camera_subdir(key) == subdir:
                return key
        raise ValueError(f"{subdir!r} matches no dataset camera key "
                         f"(frames cannot be decoded online)")

    def _select_cameras(self):
        """-c indices -> dataset camera keys, validated against the
        dataset's camera_keys (None: every camera with Step-2 outputs on
        disk is rendered)."""
        idxes = self.args.camera_idxes
        if idxes is None:
            return None
        for idx in idxes:
            if not 0 <= idx < len(self.ds_meta.camera_keys):
                raise ValueError(f"camera_idx {idx} out of range "
                                 f"(dataset has {len(self.ds_meta.camera_keys)} cameras)")
        return {idx: self.ds_meta.camera_keys[idx] for idx in idxes}

    # --- frame access -------------------------------------------------------

    def _frame_loader(self, key: str, h: int, w: int):
        """Per-step RGB loader of one camera for render_stream_video:
        decodes the step's dataset frame of that camera and resizes it to
        the camera's depth resolution, matching load_pair's image contract
        (RGB — the renderer wants (N, H, W, 3) uint8)."""
        def load(stem: str) -> np.ndarray:
            t = int(stem.rsplit("_", 1)[-1])
            frame = self._ensure_dataset()[t]
            # lerobot decodes the RGB camera videos as RGB — no reversal, the
            # renderer wants (N, H, W, 3) uint8 RGB
            rgb = np.asarray(to_pil(frame[key]), dtype=np.uint8)
            if rgb.shape[:2] != (h, w):
                rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
            return rgb[None]

        return load

    # --- disk discovery -----------------------------------------------------

    def _episode_dir(self, ep_idx: int) -> str:
        """The on-disk episode dir name of a dataset episode index."""
        return f"ep{ep_idx:06d}"

    def _discover_episodes(self) -> list[int]:
        """Sorted dataset indices of the episode dirs under depth_pose/
        (a dir counts when its name is ep<int>)."""
        eps = []
        for p in Path(self.depth_pose_dir).iterdir():
            if not p.is_dir() or not p.name.startswith("ep"):
                continue
            try:
                eps.append(int(p.name[2:]))
            except ValueError:
                continue
        return sorted(eps)

    def _segment_dirs(self, ep_idx: int) -> list[str]:
        """Sorted subtask_* dir names of an episode, as present under
        depth_pose/ep%06d/ — the Step-2 segmentation itself, never
        recomputed from dataset splits."""
        ep_dir = os.path.join(self.depth_pose_dir, self._episode_dir(ep_idx))
        if not os.path.isdir(ep_dir):
            return []
        return sorted(p.name for p in Path(ep_dir).iterdir()
                      if p.is_dir() and p.name.startswith("subtask_"))

    def _select_episodes(self, discovered: list[int]) -> list[int]:
        """Episodes to process: the discovered ones, filtered by -e (a
        requested episode with no Step-2 outputs on disk is warned about
        and skipped — never a failure) and capped by --max-episodes."""
        if self.args.episode_idxes is not None:
            requested = set(self.args.episode_idxes)
            for ep in sorted(requested - set(discovered)):
                print(f"  episode {ep}: no Step-2 outputs under "
                      f"{self.depth_pose_dir} — skipped")
            eps = [ep for ep in discovered if ep in requested]
        else:
            eps = discovered
        if self.args.max_episodes is not None:
            eps = eps[: self.args.max_episodes]
        return eps

    def _segment_cameras(self, segment: str,
                         seg_dir: str) -> list[tuple[str, str]]:
        """(camera subdir, dataset key) pairs of one segment to render.
        With -c: exactly the requested cameras, in dataset order (one
        without Step-2 outputs keeps the caller's "skip, no npz results"
        message). Without: every depth_<cam> dir present that has npz
        results and maps back to a dataset camera key — a dir with no
        outputs, or whose camera the dataset does not have, is skipped
        with a message (its frames cannot be decoded online)."""
        if self.cam_keys is not None:
            return [(self._camera_subdir(key), key)
                    for key in self.cam_keys.values()]
        cameras = []
        for d in sorted(Path(seg_dir).iterdir()):
            if not d.is_dir() or not d.name.startswith("depth_"):
                continue
            cam = d.name[len("depth_"):]
            if not list(d.glob("*.npz")):
                print(f"  [{segment}] camera {cam}: skip, no npz results "
                      f"in {d}")
                continue
            try:
                key = self._cam_key_for_subdir(cam)
            except ValueError as e:
                print(f"  [{segment}] camera {cam}: skip, {e}")
                continue
            cameras.append((cam, key))
        return cameras

    # --- orchestration ------------------------------------------------------

    def run(self) -> None:
        discovered = (self._discover_episodes()
                      if os.path.isdir(self.depth_pose_dir) else [])
        if not discovered:
            print(f"\nno Step-2 outputs under {self.depth_pose_dir} — run "
                  f"run_step2_depth_stream.py first")
            return
        eps = self._select_episodes(discovered)
        print(f"\n{len(eps)} episode(s) selected:")
        for ep in eps:
            print(f"  episode {ep}: {len(self._segment_dirs(ep))} "
                  f"sub-task segment(s)")
        for ep_idx in tqdm(eps, desc="episodes"):
            self._process_episode(ep_idx)
        print(f"\ndone: {self.n_rendered} camera video(s) -> {self.viz_dir}")

    def _process_episode(self, ep_idx: int) -> None:
        for segment in self._segment_dirs(ep_idx):
            self._process_segment(ep_idx, segment)

    def _process_segment(self, ep_idx: int, segment: str) -> None:
        seg_dir = os.path.join(self.depth_pose_dir,
                               self._episode_dir(ep_idx), segment)
        for cam, key in self._segment_cameras(segment, seg_dir):
            depth_dir = os.path.join(seg_dir, f"depth_{cam}")
            stems = sorted(p.stem for p in Path(depth_dir).glob("*.npz"))
            if not stems:  # -c picked a camera the Step-2 run skipped
                print(f"  [{segment}] camera {cam}: skip, no npz results in "
                      f"{depth_dir} (run run_step2_depth_stream.py for this "
                      f"camera)")
                continue
            depth0, _, _ = load_stream_data(cam, seg_dir, stems[0])
            h, w = depth0.shape
            out_path = os.path.join(self.viz_dir,
                                    self._episode_dir(ep_idx), segment,
                                    cam, "depth_pose.mp4")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            abs_idxes = [int(s.rsplit("_", 1)[-1]) for s in stems]
            print(f"  [{segment}] camera {cam}: {len(stems)} step(s) at "
                  f"dataset frames [{abs_idxes[0]}..{abs_idxes[-1]}] -> "
                  f"{out_path}")
            # view-fit stride derived from the stems' dataset spacing:
            # only a spatial thinning of each depth map for the auto
            # view-fit (visualize_stream._union_scene_points) — unrelated
            # to Step-2's frame stride (fallback 1: single-step segment)
            fit_stride = (abs_idxes[1] - abs_idxes[0]
                          if len(stems) > 1 else 1)
            w_vid, h_vid = (int(x) for x in self.args.size.lower().split("x"))
            render_stream_video(
                stems,
                [cam],
                seg_dir,
                out_path,
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
                frame_loader=self._frame_loader(key, h, w),
            )
            self.n_rendered += 1


def main():
    SubtaskStreamVisualize(parse_args()).run()


if __name__ == "__main__":
    main()
