"""Stream Any2Full (a3f) RGB-D densification over sub-task segments directly
from a local LeRobotDataset copy, online — no per-subtask videos needed.

Mirrors ``tools/astribot/run_subtask_stream.py``: ``DataExtract`` provides the
episode/camera selection and the sub-task split-frame inference
(``--episode-idxes`` picks the episodes, ``_load_splits()`` the split frames —
the ground-truth ``subtask_index`` column when it exists, the gripper-inferred
``subtask_splits.json`` otherwise).  Each sub-task segment
``[from_idx] + split_frames + [to_idx]`` is streamed online: frames are decoded
from the dataset one at a time and never written to disk.

The difference to run_subtask_stream.py: a3f is a single-frame RGB-D model, so
it only supports RGBD cameras.  Each selected RGB camera (default: the head and
torso cameras, the only ones with a depth pair) is paired by name convention
with its depth feature — ``observation.images.cam_X`` ↔
``observation.depth.cam_X``, stored as uint16 millimetres; the frame is
converted to sparse metric metres (``/ 1000``) and used as the a3f prompt that
grounds the prediction's metric scale.

Outputs land under ``<out-dir>/pipeline/<episode>/subtask_XX/depth_<camera>/``
with one ``frame_<step>.lz4`` per step, ``depth`` stored as uint16 millimetres
(``utils.astribot_dataloader.save_depth_m_lz4``) — the same .lz4 layout as
run_subtask_stream.py, so the same visualizers can load them.

Examples
--------
    # All sub-tasks of episode 0, head + torso (the RGBD cameras), every frame
    python tools/astribot/run_subtask_a3f.py \
        --repo-id Kronze157/astri_making_coffee_vlva \
        --data-root /data/astri_making_coffee --episode-idxes 0

    # Sub-sample every 4th frame, single head camera
    python tools/astribot/run_subtask_a3f.py \
        --repo-id Kronze157/astri_making_coffee_vlva \
        --data-root /data/astri_making_coffee --episode-idxes 0 \
        --camera-idxes 0 --stride 4
"""

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from lerobot.datasets import LeRobotDatasetMetadata
from tqdm import tqdm

from depth_models.a3f.any2full import Any2Full_PT2
from tools.astribot.extract_frames import DataExtract
from utils.astribot_dataloader import save_depth_m_lz4
from utils.visualize_mask import to_pil


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Stream Any2Full (a3f) RGB-D densification per sub-task "
                    "directly from a LeRobotDataset (online frames)."
    )
    parser.add_argument("--repo-id", "-id", required=True,
                        help="dataset repo id as seen by LeRobotDataset")
    parser.add_argument("--data-root", "-d", required=True,
                        help="root passed to LeRobotDataset; for chunked datasets give the "
                             "full chunk path, e.g. /data/x/chunk-0000/part-0000")
    parser.add_argument("--camera-idxes", "-c", nargs="+", type=int, default=None,
                        help="indices into the dataset's camera_keys of the RGBD "
                             "cameras to densify (default: every camera whose key has "
                             "a '<key>_depth' pair — the head and torso here)")
    select = parser.add_mutually_exclusive_group()
    select.add_argument("--episode-idxes", "-e", nargs="*", type=int, default=None,
                        help="only process these episode indices (default: all)")
    select.add_argument("--one-per-task", action="store_true",
                        help="select the first episode of each task")
    parser.add_argument("--max-episodes", "-x", type=int, default=None,
                        help="cap the number of processed episodes")
    parser.add_argument("--out-dir", "-o", default=None,
                        help="output root (default: <data-root>/eps_data); per-sub-task "
                             "results land under <out-dir>/pipeline/<episode>/subtask_XX/")
    parser.add_argument("--stride", type=int, default=1,
                        help="subsample the sequence (every N-th frame) "
                             "(default: %(default)s)")
    parser.add_argument("--pt2", type=str, default="weights/any2full/Any2Full_vitl_bf16.pt2",
                        help="Any2Full torch.export checkpoint path (default: %(default)s)")
    parser.add_argument("--denoise", action="store_true",
                        help="denoise the sparse sensor depth before prompting")
    parser.add_argument("--denoise_threshold", type=float, default=2.0)
    parser.add_argument("--denoise_kernel_size", type=int, default=None)
    parser.add_argument("--denoise_min_valid", type=int, default=5)
    parser.add_argument("--init_scaling", type=bool, default=True,
                        help="recover the metric scale via the affine fit against the "
                             "sparse anchors (default: %(default)s)")
    parser.add_argument("--max_depth", type=float, default=10)
    parser.add_argument("--min_depth", type=float, default=0)
    parser.add_argument("--device", default=None, choices=["cuda", "cpu"],
                        help="device (default: auto)")
    parser.add_argument("--skip-done", action="store_true",
                        help="skip sub-tasks whose pipeline output already exists")
    return parser.parse_args(argv)


class SubtaskA3FExtract(DataExtract):
    """Per-sub-task online Any2Full densification from a LeRobotDataset.

    Reuses DataExtract for dataset introspection, episode/camera selection and
    split-frame inference; frames are decoded from the dataset online, one step
    at a time, and each RGBD camera's frame is densified with the a3f model.
    """

    def __init__(self, args):
        if args.camera_idxes is None:
            keys = LeRobotDatasetMetadata(repo_id=args.repo_id,
                                          root=args.data_root).camera_keys
            args.camera_idxes = [i for i, key in enumerate(keys)
                                 if key + "_depth" in keys]
        args.mode = "videos"  # DataExtract needs one of its modes; only the
                             # output-dir/camera/episode machinery is reused
        super().__init__(args)
        self.pipeline_dir = os.path.join(self.out_dir, "pipeline")
        os.makedirs(self.pipeline_dir, exist_ok=True)
        self.depth_keys = {}
        for idx, key in self.cam_keys.items():
            dkey = f"observation.depth.{key.rsplit('.', 1)[-1]}"
            if self._feature(dkey):
                self.depth_keys[idx] = dkey
        missing = [k for idx, k in self.cam_keys.items() if idx not in self.depth_keys]
        if missing:
            raise ValueError(f"a3f needs a paired uint16 depth feature per RGB "
                             f"camera (observation.depth.<name>); missing for: "
                             f"{missing}")
        self.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    # --- model setup --------------------------------------------------------

    def _ensure_model(self) -> Any2Full_PT2:
        """Load the Any2Full model once; reused across segments."""
        if self.model is None:
            print(f"Loading Any2Full PT2: {self.args.pt2} ({self.device})")
            self.model = Any2Full_PT2(
                pt2_path=self.args.pt2,
                device=self.device,
                init_scaling=self.args.init_scaling,
                max_depth=self.args.max_depth,
                min_depth=self.args.min_depth,
            )
        return self.model

    # --- frame access ---------------------------------------------------------

    def _dataset_step(self, t: int) -> dict:
        """Online decode of dataset frame t for all selected cameras: RGB as
        BGR uint8 arrays (the dataset stores PIL RGB; flipped once here) and
        the paired depth as float32 metres (uint16 mm / 1000)."""
        frame = self._ensure_dataset()[t]
        out = {key: np.asarray(to_pil(frame[key]), dtype=np.uint8)[:, :, ::-1]
               for key in self.cam_keys.values()}
        for idx, dkey in self.depth_keys.items():
            out[dkey] = np.asarray(frame[dkey]).astype(np.float32) / 1000.0
        return out

    def _infer_frame(self, model: Any2Full_PT2, rgb_bgr: np.ndarray,
                     dep_m: np.ndarray) -> np.ndarray:
        """One RGBD frame through a3f -> dense metric depth (H, W) float32."""
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        rgb_t, dep_t = model.preprocess(
            rgb, dep_m,
            denoise=self.args.denoise,
            denoise_kwargs={
                "min_valid": self.args.denoise_min_valid,
                "threshold": self.args.denoise_threshold,
                "kernel_size": self.args.denoise_kernel_size,
            },
        )
        with torch.inference_mode():
            depth, disparity_pre_internal, prompt_depth_resized = model.infer(rgb_t, dep_t)
            pred = model.postprocess(depth, disparity_pre_internal,
                                     prompt_depth_resized)
        return pred.squeeze(0).squeeze(0).cpu().numpy()

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
        df = self._load_episode_table()
        self.task_id = (int(df["task_index"].iloc[0])
                        if "task_index" in df.columns
                        else self._episode_task_id(ep_idx))
        splits = self._load_splits()["split_frames"]
        bounds = [self.from_idx] + [int(s) for s in splits] + [self.to_idx]
        print(f"\nepisode {ep_idx}: {len(bounds) - 1} sub-task segment(s) "
              f"({self._task_description(self.task_id)})")
        for k, (lo, hi) in enumerate(zip(bounds, bounds[1:])):
            self._process_segment(k, lo, hi)

    def _process_segment(self, k: int, lo: int, hi: int) -> None:
        steps = list(range(lo, hi, self.args.stride))
        if not steps:
            print(f"  [subtask {k:02d}] skip: {hi - lo} frames with stride "
                  f"{self.args.stride} yield no steps")
            return
        seg_dir = os.path.join(self.pipeline_dir, f"ep{self.ep_idx:06d}",
                               f"subtask_{k:02d}")
        out_dirs = {idx: os.path.join(seg_dir, f"depth_{self._camera_subdir(key)}")
                    for idx, key in self.cam_keys.items()}
        if self.args.skip_done and all(os.path.isdir(d) and any(os.listdir(d))
                                       for d in out_dirs.values()):
            print(f"  [subtask {k:02d}] skip: output exists in {seg_dir}")
            return
        for d in out_dirs.values():
            os.makedirs(d, exist_ok=True)
        print(f"  [subtask {k:02d}] frames [{lo}, {hi}) -> {len(steps)} steps "
              f"(stride {self.args.stride}) -> {seg_dir}")

        model = self._ensure_model()
        t0 = time.perf_counter()
        for t in steps:
            frame = self._dataset_step(t)
            for idx, key in self.cam_keys.items():
                pred = self._infer_frame(model, frame[key],
                                         frame[self.depth_keys[idx]])
                save_depth_m_lz4(
                    pred,
                    os.path.join(out_dirs[idx], f"frame_{t:06d}.lz4"),
                )
        total = time.perf_counter() - t0
        self._save_timings(seg_dir, steps, total)
        print(f"  [subtask {k:02d}] done in {total:.1f}s "
              f"({len(steps) / total:.1f} steps/s)")

    def _save_timings(self, seg_dir: str, steps: list, total_s: float) -> None:
        report = {
            "total_s": round(total_s, 3),
            "num_frames": len(steps),
            "steps_per_s": round(len(steps) / total_s, 2) if total_s > 0 else None,
        }
        with open(os.path.join(seg_dir, "timings.json"), "w") as f:
            json.dump(report, f, indent=2)


def main():
    SubtaskA3FExtract(parse_args()).run()


if __name__ == "__main__":
    main()
