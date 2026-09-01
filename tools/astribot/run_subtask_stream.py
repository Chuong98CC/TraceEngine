"""Stream per-sub-task depth + pose directly from a local LeRobotDataset,
with optional per-chunk WAFT motion masks.

Wraps ``DataExtract`` (tools/astribot/extract_frames.py) for dataset
introspection, episode/camera selection and sub-task split-frame inference:
``--episode-idxes`` picks the episodes and ``DataExtract._load_splits()``
provides the split frames per episode (the ground-truth ``subtask_index``
column when it exists, the gripper-inferred ``subtask_splits.json``
otherwise).  Each sub-task segment ``[from_idx] + split_frames + [to_idx]``
is streamed online — frames are decoded from the dataset one chunk at a
time and never written to disk, so no per-subtask videos are needed.

WAFT optical flow is **off by default** (``--with-optical-flow`` enables
it).  When enabled it runs in per-batch interleave with the streaming
model, not over the whole segment:
for each streaming chunk (``num_views`` temporal frames) the chunk's frames
are fetched, WAFT computes the motion masks of the chunk's steps (pairs
``(t, t + stride)``), and the masks ground the chunk-to-chunk alignment.
Masks are cached by absolute step index, so frames shared between
overlapping chunks reuse the previous chunk's result instead of re-running
WAFT; only the new frames of a chunk run through the model.

The streaming backend runs through an ``OnlineStreaming`` subclass of the
concrete backend (``VGGT_OMG_Streaming`` / ``DA3_Streaming`` /
``A2F_Streaming``): ``img_list`` holds virtual ``frame_%06d.jpg`` stems
(output files are named after them, so saved frames keep their absolute
dataset indices), and the chunk's image arrays are slice-swapped in for the
model forward only.  Peak memory is one chunk of frames + masks.

Three backends (``--backend``): RGB-only cameras run ``da3`` or
``vggt_omega``; cameras with a paired raw-depth feature
(``observation.depth.<name>``, uint16 mm) can run ``a2f``, which densifies
the sensor depth with Any2Full.  For ``a2f`` the raw depth is decoded from
the dataset alongside the RGB frames and slice-swapped into
``depth_paths`` the same way — nothing is written to disk.

Examples
--------
    # All sub-tasks of episode 0, cam_head, VGGT-Omega (WAFT off by default)
    python tools/astribot/run_subtask_stream.py \
        --repo-id Kronze157/astri_making_coffee_vlva \
        --data-root /data/astri_making_coffee --episode-idxes 0

    # Stereo pair, DA3 backend, with WAFT motion masks
    python tools/astribot/run_subtask_stream.py \
        --repo-id Kronze157/astri_making_coffee_vlva \
        --data-root /data/astri_making_coffee --episode-idxes 0 \
        --camera-idxes 4 5 --backend da3 --with-optical-flow

    # RGB-D camera, a2f backend (Any2Full densifies the sensor depth;
    # defaults to the first camera with a paired raw-depth feature)
    python tools/astribot/run_subtask_stream.py \
        --repo-id Kronze157/astri_making_coffee_vlva \
        --data-root /data/astri_making_coffee --episode-idxes 0 \
        --backend a2f
"""

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from lerobot.datasets import LeRobotDatasetMetadata
from tqdm import tqdm

from depth_models.streaming.a2f_streaming import A2F_Streaming
from depth_models.streaming.da3_streaming import DA3_Streaming
from depth_models.streaming.loop_utils.config_utils import load_config
from depth_models.streaming.vggt_omg_streaming import VGGT_OMG_Streaming
from tools.astribot.extract_frames import DataExtract
from tools.general_test.module.infer_waft import _compute_motion_mask_gray
from tools.general_test.pipeline.run_depth_stream import _report_run_stats
from utils.visualize_mask import to_pil

DEFAULT_WAFT_CKPT = "weights/waftv2/waftv2_dinov3_i5_640x480_tf32.engine"


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Stream per-sub-task depth + pose directly from a "
                    "LeRobotDataset (online frames), with optional per-chunk "
                    "WAFT motion masks."
    )
    parser.add_argument("--repo-id", "-id", required=True,
                        help="dataset repo id as seen by LeRobotDataset")
    parser.add_argument("--data-root", "-d", required=True,
                        help="root passed to LeRobotDataset; for chunked datasets give the "
                             "full chunk path, e.g. /data/x/chunk-0000/part-0000")
    parser.add_argument("--camera-idxes", "-c", nargs="+", type=int, default=None,
                        help="indices into the dataset's camera_keys to stream "
                             "(default: the first RGB camera — for --backend a2f the "
                             "first camera with a paired raw-depth feature; note the "
                             "backend's num_views must be divisible by the number of "
                             "cameras, e.g. 64 %% n == 0 for VGGT-Omega)")
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
    parser.add_argument("--backend", choices=("vggt_omega", "da3", "a2f"),
                        default="vggt_omega",
                        help="streaming backend: RGB-only cameras run da3 or "
                             "vggt_omega; a2f is for RGB-D cameras with a paired "
                             "raw-depth feature (Any2Full densifies the sensor "
                             "depth) (default: %(default)s)")
    parser.add_argument("--stride", type=int, default=4,
                        help="subsample the sequence (every N-th frame); with WAFT "
                             "enabled this is also the optical-flow pair gap, and the "
                             "last N frames of each segment are not streamed (no pair) "
                             "(default: %(default)s)")
    parser.add_argument("--with-optical-flow", action="store_true",
                        help="enable WAFT optical-flow motion masks (default: off — "
                             "the chunk alignment then uses full confidence instead)")
    parser.add_argument("--motion-threshold", "-thr", type=float, default=2.0,
                        help="flow-magnitude (pixel displacement) threshold above which "
                             "a pixel counts as moving (default: %(default)s)")
    parser.add_argument("--waft-checkpoint", default=DEFAULT_WAFT_CKPT,
                        help="WAFT checkpoint; backend inferred from .engine/.onnx "
                             "(default: %(default)s)")
    parser.add_argument("--model-path", default=None,
                        help="VGGT-Omega model artifact path (.pt2); overrides the "
                             "backend's default")
    parser.add_argument("--anyview-model-path", default=None,
                        help="DA3 any-view .pt2 override")
    parser.add_argument("--metric-model-path", default=None,
                        help="DA3 metric-depth .pt2 override")
    parser.add_argument("--a2f-model-path", default=None,
                        help="Any2Full model artifact path (.pt2); overrides the "
                             "a2f backend's default")
    parser.add_argument("--no-depth-enhance", action="store_true",
                        help="a2f backend only: skip the Any2Full depth-enhance "
                             "model and feed the raw sensor depth directly into "
                             "the alignment step")
    parser.add_argument("--depth-scale", type=float, default=0.001,
                        help="a2f backend only: metres per unit of the raw depth "
                             "(default 0.001 = uint16 mm -> metres)")
    parser.add_argument("--config",
                        default="src/depth_models/streaming/configs/base_config.yaml",
                        help="alignment library/method, loop-closure settings")
    parser.add_argument("--device", default=None, choices=["cuda", "cpu"],
                        help="device (default: auto)")
    parser.add_argument("--skip-done", action="store_true",
                        help="skip sub-tasks whose pipeline output already exists")
    return parser.parse_args(argv)


class OnlineStreaming:
    """In-memory frames + optional WAFT masks for the BaseStreaming chunk loop.

    Mixed into a concrete backend (``VGGT_OMG_Streaming`` /
    ``DA3_Streaming`` / ``A2F_Streaming``) to replace the disk-folder
    image/mask/depth loading: ``img_list`` holds virtual ``frame_%06d.jpg``
    stems (the base run() names output files after them, and padding copies
    keep their stems so duplicates map to the same mask), and the chunk's
    image arrays are slice-swapped in for the model forward only.  For the
    ``a2f`` backend the raw depth arrays are swapped into ``depth_paths``
    in parallel (placeholder stems from ``_load_depth_paths``).  Frames are
    decoded from the dataset online, one chunk at a time; WAFT masks are
    computed per step and cached by absolute step index, so frames shared
    between overlapping chunks reuse the previous chunk's masks instead of
    re-running optical flow.
    """

    #: frames are stored BGR (dataset PIL RGB flipped once at fetch); models
    #: expecting RGB (VGGT-Omega) flip back when assembling a chunk
    _rgb_input = False

    def attach(self, wrapper) -> None:
        """Bind the dataset wrapper that provides frames + the WAFT model."""
        self.wrapper = wrapper
        self._cam_keys = list(wrapper.cam_keys.values())

    def prepare(self, steps, seg_lo, seg_hi, output_dir) -> None:
        """Per-segment state.  run() resets its own per-run attributes
        (img_list, mask_paths, chunk_times), so one backend instance is
        reused across segments."""
        self.steps = list(steps)
        self.seg_lo, self.seg_hi = seg_lo, seg_hi
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._frame_cache: dict[int, dict] = {}
        self._mask_cache: dict[int, list[np.ndarray]] = {}

    # --- BaseStreaming hook overrides --------------------------------------

    def _load_image_list(self) -> int:
        """Build the virtual image list: one stem per (step, camera)."""
        n = len(self.steps)
        self.img_list = [f"frame_{t:06d}.jpg" for t in self.steps
                         for _ in range(self.num_cams)]
        print(f"  Online: {n} time steps x {self.num_cams} cameras "
              "(virtual stems, frames decoded per chunk)")
        return n

    def _load_mask_paths(self) -> None:
        """Placeholder stems parallel to the (padded) img_list; the actual
        masks are computed lazily per chunk in _stack_chunk_masks."""
        if self.wrapper.waft_model is None:
            self.mask_paths = None
        else:
            self.mask_paths = list(self.img_list)

    def _load_depth_paths(self) -> None:
        """Placeholder stems parallel to the (padded) img_list; the actual
        raw-depth arrays are swapped in per chunk in _process_chunk (the
        a2f backend only)."""
        if self.depth_dirs is None:
            self.depth_paths = None
        else:
            self.depth_paths = list(self.img_list)

    def _process_chunk(self, start: int, end: int) -> dict:
        """Fetch this chunk's frames online, slice-swap them into img_list
        for the model forward, then restore the virtual stems."""
        steps = [int(p.rsplit("_", 1)[-1].split(".", 1)[0])
                 for p in self.img_list[start:end]]
        t_min, t_max = min(steps), max(steps)
        # WAFT pairs (t, t+stride) extend one stride beyond the chunk's last
        # step; capped at the segment end (no WAFT -> no overhang).
        overhang = self.wrapper.args.stride if self.wrapper.waft_model is not None else 0
        t_fetch = min(t_max + overhang + 1, self.seg_hi)
        for t in range(t_min, t_fetch):
            if t not in self._frame_cache:
                self._frame_cache[t] = self.wrapper._dataset_step(t)
        # drop older steps: keep only the fetch range (which includes the
        # next chunk's overhang)
        keep = set(range(t_min, t_fetch))
        self._frame_cache = {t: d for t, d in self._frame_cache.items() if t in keep}

        frames = []
        for i, t in enumerate(steps):
            arr = self._frame_cache[t][self._cam_keys[i % self.num_cams]]
            if self._rgb_input:
                arr = arr[:, :, ::-1]
            frames.append(arr)

        # a2f backend: swap the chunk's raw depth arrays into depth_paths in
        # parallel with the RGB frames (placeholder stems built by
        # _load_depth_paths).  The pipeline expects depth at the frame
        # resolution, so resize when the dataset stores it differently.
        saved_depth = None
        if self.depth_paths is not None:
            depth_keys = self.wrapper.depth_keys
            depths = []
            for i, t in enumerate(steps):
                depth = self._frame_cache[t][depth_keys[i % self.num_cams]]
                if depth.shape[:2] != frames[i].shape[:2]:
                    depth = cv2.resize(depth, (frames[i].shape[1], frames[i].shape[0]),
                                       interpolation=cv2.INTER_NEAREST)
                depths.append(depth)
            saved_depth = self.depth_paths[start:end]
            self.depth_paths[start:end] = depths

        saved = self.img_list[start:end]
        self.img_list[start:end] = frames
        try:
            return super()._process_chunk(start, end)
        finally:
            self.img_list[start:end] = saved
            if saved_depth is not None:
                self.depth_paths[start:end] = saved_depth

    def _stack_chunk_masks(self, mask_stems: list[str], h: int, w: int) -> np.ndarray:
        """WAFT masks of one chunk: computed per step and cached by absolute
        step index across chunks, so steps shared with the previous chunk
        (overlap) reuse its masks instead of re-running optical flow.
        ``mask_stems`` is the chunk's placeholder stem list (padding copies
        carry their original stems, so duplicates map to the cached mask).
        Returns (N, h, w) float32 with 1 = static, 0 = moving."""
        steps = [int(s.rsplit("_", 1)[-1].split(".", 1)[0]) for s in mask_stems]
        stacked = []
        for i, t in enumerate(steps):
            m = self._mask_cache.get(t)
            if m is None:
                m = self._compute_step_masks(t)
                self._mask_cache[t] = m
            img = m[i % self.num_cams]
            if (img.shape[0], img.shape[1]) != (h, w):
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
            stacked.append((img < 127).astype(np.float32))
        keep = set(steps)
        self._mask_cache = {t: m for t, m in self._mask_cache.items() if t in keep}
        return np.stack(stacked)

    # --- helpers ------------------------------------------------------------

    def _compute_step_masks(self, t: int) -> list[np.ndarray]:
        """One motion mask per camera for step t: WAFT flow between the BGR
        frames (t, t + stride).  255 = moving, 0 = static (the stacker above
        inverts to 1 = static, matching the disk-mask convention)."""
        stride = self.wrapper.args.stride
        thr = self.wrapper.args.motion_threshold
        masks = []
        for cam_key in self._cam_keys:
            flow = self.wrapper.waft_model(self._frame_cache[t][cam_key],
                                           self._frame_cache[t + stride][cam_key])
            masks.append(_compute_motion_mask_gray(flow, thr))
        return masks


class OnlineVGGTStreaming(OnlineStreaming, VGGT_OMG_Streaming):
    """VGGT-Omega streaming with online frames; the preprocess expects RGB."""

    _rgb_input = True


class OnlineDA3Streaming(OnlineStreaming, DA3_Streaming):
    """DA3 streaming with online frames; the preprocess expects BGR arrays."""

    _rgb_input = False


class OnlineA2FStreaming(OnlineStreaming, A2F_Streaming):
    """Any2Full RGB-D streaming with online frames; the preprocess expects
    BGR arrays and the raw depth comes from the dataset (the a2f backend's
    depth_dirs are virtual — see SubtaskStreamExtract._ensure_stream)."""

    _rgb_input = False


def _paired_depth_key(cam_key: str, features: dict) -> str | None:
    """Raw-depth feature key paired with ``cam_key``, or None when the camera
    has no usable depth.

    Mirrors ``DataExtract._depth_key_for`` plus the uint16-dtype check of
    ``_save_subtask_frames``, but works from the dataset metadata before the
    extractor is constructed (camera selection happens pre-``__init__``)."""
    name = DataExtract._camera_subdir(cam_key)
    raw = f"observation.depth.{name}"
    if raw in features and features[raw].get("dtype") == "uint16":
        return raw
    video = f"{cam_key}_depth"
    if video in features:
        info = features[video].get("info") or {}
        if info.get("video.is_depth_map"):
            return video
    return None


class SubtaskStreamExtract(DataExtract):
    """Per-sub-task online streaming with optional WAFT motion masks.

    Reuses DataExtract for dataset introspection, episode/camera selection
    and split-frame inference; frames are decoded from the LeRobotDataset
    online and fed to the streaming backend in per-chunk batches.
    """

    def __init__(self, args):
        if args.camera_idxes is None:
            meta = LeRobotDatasetMetadata(repo_id=args.repo_id,
                                          root=args.data_root)
            keys = meta.camera_keys
            features = getattr(meta.info, "features", None)
            if features is None:
                features = getattr(meta, "features", {})
            if args.backend == "a2f":
                # first camera with a paired uint16 mm raw-depth feature
                args.camera_idxes = [i for i, key in enumerate(keys)
                                     if _paired_depth_key(key, features)][:1]
            else:
                args.camera_idxes = [i for i, key in enumerate(keys) if "depth" not in key][:1]
        args.mode = "videos"  # DataExtract needs one of its modes; only the
                             # output-dir/camera/episode machinery is reused
        super().__init__(args)
        self.pipeline_dir = os.path.join(self.out_dir, "pipeline")
        os.makedirs(self.pipeline_dir, exist_ok=True)
        self.waft_model = None
        self.stream = None
        # per-camera raw-depth feature keys (None for RGB-only cameras),
        # resolved once via DataExtract._depth_key_for
        self.depth_keys: list[str | None] = [self._depth_key_for(k)
                                             for k in self.cam_keys.values()]
        if args.backend == "a2f":
            self._validate_depth_cameras()
        if args.with_optical_flow:
            self._ensure_waft()

    # --- model setup --------------------------------------------------------

    def _validate_depth_cameras(self) -> None:
        """a2f backend: every selected camera must have a paired raw-depth
        feature (uint16 mm, see _depth_key_for)."""
        if not self.cam_keys:
            raise ValueError(
                "--backend a2f found no camera with a paired raw-depth feature "
                "in this dataset; pass --camera-idxes explicitly, or use "
                "--backend da3/vggt_omega for RGB-only cameras")
        missing = [k for k, dk in zip(self.cam_keys.values(), self.depth_keys)
                   if dk is None or self._feature(dk).get("dtype") != "uint16"]
        if missing:
            raise ValueError(
                f"--backend a2f needs one paired raw-depth feature (uint16 mm) "
                f"per camera; missing for: {missing}. Use --backend da3 or "
                f"vggt_omega for RGB-only cameras.")

    def _ensure_waft(self) -> None:
        """Load the WAFT model (backend inferred from the checkpoint
        extension, like infer_waft.py)."""
        ckpt = self.args.waft_checkpoint
        if ckpt.endswith(".engine"):
            from flow_models.waft import WAFT
            print(f"Loading TensorRT engine: {ckpt}")
            assert os.path.exists(ckpt), f"Checkpoint not found: {ckpt}, please compile the TRT engine first or change to ONNX model, or drop the flag --with-optical-flow . To compile the TRT engine, use command: ./scripts/general_test/export_trt_docker.sh <abs/path/to>/weights/waftv2/waftv2_dinov3_i5_640x480.onnx. "
            self.waft_model = WAFT(ckpt, bgr_input=True)
        else:
            from flow_models.waft import WAFTOnnx
            print(f"Loading ONNX model: {ckpt}")
            self.waft_model = WAFTOnnx(ckpt, device=self.args.device,
                                       bgr_input=True)

    def _ensure_stream(self) -> OnlineStreaming:
        """Construct the streaming backend once; run() is called once per
        segment (its per-run state resets itself)."""
        if self.stream is None:
            config = load_config(self.args.config)
            input_dirs = [self._camera_subdir(k) for k in self.cam_keys.values()]
            if self.args.backend == "vggt_omega":
                self.stream = OnlineVGGTStreaming(
                    input_dirs=input_dirs, save_dir=self.pipeline_dir,
                    config=config, device=self.args.device,
                    model_path=self.args.model_path)
            elif self.args.backend == "da3":
                self.stream = OnlineDA3Streaming(
                    input_dirs=input_dirs, save_dir=self.pipeline_dir,
                    config=config, device=self.args.device,
                    anyview_model_path=self.args.anyview_model_path,
                    metric_model_path=self.args.metric_model_path)
            else:
                # virtual depth folders: the raw depth is decoded from the
                # dataset per chunk (OnlineStreaming._load_depth_paths), the
                # folders only satisfy the a2f input contract
                self.stream = OnlineA2FStreaming(
                    input_dirs=input_dirs, save_dir=self.pipeline_dir,
                    config=config, device=self.args.device,
                    anyview_model_path=self.args.anyview_model_path,
                    a2f_model_path=self.args.a2f_model_path,
                    depth_dirs=[f"depth_{d}" for d in input_dirs],
                    depth_scale=self.args.depth_scale,
                    use_depth_enhance=not self.args.no_depth_enhance)
            self.stream.attach(self)
        return self.stream

    # --- frame access ---------------------------------------------------------

    def _dataset_step(self, t: int) -> dict:
        """Online decode of dataset frame t for all selected cameras: RGB
        frames as BGR uint8 arrays (the dataset stores PIL RGB; flipped once
        here — WAFT, DA3 and a2f take BGR, VGGT flips back in the stream)
        plus the raw uint16 mm depth of each camera that has a paired depth
        feature, keyed by the feature key (the a2f prompt)."""
        frame = self._ensure_dataset()[t]
        step = {key: np.asarray(to_pil(frame[key]), dtype=np.uint8)[:, :, ::-1]
                for key in self.cam_keys.values()}
        for key, dkey in zip(self.cam_keys.values(), self.depth_keys):
            if dkey is not None:
                step[dkey] = np.asarray(frame[dkey]).astype(np.uint16)
        return step

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
        seg_dir = os.path.join(self.pipeline_dir, f"ep{self.ep_idx:06d}",
                               f"subtask_{k:02d}")
        if self.args.skip_done and any(p for p in Path(seg_dir).glob("depth_*")
                                       if any(p.iterdir())):
            print(f"  [subtask {k:02d}] skip: output exists in {seg_dir}")
            return
        steps = list(range(lo, hi, self.args.stride))
        if self.waft_model is not None:
            # the last `stride` frames have no WAFT pair (t, t+stride)
            steps = [t for t in steps if t + self.args.stride < hi]
        if not steps:
            print(f"  [subtask {k:02d}] skip: {hi - lo} frames with stride "
                  f"{self.args.stride} yield no masked steps")
            return
        print(f"  [subtask {k:02d}] frames [{lo}, {hi}) -> {len(steps)} steps "
              f"(stride {self.args.stride}) -> {seg_dir}")
        stream = self._ensure_stream()
        stream.prepare(steps=steps, seg_lo=lo, seg_hi=hi, output_dir=seg_dir)
        # peak GPU stats reset per segment, so the report reflects this
        # segment's inference only (weights load before that point)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        stats = stream.run()
        # same timing/memory summary + timings.json schema as run_depth_stream.py
        _report_run_stats(self.args.backend, seg_dir, stats)
        print(f"  [subtask {k:02d}] done in {time.perf_counter() - t0:.1f}s")


def main():
    SubtaskStreamExtract(parse_args()).run()


if __name__ == "__main__":
    main()
