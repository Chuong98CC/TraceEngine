"""Step 3a' — WAFT motion masks of the sub-task starts (Step-3 sampling
pipeline, optional).

Optical-flow pass of the Step-3 sampling pipeline, run by the
run_step3_init_points.py driver between Step 3a and Step 3b under
--with-optical-flow-mask: for every sub-task of the selected (episode,
camera) it runs WAFTv2 online over the frames of the sub-task's opening
window and saves the motion mask of the first pair whose flow is
significant. Step 3b then unions that mask into the **manipulator**
prompt's row-0 SAM3 mask (the SAM ∪ optical-flow mask) — rescuing the
manipulator when the Step-3a RexOmni detection missed it.

Frames are decoded from the LeRobotDataset **online** (nothing extracted
to disk) — the same WAFT path as run_step2_depth_stream.py's
--with-optical-flow: WAFTv2_PT2 over RGB uint8 pairs, motion mask from
_compute_motion_mask_gray (infer_waft) at --motion-threshold.

    detections JSON (per-sub-task key-frames) + dataset videos
               │
               ▼  WAFT pairs (kf0, kf0 + k*stride), first significant wins
    ┌──────────────────────────────┐
    │  motion mask per sub-task    │   init_points/ep{ep}/subtask_XX/<cam>/
    │  (significant pairs only)    │   motion_rle.json (COCO RLE)
    └──────────────────────────────┘

**Window.** The first frame is fixed at the sub-task's 1st key-frame
(its first saved frame = the sub-task start — the same frame Step 3b
samples the manipulator's points on) and the second frame walks from
kf0+stride in stride steps up to the sub-task's **2nd key-frame**: the
gripper-close transport span starts there, and the carried object moving
with the arm would contaminate a manipulator mask. Every candidate pair
is therefore expressed at kf0's frame, so the chosen mask ORs 1:1 into
Step-3b's row-0 mask (no cross-frame geometry shift).

**Early stop.** A pair where the arm has not started moving yet shows ~no
flow, so the first frames may be empty; the pair whose motion mask is
significant (moving pixels > --motion-ratio of the frame) ends the scan —
longer baselines accumulate the sub-threshold per-stride motion, catching
the onset.

**Saving.** Only a *significant* mask is saved — as
`motion_rle.json` (COCO RLE, `utils.file_io.mask_rle.encode_rle`,
carrying the provenance of the pair that produced it) in the (sub-task,
camera) folder of Step-3b's init-points tree
(`<out-dir>/init_points/ep{ep}/subtask_XX/<camera>/`, beside the
`<prompt_slug>/` subtrees Step 3b writes) — the rescue then sits next to
the init points it shaped. When no pair in the window reaches
significance nothing is written: the rescue only unions real motion, so
Step 3b then keeps that sub-task's SAM mask alone. `--visualize`
additionally writes the significant pair's `flow.png` (flow_to_image on a
black background: pixels below --motion-threshold are zeroed, so the
coloured area is exactly the moving-pixel set) — the mask itself is in
the RLE, and a scan with no significant pair writes nothing at all.

Usage
-----
    # Standalone (the driver passes the same flags under
    # --with-optical-flow-mask):
    python tools/astribot/run_step3_motion_masks.py
        --repo-id Kronze157/astri_making_coffee_vlva
        --data-root /data/astri_making_coffee_v1 --episode-idxes 0
        --camera-idxes 0 --visualize
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from tqdm import tqdm

from tools.general_test.module.infer_waft import _compute_motion_mask_gray
from utils.file_io.mask_rle import encode_rle
from utils.keyframe_utils import cap_keyframes, sampling_points_root
from utils.visualize.visualize_flow import flow_to_image
from utils.visualize.visualize_mask import to_pil

#: WAFTv2 torch.export artifact backing the motion masks (the same
#: checkpoint as run_step2_depth_stream.py).
DEFAULT_WAFT_PT2 = "weights/waftv2/waftv2_dinov3_i5_640x480_bf16.pt2"
#: default gap between the fixed first frame and the walking second frame
#: (also Step 2's --stride).
DEFAULT_STRIDE = 4
#: default flow-magnitude (pixel displacement) threshold of a moving pixel.
DEFAULT_MOTION_THRESHOLD = 2.0
#: default moving-pixel fraction of the frame that makes a mask
#: "significant" (the scan then early-stops).
DEFAULT_MOTION_RATIO = 0.03

_EP_RE = re.compile(r"^ep(\d{6})$")


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Step 3a': WAFT motion masks of the sub-task starts, "
                    "online from the dataset (fed to Step 3b's manipulator "
                    "mask union under --with-optical-flow-mask)."
    )
    parser.add_argument("--repo-id", "-id", required=True,
                        help="dataset repo id as seen by LeRobotDataset")
    parser.add_argument("--data-root", "-d", required=True,
                        help="root of the local dataset copy (frames are "
                             "decoded online from it)")
    parser.add_argument("--camera-idxes", "-c", nargs="+", type=int, default=None,
                        help="indices into the dataset's camera_keys to "
                             "process (default: every non-depth camera)")
    parser.add_argument("--episode-idxes", "-e", nargs="*", type=int, default=None,
                        help="only process these episode indices (default: all "
                             "episodes with a Step-3a detections JSON)")
    parser.add_argument("--max-episodes", "-x", type=int, default=None,
                        help="cap the number of processed episodes")
    parser.add_argument("--out-dir", "-o", default=None,
                        help="output root (default: <data-root>/eps_data/"
                             "sampling_points); the motion masks land inside "
                             "Step 3b's init-points tree — "
                             "<out-dir>/init_points/ep{ep}/subtask_XX/"
                             "<camera>/, next to the prompt subtrees of that "
                             "same (sub-task, camera)")
    parser.add_argument("--detections-dir", default=None,
                        help="Step-3a detections root (default: "
                             "<out-dir>/detections)")
    parser.add_argument("--max-keyframes", type=int, default=8,
                        help="cap the key-frames per sub-task, applied to the "
                             "Step-3a list exactly like Step 3b does — the "
                             "window's 1st/2nd key-frame must match the ones "
                             "Step 3b samples on (default: %(default)s)")
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE,
                        help="gap between the fixed first frame and the "
                             "walking second frame of each flow pair "
                             "(default: %(default)s)")
    parser.add_argument("--motion-threshold", "-thr", type=float,
                        default=DEFAULT_MOTION_THRESHOLD,
                        help="flow-magnitude (pixel displacement) threshold "
                             "above which a pixel counts as moving "
                             "(default: %(default)s)")
    parser.add_argument("--motion-ratio", type=float, default=DEFAULT_MOTION_RATIO,
                        help="moving-pixel fraction of the frame above which "
                             "a mask is significant and the scan early-stops "
                             "(default: %(default)s)")
    parser.add_argument("--visualize", action="store_true",
                        help="also write the significant pair's flow.png "
                             "(flow_to_image) next to its "
                             "motion_rle.json for eyeballing; a "
                             "sub-task with no significant pair writes "
                             "nothing (default: off)")
    parser.add_argument("--skip-done", action="store_true",
                        help="skip sub-tasks whose motion_rle.json "
                             "already exists (a sub-task with no significant "
                             "pair has no file and is re-scanned)")
    return parser.parse_args(argv)


class MotionMaskExtract:
    """Per-(sub-task, camera) WAFT motion masks of the sub-task starts.

    The key-frames of each sub-task come from the Step-3a detections JSON
    (detections/ep{ep}/<camera>.json), whose per-sub-task key-frame list
    Step 3b caps identically — the JSON is the single source both steps
    agree on. The frames themselves are decoded online from the dataset;
    nothing is written to disk except the significant masks:
    init_points/ep{ep}/subtask_XX/<camera>/motion_rle.json (plus the
    --visualize flow.png of the chosen pair) — the same folder Step 3b
    writes its <prompt_slug>/ init points into.
    """

    def __init__(self, args):
        self.args = args
        self.meta = LeRobotDatasetMetadata(repo_id=args.repo_id,
                                           root=args.data_root)
        self.out_dir = args.out_dir or str(sampling_points_root(args.data_root))
        if args.detections_dir is None:
            args.detections_dir = str(Path(self.out_dir) / "detections")
        self.det_root = Path(args.detections_dir)
        if not self.det_root.is_dir():
            raise FileNotFoundError(
                f"{self.det_root} missing: run Step 3a first "
                "(.venv-rexomni/bin/python tools/general_test/"
                "run_object_detection.py ...)")
        idxes = args.camera_idxes
        if idxes is None:
            idxes = [i for i, key in enumerate(self.meta.camera_keys)
                     if "depth" not in key]
        # camera subdir name (the detections JSON key) -> dataset camera key
        self.cams = {key.rsplit(".", 1)[-1]: key
                     for i, key in enumerate(self.meta.camera_keys)
                     if i in idxes}
        if not self.cams:
            raise ValueError("no camera selected (--camera-idxes)")
        discovered = sorted(int(m.group(1)) for p in self.det_root.iterdir()
                            if p.is_dir() and (m := _EP_RE.match(p.name)))
        eps = discovered
        if args.episode_idxes is not None:
            missing = sorted(set(args.episode_idxes) - set(discovered))
            if missing:
                raise FileNotFoundError(
                    f"episode(s) {missing} have no Step-3a detections JSON "
                    f"under {self.det_root}")
            eps = [e for e in discovered if e in args.episode_idxes]
        if args.max_episodes is not None:
            eps = eps[: args.max_episodes]
        self.ep_idxes = eps
        # Step 3b's init-points tree: the motion-mask artefacts of a
        # (sub-task, camera) land in that sub-task's folder itself (the
        # parent of its <prompt_slug>/ subtrees), so they sit next to the
        # init points they shaped.
        self.init_dir = Path(self.out_dir) / "init_points"
        self.waft_model = None
        self.dataset = None  # LeRobotDataset handle, opened lazily
        self.ep_idx = 0
        self.cam_key = ""
        self._frames: dict[int, np.ndarray] = {}

    # --- model / dataset setup ---------------------------------------------

    def _ensure_waft(self):
        """Load the WAFTv2 torch.export model once (RGB input — the shared
        image_io path, same as run_step2_depth_stream.py)."""
        from flow_models.waftv2.waftv2_pt2 import WAFTv2_PT2
        print(f"Loading WAFTv2 .pt2 artifact: {DEFAULT_WAFT_PT2}")
        self.waft_model = WAFTv2_PT2(DEFAULT_WAFT_PT2)  # CUDA by default

    def _ensure_dataset(self) -> LeRobotDataset:
        if self.dataset is None:
            self.dataset = LeRobotDataset(repo_id=self.args.repo_id,
                                          root=self.args.data_root,
                                          download_videos=False)
        return self.dataset

    def _frame(self, t: int) -> np.ndarray:
        """Online decode of dataset frame t for the current camera: uint8
        HWC RGB (the dataset stores PIL RGB; no flip — WAFTv2 consumes RGB),
        cached across the pairs of one sub-task (the fixed first frame + the
        walked seconds)."""
        cached = self._frames.get(t)
        if cached is None:
            frame = self._ensure_dataset()[t]
            cached = np.ascontiguousarray(
                np.asarray(to_pil(frame[self.cam_key]), dtype=np.uint8))
            self._frames[t] = cached
        return cached

    def _pair_flow(self, a: int, b: int) -> np.ndarray:
        """Sanitised flow of one pair (H, W, 2) — the exact
        run_step2_depth_stream.py WAFT path."""
        flow = self.waft_model(self._frame(a), self._frame(b))
        # Legacy WAFTBase.__call__ sanitised NaN / Inf before returning;
        # keep the same parity for the masks / flow artefacts below.
        return np.nan_to_num(flow, nan=0.0, posinf=0.0, neginf=0.0)

    def _write_visuals(self, seg_dir: Path, flow: np.ndarray,
                       thr: float) -> None:
        """Debug artefact of one flow pair (--visualize): flow.png
        (flow_to_image, colour wheel) on a black background — the static
        pixels are zeroed (flow_to_image renders radius ~0 as bright
        white), at the same --motion-threshold that decides the mask, so
        the coloured area is exactly the pair's moving pixels."""
        seg_dir.mkdir(parents=True, exist_ok=True)
        vis = flow_to_image(flow, convert_to_bgr=True)
        vis[np.linalg.norm(flow, axis=-1) <= thr] = 0
        cv2.imwrite(str(seg_dir / "flow.png"), vis)

    # --- orchestration -------------------------------------------------------

    def run(self) -> None:
        if not self.ep_idxes:
            print("no episodes selected (no Step-3a detections JSON on disk)")
            return
        print(f"\n{len(self.ep_idxes)} episode(s) selected: {self.ep_idxes}")
        for ep_idx in tqdm(self.ep_idxes, desc="episodes"):
            self._process_episode(ep_idx)
        print(f"\ndone: {len(self.ep_idxes)} episode(s) -> {self.init_dir}")

    def _process_episode(self, ep_idx: int) -> None:
        self.ep_idx = ep_idx
        det_ep = self.det_root / f"ep{ep_idx:06d}"
        have = sorted(p.stem for p in det_ep.glob("*.json")) \
            if det_ep.is_dir() else []
        cams = [c for c in self.cams if c in have]
        if not cams:
            print(f"episode {ep_idx}: no Step-3a detections JSON under "
                  f"{det_ep} for the requested cameras "
                  f"{sorted(self.cams)} — skipped")
            return
        print(f"\nepisode {ep_idx}: detections loaded from {det_ep} "
              f"({len(cams)} camera(s): {cams})")
        for cam in cams:
            self._process_camera(cam)

    def _process_camera(self, cam: str) -> None:
        """Motion masks of one (episode, camera): every sub-task with >= 2
        key-frames gets the opening-window scan (see the module docstring)."""
        self.cam_key = self.cams[cam]
        path = self.det_root / f"ep{self.ep_idx:06d}" / f"{cam}.json"
        with open(path) as f:
            detections = json.load(f)
        self._frames = {}  # per-camera decode cache
        try:
            for k, sub in sorted(detections.get("subtasks", {}).items(),
                                 key=lambda kv: int(kv[0])):
                keys = cap_keyframes(
                    sorted(int(t) for t in sub.get("keyframes", [])),
                    self.args.max_keyframes)
                if len(keys) < 2:
                    print(f"  [subtask {int(k):02d}] ({cam}) skip: "
                          f"{len(keys)} key-frame(s) in the Step-3a JSON "
                          f"(need >= 2)")
                    continue
                self._process_segment(int(k), keys, cam)
        finally:
            self._frames = {}

    def _process_segment(self, k: int, keys: list[int], cam: str) -> None:
        """Opening-window scan of one sub-task: fixed first frame keys[0],
        walking second frame keys[0] + m*stride up to keys[1]; the first
        significant mask wins the scan. Only that mask is saved — as
        motion_rle.json (COCO RLE) under init_points/ep{ep}/
        subtask_{k}/<cam>/ (Step 3b's folder for that sub-task, beside its
        <prompt_slug>/ subtrees); a sub-task with no significant pair
        writes nothing (Step 3b then keeps its SAM mask alone).
        --visualize writes the same pair's flow.png too.
        """
        seg_dir = self.init_dir / f"ep{self.ep_idx:06d}" \
            / f"subtask_{k:02d}" / cam
        rle_path = seg_dir / "motion_rle.json"
        if self.args.skip_done and rle_path.is_file():
            print(f"  [subtask {k:02d}] ({cam}) skip: {rle_path} exists")
            return
        a = keys[0]
        b_max = keys[1]  # the 2nd key-frame: transport starts beyond it
        stride = self.args.stride
        if a + stride > b_max:
            print(f"  [subtask {k:02d}] ({cam}) skip: no flow pair in "
                  f"[{a}, {b_max}] at stride {stride}")
            return
        if self.waft_model is None:
            self._ensure_waft()
        thr = self.args.motion_threshold
        ratio = self.args.motion_ratio

        pairs: list[list[int]] = []
        counts: list[int] = []
        chosen: tuple[int, int] | None = None
        chosen_mask: np.ndarray | None = None
        chosen_flow: np.ndarray | None = None
        b = a + stride
        while b <= b_max:
            flow = self._pair_flow(a, b)
            mask = _compute_motion_mask_gray(flow, thr)
            n = int((mask > 0).sum())
            pairs.append([a, b])
            counts.append(n)
            if n > ratio * mask.size:
                chosen, chosen_mask, chosen_flow = (a, b), mask, flow
                break
            b += stride
        if chosen is None:
            sig = "no significant pair" if counts else "no flow pair"
            print(f"  [subtask {k:02d}] ({cam}) skip: {sig} in "
                  f"[{a}, {b_max}] at stride {stride} — nothing saved")
            return
        n = int((chosen_mask > 0).sum())

        seg_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "episode": int(self.ep_idx),
            "subtask": int(k),
            "camera_key": self.cam_key,
            "camera": cam,
            "keyframes": [int(t) for t in keys],
            "window": [int(a), int(b_max)],
            "stride": int(stride),
            "motion_threshold": float(thr),
            "motion_ratio": float(ratio),
            "pairs": pairs,
            "moving_pixels": counts,
            "chosen": [int(chosen[0]), int(chosen[1])],
            "mask": encode_rle(chosen_mask > 0),
        }
        with open(rle_path, "w") as f:
            json.dump(meta, f, indent=2)
        if self.args.visualize:
            self._write_visuals(seg_dir, chosen_flow, thr)
        print(f"  [subtask {k:02d}] ({cam}) chosen pair {chosen} "
              f"({n} moving px, significant) -> {rle_path}")


def main():
    MotionMaskExtract(parse_args()).run()


if __name__ == "__main__":
    main()
