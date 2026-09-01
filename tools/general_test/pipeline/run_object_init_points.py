"""Step 3b — initial keypoints of the interacting objects.

Continues pipeline Step 3 (Sampling Keypoints) from the Step-3a detections
(run_object_detection.py): for each sub-task of each episode and each text
prompt, segment the object's masks on the sub-task's key-frames with SAM3
(bounding boxes + text prompt from the 3a detections, or text-only when a
frame has no detection), match keypoints across the key-frames with RoMAv2
on enlarged bounding-box crops (the mask is cropped with the same box and
fed to RoMAv2, so points are sampled inside the object only), and keep the
top-k keypoints that fall inside the object masks.

The key-frames are the jpgs that Step 1 saved to disk
(tools/astribot/extract_frames.py --mode key_frames, layout
<keyframes_root>/ep{ep:06d}/subtask_XX/<camera>/frame_<idx>.jpg): Step 3
only runs inference on a sparse set of frames, so they are persisted instead
of decoding the episode videos online (unlike Step 2, which streams every
frame). Only the results are written under

    <out_dir>/init_points/ep{ep:06d}/subtask_{k:02d}/<prompt_slug>/
        init_points.npz    keypoints (K, N, 2) px + per-key-frame masks/boxes
        masks_rle.json     SAM3 masks as COCO-style RLE (portable reuse)
        init_points.json   metadata (incl. empty_reason)
        viz.png            key-frames with masks, boxes and the tracks

The detection results come from Step 3a (one JSON per episode under
``<out_dir>/detections/``). Without them, pass ``--no-rexomni`` to run with
SAM3 text-only prompts. All model checkpoints are the repo defaults — no
overrides.

The same pipeline can run on a single folder of key-frame images (one
sub-task of one camera) instead of the episode layout: pass
``--keyframes-dir`` — the folder is sub-task 00 of a synthetic episode
labelled ``--episode-idx`` (default 0). run_e2e_init_points.py drives this mode
end-to-end.

Examples
--------
    # Uses the Step-3a detections (default prompts)
    python tools/general_test/pipeline/run_object_init_points.py \
        --data-root /data/astri_making_coffee --episode-idxes 0

    # SAM3 text-only (no Step-3a JSON)
    python tools/general_test/pipeline/run_object_init_points.py \
        --data-root /data/astri_making_coffee --episode-idxes 0 \
        --no-rexomni

    # One sub-task's key-frame folder (sub-task 00 of episode 0)
    python tools/general_test/pipeline/run_object_init_points.py \
        --keyframes-dir .../subtask_00/cam_head --episode-idx 0
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from det_seg_models.romav2.romav2 import RoMaV2PT2
from det_seg_models.romav2.utils import to_pixel
from det_seg_models.sam3 import Sam3Image, normalize_bbox
from det_seg_models.sam3.utils import box_xyxy_to_cxcywh
from utils.keyframe_utils import (
    camera_subdirs,
    cap_keyframes,
    discover_episodes,
    discover_folder_frames,
    discover_subtask_frames,
    keyframe_path,
    keyframes_root,
    select_camera,
    select_episodes,
)
from utils.rle_utils import encode_rle

DEFAULT_SAM3_CKPT = "weights/sam3/sam3_image_exported_bf16.pt2"
DEFAULT_ROMAV2_CKPT = "weights/romav2/romav2.pt2"
DEFAULT_PROMPTS = ["brown coffee cup", "robot gripper"]
#: fixed box-prompt slots of the exported SAM3 graph (callers truncate).
SAM3_BOXES_MAX = 8
#: max round-trip warp error in normalized coords (RoMAv2 cycle mode).
CYCLE_TH = 0.01

#: per-track colors for the visualization (same palette as infer_romav2.py).
_PALETTE = np.array(
    [
        [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
        [0, 255, 255], [255, 0, 255], [255, 128, 0], [128, 0, 255],
    ]
)


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Step 3b: SAM3 masks + RoMAv2 keypoints on the saved "
                    "sub-task key-frames (uses the Step-3a detections JSON)."
    )
    parser.add_argument("--data-root", "-d", default=None,
                        help="root of the local dataset copy; the default "
                             "key-frames root and output root derive from it "
                             "(not used with --keyframes-dir)")
    parser.add_argument("--keyframes-dir", default=None,
                        help="run on a single folder of key-frame images "
                             "(one sub-task of one camera) instead of the "
                             "episode layout; exclusive with --data-root / "
                             "--keyframes-root")
    parser.add_argument("--episode-idx", type=int, default=0,
                        help="episode index labelling the outputs (folder "
                             "mode only; default: %(default)s)")
    parser.add_argument("--camera-key", default=None,
                        help="camera key recorded in the outputs (folder "
                             "mode only; default: the folder name)")
    parser.add_argument("--keyframes-root", default=None,
                        help="root of the key-frames saved by Step 1 "
                             "(default: <data-root>/eps_data/key_frames)")
    parser.add_argument("--camera-keys", nargs="+", default=None,
                        help="camera subdir names (e.g. cam_head); the first "
                             "one present on disk wins (default: the first "
                             "RGB camera saved on disk, or the camera Step 3a "
                             "used)")
    parser.add_argument("--episode-idxes", "-e", nargs="*", type=int, default=None,
                        help="only process these episode indices (default: all "
                             "episodes with key-frames on disk)")
    parser.add_argument("--max-episodes", "-x", type=int, default=None,
                        help="cap the number of processed episodes")
    parser.add_argument("--out-dir", "-o", default=None,
                        help="output root (default: <data-root>/eps_data); "
                             "results land under <out-dir>/init_points/")
    parser.add_argument("--text-prompts", nargs="+", default=DEFAULT_PROMPTS,
                        help="object prompts (one output folder per prompt; "
                             "default: %(default)s)")
    parser.add_argument("--top-k", type=int, default=128,
                        help="final keypoints kept per prompt per sub-task "
                             "(default: %(default)s)")
    parser.add_argument("--bbox-scale", type=float, default=1.5,
                        help="enlargement factor of the bounding-box crops fed "
                             "to RoMAv2 (centered, clamped to the frame) "
                             "(default: %(default)s)")
    parser.add_argument("--num-corresp", type=int, default=2000,
                        help="RoMAv2 candidate points sampled in the anchor "
                             "crop before filtering (default: %(default)s)")
    parser.add_argument("--strategy", choices=("reference", "cycle"),
                        default="reference",
                        help="RoMAv2 matching strategy (default: %(default)s)")
    parser.add_argument("--no-rexomni", action="store_true",
                        help="skip the Step-3a detections JSON: SAM3 uses "
                             "text-only prompts and the key-frames are "
                             "discovered from disk directly")
    parser.add_argument("--detections-dir", default=None,
                        help="Step-3a detections root (default: "
                             "<out-dir>/detections)")
    parser.add_argument("--max-keyframes", type=int, default=8,
                        help="cap the key-frames per sub-task (evenly spaced, "
                             "applied to the Step-3a list or the discovered "
                             "frames); None disables the cap (default: "
                             "%(default)s)")
    parser.add_argument("--device", default=None, choices=["cuda", "cpu"],
                        help="device (default: auto)")
    parser.add_argument("--skip-done", action="store_true",
                        help="skip sub-tasks whose prompt output already exists")
    parser.add_argument("--no-viz", action="store_true",
                        help="skip the viz.png rendering")
    return parser.parse_args(argv)


class InitPointsExtract:
    """Per-sub-task SAM3 masks + RoMAv2 keypoints on the saved key-frames.

    The key-frames are the jpgs written by extract_frames.py --mode
    key_frames (sub-task segments and frame indices come from the folder and
    file names) — no dataset or video access is needed. The Step-3a
    detections JSON provides the per-key-frame object boxes (SAM3 prompt)
    and the key-frame indices; without it (--no-rexomni) the frames are
    discovered from disk directly.
    """

    def __init__(self, args):
        self.args = args
        self.folder_mode = args.keyframes_dir is not None
        if self.folder_mode:
            self.folder = Path(args.keyframes_dir)
            if not self.folder.is_dir():
                raise FileNotFoundError(f"key-frames folder {self.folder} "
                                        f"missing")
            self.folder_frames = discover_folder_frames(self.folder)
            self.folder_map = dict(self.folder_frames)
            self.ep_idxes = [args.episode_idx]
            self.out_dir = args.out_dir or str(self.folder.parent
                                               / "step3_output")
            print(f"key-frames folder: {self.folder} "
                  f"({len(self.folder_frames)} image(s))")
        else:
            if not args.data_root:
                sys.exit("--data-root is required (or --keyframes-dir to "
                         "run on a single folder of key-frame images)")
            self.root = Path(args.keyframes_root) if args.keyframes_root \
                else keyframes_root(args.data_root)
            if not self.root.is_dir():
                raise FileNotFoundError(
                    f"key-frames root {self.root} missing: run Step 1 first "
                    f"(python tools/astribot/extract_frames.py --mode detect_subtask "
                    f"then --mode key_frames --camera-idxes <camera>)")
            self.ep_idxes = select_episodes(self.root, args.episode_idxes,
                                            args.max_episodes)
            self.out_dir = args.out_dir or (args.data_root + "/eps_data")
            print(f"key-frames: {self.root}")
            print(f"episodes on disk: {len(discover_episodes(self.root))} -> "
                  f"{len(self.ep_idxes)} selected")
        self.init_dir = os.path.join(self.out_dir, "init_points")
        os.makedirs(self.init_dir, exist_ok=True)
        if args.detections_dir is None:
            args.detections_dir = os.path.join(self.out_dir, "detections")
        self._device = args.device or ("cuda" if torch.cuda.is_available()
                                       else "cpu")
        self.sam3 = None
        self.romav2 = None

    # --- model setup --------------------------------------------------------

    def _ensure_sam3(self) -> Sam3Image:
        if self.sam3 is None:
            print(f"Loading SAM3 ({DEFAULT_SAM3_CKPT})...")
            self.sam3 = Sam3Image(DEFAULT_SAM3_CKPT, device=self._device)
        return self.sam3

    def _ensure_romav2(self) -> RoMaV2PT2:
        if self.romav2 is None:
            print(f"Loading RoMAv2 ({DEFAULT_ROMAV2_CKPT})...")
            self.romav2 = RoMaV2PT2(DEFAULT_ROMAV2_CKPT, device=self._device)
        return self.romav2

    # --- key-frames -----------------------------------------------------------

    def _load_keyframe(self, k: int, t: int) -> np.ndarray:
        """RGB uint8 HWC array of saved key-frame t (SAM3 / RoMAv2 take
        RGB numpy arrays)."""
        if self.folder_mode:
            path = self.folder_map.get(t)
            if path is None:
                raise FileNotFoundError(
                    f"key-frame {t} not in {self.folder} "
                    f"(indices {sorted(self.folder_map)})")
        else:
            path = keyframe_path(self.root, self.ep_idx, self.cam_key, k, t)
        if not path.is_file():
            raise FileNotFoundError(
                f"key-frame {path} missing: run extract_frames.py "
                f"--mode key_frames for episode {self.ep_idx}")
        return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)

    # --- Step-3a detections ---------------------------------------------------

    def _detections_path(self, ep_idx: int) -> Path:
        return Path(self.args.detections_dir) / f"ep{ep_idx:06d}.json"

    def _load_detections(self, ep_idx: int) -> dict:
        """The Step-3a detections JSON of the episode."""
        path = self._detections_path(ep_idx)
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} missing: run Step 3a first "
                "(.venv-rexomni/bin/python tools/general_test/"
                "run_object_detection.py ...), or pass --no-rexomni to use "
                "SAM3 text-only prompts")
        with open(path) as f:
            return json.load(f)

    # --- SAM3 ------------------------------------------------------------------

    def _sam3_predict(self, rgb: np.ndarray, prompt: str,
                      boxes_xyxy: list[list[float]]
                      ) -> tuple[np.ndarray | None, np.ndarray | None, float | None]:
        """SAM3 union mask of the object in one key-frame.

        Args:
            rgb: (H, W, 3) uint8 RGB frame.
            prompt: SAM3 text prompt.
            boxes_xyxy: detection boxes (absolute pixels) for this frame;
                empty -> text-only prompting.

        Returns:
            (mask (H, W) bool | None, best box (4,) xyxy | None, score | None)
            — None when SAM3 produced nothing.
        """
        h, w = rgb.shape[:2]
        if boxes_xyxy:
            boxes_xyxy = sorted(boxes_xyxy, key=lambda b: max(0.0, b[2] - b[0])
                                * max(0.0, b[3] - b[1]), reverse=True)
            boxes_xyxy = boxes_xyxy[:SAM3_BOXES_MAX]
            norm_boxes = normalize_bbox(
                box_xyxy_to_cxcywh(torch.tensor(boxes_xyxy,
                                                dtype=torch.float32)), w, h)
            labels = [True] * len(boxes_xyxy)
        else:
            norm_boxes, labels = None, None

        # Restore the SAM3 trace-time environment around the call (the
        # exported graph was traced with TF32 and the flash/mem-efficient
        # SDPA kernels enabled, e.g. its attention output is viewed with
        # trace-time strides) — same pattern as infer_tapip3d.py.
        flash_sdp = torch.backends.cuda.flash_sdp_enabled()
        mem_eff_sdp = torch.backends.cuda.mem_efficient_sdp_enabled()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        try:
            state = self._ensure_sam3().predict(
                rgb, text_prompt=prompt, boxes=norm_boxes, labels=labels)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.enable_flash_sdp(flash_sdp)
            torch.backends.cuda.enable_mem_efficient_sdp(mem_eff_sdp)

        masks = state["masks"]  # (N, 1, H, W) bool at the frame size
        if masks.shape[0] == 0:
            return None, None, None
        best = int(state["scores"].argmax())
        return (masks.any(dim=0)[0].cpu().numpy(),
                state["boxes"][best].cpu().numpy(),
                float(state["scores"][best]))

    # --- RoMAv2 -----------------------------------------------------------------

    def _enlarge_crop(self, rgb: np.ndarray, box_xyxy: list[float]
                      ) -> tuple[np.ndarray, tuple[int, int]] | tuple[None, None]:
        """Crop the region around a bounding box, enlarged by --bbox-scale
        (centered, clamped to the frame). Returns (crop, (ox, oy)) or
        (None, None) for a degenerate box."""
        h, w = rgb.shape[:2]
        x0, y0, x1, y1 = box_xyxy
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        bw, bh = (x1 - x0) * self.args.bbox_scale, (y1 - y0) * self.args.bbox_scale
        c0x, c0y = int(max(0, cx - bw / 2)), int(max(0, cy - bh / 2))
        c1x, c1y = int(min(w, cx + bw / 2)), int(min(h, cy + bh / 2))
        if c1x - c0x < 4 or c1y - c0y < 4:
            return None, None
        return np.ascontiguousarray(rgb[c0y:c1y, c0x:c1x]), (c0x, c0y)

    def _match_object(self, crops: list[np.ndarray],
                      crop_masks: list[np.ndarray | None] | None = None
                      ) -> np.ndarray | None:
        """RoMAv2 matching of the crops: returns (K, N, 2) pixel coordinates
        in crop space (already ranked by worst-case overlap), or None.

        crop_masks: per-crop object masks (one per crop, already cropped with
        the same box as the image; None = unconstrained). Passed to RoMAv2 so
        the sampled points lie inside the object masks.
        """
        positions, _ = self._ensure_romav2().match(
            crops, strategy=self.args.strategy,
            num_corresp=self.args.num_corresp,
            overlap_th=None, top_k=self.args.top_k * 4, cycle_th=CYCLE_TH,
            masks=crop_masks)
        if positions.shape[0] == 0:
            return None
        dims = [c.shape[:2] for c in crops]
        return torch.stack(
            [to_pixel(positions[:, j], H=dims[j][0], W=dims[j][1])
             for j in range(len(crops))], dim=1).cpu().numpy()

    def _filter_top_k(self, matches_crop: np.ndarray, masks: np.ndarray,
                      offsets: list[tuple[int, int]], h: int, w: int
                      ) -> tuple[np.ndarray, int]:
        """Full-frame keypoints + the required in-mask frame count.

        A track must lie inside the object mask on at least half of the
        key-frames that have a mask (ceil, min 1) — objects are occluded or
        move between key-frames, so a strict all-frames check would drop
        every point. Survivors are capped at --top-k; they are already
        ranked by worst-case overlap across the key-frames.
        """
        n = matches_crop.shape[1]
        full = matches_crop.copy()
        for j in range(n):
            full[:, j, 0] += offsets[j][0]
            full[:, j, 1] += offsets[j][1]
        masked = [j for j in range(n) if masks[j].any()]
        need = max(1, int(np.ceil(len(masked) / 2)))
        keep = []
        for i in range(full.shape[0]):
            hits = 0
            for j in masked:
                x, y = int(round(full[i, j, 0])), int(round(full[i, j, 1]))
                if 0 <= x < w and 0 <= y < h and masks[j, y, x]:
                    hits += 1
            if hits >= need:
                keep.append(i)
        return full[keep][: self.args.top_k], need

    # --- orchestration ---------------------------------------------------------

    def run(self) -> None:
        print(f"\n{len(self.ep_idxes)} episode(s) selected: {self.ep_idxes}")
        for ep_idx in tqdm(self.ep_idxes, desc="episodes"):
            self._process_episode(ep_idx)
        print(f"\ndone: {len(self.ep_idxes)} episode(s) -> {self.init_dir}")

    def _process_episode(self, ep_idx: int) -> None:
        if self.folder_mode:
            self._process_folder(ep_idx)
            return
        self.ep_idx = ep_idx
        detections = None
        if not self.args.no_rexomni:
            detections = self._load_detections(ep_idx)
            print(f"\nepisode {ep_idx}: detections loaded from "
                  f"{self._detections_path(ep_idx)}")
        # camera: the one Step 3a used when its JSON is loaded, else an
        # explicit --camera-keys entry or the first RGB camera on disk
        cam = (detections or {}).get("camera_key")
        if cam is None:
            cam = select_camera(self.root, ep_idx, self.args.camera_keys)
        elif cam not in camera_subdirs(self.root, ep_idx):
            raise FileNotFoundError(
                f"Step-3a camera {cam!r} of episode {ep_idx} has no key-frames "
                f"on disk ({camera_subdirs(self.root, ep_idx)}); run "
                f"extract_frames.py --mode key_frames for that camera")
        self.cam_key = cam
        print(f"episode {ep_idx} (camera {cam})")

        if detections is not None:
            # key-frames come from the Step-3a JSON so both steps agree
            items = []
            for k, sub in sorted(detections.get("subtasks", {}).items(),
                                 key=lambda kv: int(kv[0])):
                keys = cap_keyframes(
                    [int(t) for t in sub.get("keyframes", [])],
                    self.args.max_keyframes)
                if not keys:
                    print(f"  [subtask {k:02d}] skip: no key-frames in the "
                          f"Step-3a JSON")
                    continue
                items.append((int(k), keys, sub.get("detections") or {}))
        else:
            # --no-rexomni: discover the saved key-frames directly
            frames_by_sub = discover_subtask_frames(self.root, ep_idx, cam)
            items = [(k, cap_keyframes(frames, self.args.max_keyframes), None)
                     for k, frames in sorted(frames_by_sub.items()) if frames]
        for k, keys, seg_dets in items:
            self._process_segment(k, keys, seg_dets)

    def _process_folder(self, ep_idx: int) -> None:
        """The folder of key-frame images = one sub-task (subtask 00) of a
        synthetic episode labelled --episode-idx; same outputs as the episode
        mode."""
        self.ep_idx = ep_idx
        detections = None
        if not self.args.no_rexomni:
            detections = self._load_detections(ep_idx)
            print(f"\nepisode {ep_idx}: detections loaded from "
                  f"{self._detections_path(ep_idx)}")
        # camera: the one Step 3a recorded (when its JSON is loaded), else an
        # explicit --camera-key, else the folder name
        cam = (detections or {}).get("camera_key") or \
            self.args.camera_key or self.folder.name
        self.cam_key = cam
        print(f"episode {ep_idx} (folder {self.folder.name}, camera {cam})")
        if detections is not None:
            # key-frames come from the Step-3a JSON so both steps agree; they
            # must still be files of the input folder
            sub = (detections.get("subtasks") or {}).get("0")
            if sub is None:
                raise FileNotFoundError(
                    f"no subtask \"0\" in {self._detections_path(ep_idx)}")
            keys = cap_keyframes([int(t) for t in sub.get("keyframes", [])],
                                 self.args.max_keyframes)
            if not keys:
                print(f"  skip: no key-frames in the Step-3a JSON")
                return
            missing = [t for t in keys if t not in self.folder_map]
            if missing:
                raise FileNotFoundError(
                    f"key-frame(s) {missing} of the Step-3a JSON not in "
                    f"{self.folder} (indices {sorted(self.folder_map)})")
            items = [(0, keys, sub.get("detections") or {})]
        else:
            # --no-rexomni: discover the folder's images directly
            keys = cap_keyframes([i for i, _ in self.folder_frames],
                                 self.args.max_keyframes)
            if not keys:
                print(f"  skip: no key-frame images in {self.folder}")
                return
            items = [(0, keys, None)]
        for k, keys, seg_dets in items:
            self._process_segment(k, keys, seg_dets)

    def _process_segment(self, k: int, keys: list[int],
                         seg_dets: dict | None) -> None:
        seg_dir = os.path.join(self.init_dir, f"ep{self.ep_idx:06d}",
                               f"subtask_{k:02d}")
        print(f"  [subtask {k:02d}] {len(keys)} key-frames {keys}")
        frames = [self._load_keyframe(k, t) for t in keys]
        for prompt in self.args.text_prompts:
            self._process_prompt(seg_dir, k, keys, frames, seg_dets, prompt)

    def _process_prompt(self, seg_dir: str, k: int,
                        keyframes: list[int], frames: list[np.ndarray],
                        seg_dets: dict | None, prompt: str) -> None:
        slug = re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")
        pdir = os.path.join(seg_dir, slug)
        os.makedirs(pdir, exist_ok=True)
        npz_path = os.path.join(pdir, "init_points.npz")
        if self.args.skip_done and os.path.isfile(npz_path):
            print(f"    [{slug}] skip: {npz_path} exists")
            return

        n = len(keyframes)
        h, w = frames[0].shape[:2]
        empty_reason = None
        in_mask_need = 0
        if n < 2:
            empty_reason = "insufficient keyframes"

        # SAM3 masks + best box per key-frame (text-only when the frame has
        # no detection).
        masks = np.zeros((n, h, w), dtype=bool)
        boxes = np.zeros((n, 4), dtype=np.float32)
        scores = np.full(n, -1.0, dtype=np.float32)
        any_mask = False
        for j, (t, rgb) in enumerate(zip(keyframes, frames)):
            det_boxes: list[list[float]] = []
            if seg_dets:
                preds = (seg_dets.get(str(t)) or {}).get(prompt) or []
                det_boxes = [d["coords"] for d in preds
                             if d.get("type") == "box"]
            m, b, s = self._sam3_predict(rgb, prompt, det_boxes)
            if m is not None:
                masks[j] = m
                any_mask = True
            if b is not None:
                boxes[j] = b
                scores[j] = s if s is not None else -1.0

        # Per-key-frame crop boxes: each key-frame is cropped around its
        # own box (the object can move between key-frames) — the frame's
        # best SAM3 box, else its largest detection box, else the first box
        # available anywhere.
        first_box: list[float] | None = None
        crop_boxes: list[list[float] | None] = []
        for j, t in enumerate(keyframes):
            b = boxes[j].tolist() if boxes[j].any() else None
            if b is None and seg_dets:
                preds = (seg_dets.get(str(t)) or {}).get(prompt) or []
                dets = [d["coords"] for d in preds if d.get("type") == "box"]
                if dets:
                    b = max(dets, key=lambda bb: max(0.0, bb[2] - bb[0])
                            * max(0.0, bb[3] - bb[1]))
            if b is None:
                b = first_box
            else:
                first_box = first_box or b
            crop_boxes.append(b)
        if first_box is not None:
            # backfill frames without any box (e.g. leading frames before
            # the first detection) with the first available box
            crop_boxes = [b if b is not None else first_box
                          for b in crop_boxes]
        if empty_reason is None and (not any_mask or first_box is None):
            empty_reason = "no mask or box"

        # RoMAv2 matching on the enlarged-bbox crops of all key-frames. The
        # full-frame mask is cropped with the same box as the image and fed to
        # RoMAv2, so points are sampled inside the object only (frames without
        # a mask stay unconstrained).
        keypoints = np.zeros((0, n, 2), dtype=np.float32)
        if empty_reason is None:
            crops, offsets, crop_masks = [], [], []
            for rgb, cb, m in zip(frames, crop_boxes, masks):
                crop, off = self._enlarge_crop(rgb, cb)
                if crop is None:
                    empty_reason = "degenerate crop"
                    break
                crops.append(crop)
                offsets.append(off)
                c0x, c0y = off
                crop_masks.append(
                    m[c0y:c0y + crop.shape[0], c0x:c0x + crop.shape[1]]
                    if m.any() else None)
            if empty_reason is None:
                matches = self._match_object(crops, crop_masks)
                if matches is None:
                    empty_reason = "no matches"
                else:
                    keypoints, in_mask_need = self._filter_top_k(
                        matches, masks, offsets, h, w)

        # Save (uniform schema; failures are recorded in init_points.json).
        np.savez(npz_path,
                 keypoints=keypoints,
                 frame_indices=np.asarray(keyframes, dtype=np.int64),
                 masks=masks, boxes=boxes, scores=scores)
        rle = {str(t): encode_rle(masks[j])
               for j, t in enumerate(keyframes) if masks[j].any()}
        with open(os.path.join(pdir, "masks_rle.json"), "w") as f:
            json.dump(rle, f, indent=2)
        meta = {
            "episode": int(self.ep_idx),
            "subtask": int(k),
            "segment": [int(min(keyframes)), int(max(keyframes)) + 1],
            "camera_key": self.cam_key,
            "prompt": prompt,
            "prompt_slug": slug,
            "keyframes": [int(t) for t in keyframes],
            "num_keypoints": int(len(keypoints)),
            "top_k": self.args.top_k,
            "bbox_scale": self.args.bbox_scale,
            "num_corresp": self.args.num_corresp,
            "match_top_k": self.args.top_k * 4,
            "in_mask_min_frames": int(in_mask_need),
            "strategy": self.args.strategy,
            "detections_file": str(self._detections_path(self.ep_idx))
                                if not self.args.no_rexomni else None,
            "sam3_checkpoint": DEFAULT_SAM3_CKPT,
            "romav2_checkpoint": DEFAULT_ROMAV2_CKPT,
            "empty_reason": empty_reason,
        }
        with open(os.path.join(pdir, "init_points.json"), "w") as f:
            json.dump(meta, f, indent=2)
        if not self.args.no_viz and len(keypoints):
            self._visualize(pdir, frames, keypoints, masks, boxes, prompt)
        status = empty_reason or f"{len(keypoints)} keypoints"
        print(f"    [{slug}] {status} -> {pdir}")

    def _visualize(self, pdir: str, frames: list[np.ndarray],
                   keypoints: np.ndarray, masks: np.ndarray,
                   boxes: np.ndarray, prompt: str) -> None:
        """Key-frames side-by-side with the masks, boxes and the tracks."""
        imgs = [f[:, :, ::-1] for f in frames]  # RGB -> BGR for cv2
        H0, W0 = imgs[0].shape[:2]
        imgs = [cv2.resize(im, (W0, H0)) for im in imgs]
        stacked = np.hstack(imgs)
        for m, pts in enumerate(keypoints):
            color = tuple(int(c) for c in _PALETTE[m % len(_PALETTE)])
            for j, (x, y) in enumerate(pts):
                cx, cy = int(x) + j * W0, int(y)
                cv2.circle(stacked, (cx, cy), 3, color, -1)
                if j > 0:
                    px, py = (int(pts[j - 1][0]) + (j - 1) * W0,
                              int(pts[j - 1][1]))
                    cv2.line(stacked, (px, py), (cx, cy), color, 1)
        for j in range(len(imgs)):
            if masks[j].any():
                red = np.zeros_like(stacked)
                m = masks[j]
                if m.shape[:2] != (H0, W0):
                    m = cv2.resize(m.astype(np.uint8), (W0, H0)).astype(bool)
                # the stacked canvas holds frame j at columns [j*W0, (j+1)*W0)
                red[:, j * W0:(j + 1) * W0][m] = (0, 0, 255)
                stacked = cv2.addWeighted(stacked, 1.0, red, 0.35, 0)
            if boxes[j].any():
                x0, y0, x1, y1 = [int(v) for v in boxes[j]]
                cv2.rectangle(stacked, (x0 + j * W0, y0), (x1 + j * W0, y1),
                              (0, 255, 0), 2)
        cv2.imwrite(os.path.join(pdir, "viz.png"), stacked)


def main():
    InitPointsExtract(parse_args()).run()


if __name__ == "__main__":
    main()
