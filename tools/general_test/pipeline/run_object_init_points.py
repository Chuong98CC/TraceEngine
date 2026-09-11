"""Step 3b — SAM3 masks + RoMAv2 keypoints on the saved sub-task key-frames
(initial keypoints of the interacting objects).

Keypoint stage of pipeline Step 3, continuing from the Step-3a detections:
for every prompt of every sub-task it segments the object on the sub-task's
key-frames with SAM3 (boxes + text prompt from the detections, text-only
when a frame has no detection), matches keypoints across the key-frames
with RoMAv2 on enlarged box crops and keeps the top-k points inside the
object masks:

    detections JSON (per-sub-task prompts + boxes)
               │
               ▼  SAM3 masks, then RoMAv2 keypoints per prompt
    ┌──────────────────────────────┐
    │  init_points per prompt      │
    └──────────────────────────────┘

Per prompt: init_points.npz (top-k keypoints), masks_rle.json (SAM3
masks as COCO RLE), viz.png (key-frames + masks + tracks) — in episode
mode under the sub-task's own sampling_points folder of the episodes tree
(utils.astribot_paths), one subtree per camera, mirroring the per-sub-task
detections JSONs of Step 3a:

    <out-dir>/ep{ep:03d}/subtask_{k:02d}/sampling_points/init_points/<camera>/<prompt_slug>/

(folder mode writes the flat .../subtask_{XX}/<prompt_slug>/ — the folder
is the camera).

Prompts are read per sub-task from the Step-3a detections JSON (Step 3a
recorded them from the dataset's meta/subtasks.csv in episode mode —
together with each prompt's role, the column it was read from — or from
its --text-prompts in folder mode, without roles).

Episode mode processes every --camera-keys entry with a Step-3a detections
JSON under the episode's sampling_points folders (default: all such
cameras). An
**object** prompt is matched only between the sub-task's 2nd and
2nd-to-last key-frame (the gripper close .. open transport span, where
the object is static on the dropped boundary frames anyway); the
**manipulator** — and any prompt whose role is unrecorded (folder mode,
older Step-3a JSONs) — is matched across all key-frames. There is no
JSON-less fallback — run Step 3a first.

By default RoMAv2
--sampling-mode mask: samples its candidate points inside the object masks
--sampling-mode uniform: samples over the whole enlarged crops instead
and the in-mask top-k filter alone decides —same crops and output criterion,
different pool.
--sampling-mode no_roma: is the simple baseline: no RoMAv2 — top-k points
are uniformly sampled inside the mask of the span's first frame only
(the manipulator's 1st key-frame, the object's 2nd — the first of its transport span),

--with-optical-flow-mask (episode mode, off by default) unions the
sub-task's WAFT motion mask (Step 3a', run_step3_motion_masks.py, read
from the sub-task's own init_points/<camera>/ folder of this tree:
.../subtask_{k:02d}/sampling_points/init_points/<camera>/motion_rle.json)
into the
**manipulator** prompt's row-0 SAM3 mask before sampling — the SAM ∪
optical-flow mask that rescues the arm when the Step-3a detection missed
it. The flow mask is anchored at the manipulator's 1st key-frame (the
same frame row 0 samples from), so it ORs in-place. Step 3a' saves the
mask only for significant pairs, so a sub-task without one (no file)
simply keeps its SAM mask; a stale/mismatched mask is skipped with a
note. --viz-motion-union renders that union (union_mask.png next to the
manipulator prompt's init points: the row-0 key-frame tinted SAM-only /
SAM ∩ motion / motion-only) — the motion-only pixels being exactly what
the rescue added.


Examples
--------
    # From the Step-3a detections JSON
    python tools/general_test/pipeline/run_object_init_points.py
        --data-root /data/astribot_making_coffee_vlva_full --episode-idxes 0

    # One sub-task's key-frame folder (the prompts are the ones a Step 3a
    # run on the same folder recorded in its JSON)
    python tools/general_test/pipeline/run_object_init_points.py
        --keyframes-dir .../subtask_00/cam_head --episode-idx 0
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import zlib
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
from utils import astribot_paths as ap
from utils.keyframe_utils import (
    camera_subdirs,
    cap_keyframes,
    discover_episodes,
    discover_folder_frames,
    keyframe_path,
    select_episodes,
)
from utils.file_io.mask_rle import decode_rle, encode_rle

DEFAULT_SAM3_CKPT = "weights/sam3/sam3_image_exported_bf16.pt2"
DEFAULT_ROMAV2_CKPT = "weights/romav2/romav2.pt2"
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


def _prompt_top_k(top_k: int, manipulator_top_k: int | None,
                  role: str | None) -> int:
    """Per-prompt keypoint cap of one prompt: the manipulator role samples
    ``manipulator_top_k`` (falling back to ``top_k``), every other prompt
    ``top_k``. The manipulator is sampled denser so enough of its points
    survive the Step-4 static filters (--filter-static-*)."""
    return (manipulator_top_k or top_k) if role == "manipulator" else top_k


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Step 3b: SAM3 masks + RoMAv2 keypoints on the saved "
                    "sub-task key-frames (uses the Step-3a detections JSON)."
    )
    parser.add_argument("--data-root", "-d", default=None,
                        help="root of the local dataset copy; the default "
                             "episodes root derives from it (not used with "
                             "--keyframes-dir)")
    parser.add_argument("--keyframes-dir", default=None,
                        help="run on a single folder of key-frame images "
                             "(one sub-task of one camera) instead of the "
                             "episode layout; exclusive with --data-root / "
                             "--out-dir")
    parser.add_argument("--episode-idx", type=int, default=0,
                        help="episode index labelling the outputs (folder "
                             "mode only; default: %(default)s)")
    parser.add_argument("--camera-key", default=None,
                        help="camera key recorded in the outputs (folder "
                             "mode only; default: the folder name)")
    parser.add_argument("--camera-keys", nargs="+", default=None,
                        help="camera subdir names (e.g. cam_head) to process, "
                             "in order; entries without a Step-3a detections "
                             "JSON are skipped (default: every camera Step "
                             "3a detected for the episode)")
    parser.add_argument("--episode-idxes", "-e", nargs="*", type=int, default=None,
                        help="only process these episode indices (default: all "
                             "episodes with key-frames on disk)")
    parser.add_argument("--max-episodes", "-x", type=int, default=None,
                        help="cap the number of processed episodes")
    parser.add_argument("--out-dir", "-o", default=None,
                        help="episodes root (default: <data-root>/episodes); "
                             "results land per sub-task under "
                             "<out-dir>/ep{ep:03d}/subtask_{k:02d}/"
                             "sampling_points/init_points/<camera>/"
                             "<prompt_slug>/ (folder mode: the output root "
                             "next to the key-frame folder it was given; "
                             "default: <keyframes-dir>/../step3_output)")
    parser.add_argument("--object-top-k", type=int, default=128,
                        help="final keypoints kept per object prompt per "
                             "sub-task (also the cap of role-less prompts, "
                             "e.g. folder mode; default: %(default)s)")
    parser.add_argument("--manipulator-top-k", type=int, default=None,
                        help="final keypoints kept per manipulator-role "
                             "prompt per sub-task — the manipulator is "
                             "sampled denser so enough of its points "
                             "survive the Step-4 static filters "
                             "(overrides --object-top-k for the "
                             "manipulator role; default: same as "
                             "--object-top-k)")
    parser.add_argument("--bbox-scale", type=float, default=1.5,
                        help="enlargement factor of the bounding-box crops fed "
                             "to RoMAv2 (centered, clamped to the frame) "
                             "(default: %(default)s)")
    parser.add_argument("--num-corresp", type=int, default=2000,
                        help="RoMAv2 candidate points sampled in the anchor "
                             "crop before filtering (default: %(default)s)")
    parser.add_argument("--sampling-mode",
                        choices=("mask", "uniform", "no_roma"),
                        default="mask",
                        help="where RoMAv2 samples its candidate points: "
                             "'mask' (default) constrains the pool inside "
                             "the object masks by passing them to RoMAv2; "
                             "'uniform' samples over the whole enlarged "
                             "crops instead and the in-mask top-k filter "
                             "alone decides (same crops, same filter — "
                             "run both modes into separate --out-dirs to "
                             "compare); 'no_roma' skips RoMAv2 entirely — "
                             "the simple baseline uniformly sampling "
                             "top-k points inside the mask of the span's "
                             "first frame only (manipulator: the 1st "
                             "key-frame; object: the 2nd, first of its "
                             "transport span), same masks and outputs "
                             "otherwise")
    parser.add_argument("--strategy", choices=("reference", "cycle"),
                        default="reference",
                        help="RoMAv2 matching strategy (default: %(default)s)")
    parser.add_argument("--detections-dir", default=None,
                        help="Step-3a detections root of the folder mode "
                             "(default: <out-dir>/detections); episode mode "
                             "reads the detections from the fixed episodes "
                             "layout under --out-dir")
    parser.add_argument("--max-keyframes", type=int, default=8,
                        help="cap the key-frames per sub-task (evenly spaced, "
                             "applied to the Step-3a list or the discovered "
                             "frames); None disables the cap (default: "
                             "%(default)s)")
    parser.add_argument("--with-optical-flow-mask", action="store_true",
                        help="union the sub-task's WAFT motion mask (Step "
                             "3a', motion_rle.json read from the sub-task's "
                             "own init_points/<camera>/ folder) into the "
                             "manipulator prompt's row-0 SAM3 mask before "
                             "sampling — the SAM + optical-flow rescue of a "
                             "manipulator the detection missed (episode mode "
                             "only; default: off)")
    parser.add_argument("--init-points-dir", default=None,
                        help="init-points root of the folder mode — where "
                             "the per-prompt outputs land (default: "
                             "<out-dir>/init_points); episode mode writes the "
                             "fixed episodes layout under --out-dir")
    parser.add_argument("--viz-motion-union", action="store_true",
                        help="render the manipulator's SAM ∪ motion union "
                             "(union_mask.png next to that prompt's init "
                             "points): the row-0 key-frame tinted SAM-only / "
                             "SAM ∩ motion / motion-only, i.e. the pixels the "
                             "flow rescue added — requires "
                             "--with-optical-flow-mask (default: off)")
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
    detections JSON provides the per-key-frame object boxes (SAM3 prompt),
    the key-frame indices and the per-sub-task prompts.
    """

    def __init__(self, args):
        self.args = args
        self.folder_mode = args.keyframes_dir is not None
        if self.folder_mode and args.with_optical_flow_mask:
            sys.exit("--with-optical-flow-mask needs the dataset (the WAFT "
                     "motion masks are computed online from it): the "
                     "--keyframes-dir folder mode cannot run it")
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
            if args.detections_dir or args.init_points_dir:
                sys.exit("--detections-dir / --init-points-dir are "
                         "folder-mode only: episode mode reads and writes "
                         "the fixed episodes layout under --out-dir")
            # episode mode reads the key-frames, the Step-3a detections and
            # the Step-3a' motion masks, and writes the init points, all
            # inside the episodes tree (see utils.astribot_paths)
            self.root = Path(args.out_dir) if args.out_dir \
                else ap.episodes_root(args.data_root)
            if not self.root.is_dir():
                raise FileNotFoundError(
                    f"episodes root {self.root} missing: run Step 1 first "
                    f"(python tools/astribot/extract_frames.py --mode detect_subtask "
                    f"then --mode key_frames --camera-idxes <camera>)")
            self.ep_idxes = select_episodes(self.root, args.episode_idxes,
                                            args.max_episodes)
            print(f"episodes root: {self.root}")
            print(f"episodes on disk: {len(discover_episodes(self.root))} -> "
                  f"{len(self.ep_idxes)} selected")
        if self.folder_mode:
            # folder mode is a flat tree next to the key-frame folder it was
            # given (the folder is the camera, its one sub-task is 0)
            self.init_dir = str(args.init_points_dir) if args.init_points_dir \
                else os.path.join(self.out_dir, "init_points")
            os.makedirs(self.init_dir, exist_ok=True)
            if args.detections_dir is None:
                args.detections_dir = os.path.join(self.out_dir, "detections")
        # where the outputs land, for the closing print
        self.dest = Path(self.init_dir) if self.folder_mode else self.root
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

    def _detections_path(self, ep_idx: int, k: int,
                         cam: str | None = None) -> Path:
        """Step-3a detections JSON of one (episode, sub-task, camera):
        episode mode reads the sub-task's own file under its sampling_points
        folder (…/subtask_{k:02d}/sampling_points/detections/<camera>.json —
        that file *is* the sub-task); folder mode reads the flat ep{ep}.json
        Step 3a wrote next to the folder it was given (the folder is the
        camera, its one sub-task 0)."""
        if self.folder_mode:
            return Path(self.args.detections_dir) / f"ep{ep_idx:06d}.json"
        return ap.detections_dir(self.root, ep_idx, k) / f"{cam}.json"

    def _load_detections(self, ep_idx: int, k: int,
                         cam: str | None = None) -> dict:
        """The Step-3a detections JSON of one (episode, sub-task, camera)."""
        path = self._detections_path(ep_idx, k, cam)
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} missing: run Step 3a first "
                "(.venv-rexomni/bin/python tools/general_test/"
                "run_object_detection.py ...)")
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
                      crop_masks: list[np.ndarray | None] | None = None,
                      top_k: int | None = None) -> np.ndarray | None:
        """RoMAv2 matching of the crops: returns (K, N, 2) pixel coordinates
        in crop space (already ranked by worst-case overlap), or None.

        crop_masks: per-crop object masks (one per crop, already cropped with
        the same box as the image; None = unconstrained). Passed to RoMAv2 so
        the sampled points lie inside the object masks. Pass None (or omit)
        for uniform sampling over the crops — the caller's in-mask top-k
        filter then decides alone (--sampling-mode uniform).

        top_k: the prompt's keypoint cap (default: --object-top-k; the RoMAv2
        candidate pool is 4x the cap).
        """
        if top_k is None:
            top_k = self.args.object_top_k
        positions, _ = self._ensure_romav2().match(
            crops, strategy=self.args.strategy,
            num_corresp=self.args.num_corresp,
            overlap_th=None, top_k=top_k * 4, cycle_th=CYCLE_TH,
            masks=crop_masks)
        if positions.shape[0] == 0:
            return None
        dims = [c.shape[:2] for c in crops]
        return torch.stack(
            [to_pixel(positions[:, j], H=dims[j][0], W=dims[j][1])
             for j in range(len(crops))], dim=1).cpu().numpy()

    def _filter_top_k(self, matches_crop: np.ndarray, masks: np.ndarray,
                      offsets: list[tuple[int, int]], h: int, w: int,
                      top_k: int | None = None) -> tuple[np.ndarray, int]:
        """Full-frame keypoints + the required in-mask frame count.

        A track must lie inside the object mask on at least half of the
        key-frames that have a mask (ceil, min 1) — objects are occluded or
        move between key-frames, so a strict all-frames check would drop
        every point. Survivors are capped at the prompt's top-k (default:
        --object-top-k); they are already ranked by worst-case overlap across the
        key-frames.
        """
        if top_k is None:
            top_k = self.args.object_top_k
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
        return full[keep][:top_k], need

    def _sample_in_mask(self, mask: np.ndarray, n: int, seed: int
                        ) -> np.ndarray:
        """n distinct pixels uniformly sampled inside a boolean mask.

        Returns (K, 2) float32 full-frame x/y coordinates, K = min(n, mask
        pixels) — the deterministic no_roma baseline (--sampling-mode
        no_roma samples its points here instead of matching across the
        key-frames with RoMAv2).
        """
        ys, xs = np.nonzero(mask)
        k = min(n, len(xs))
        if k == 0:
            return np.zeros((0, 2), dtype=np.float32)
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(xs), size=k, replace=False)
        return np.stack([xs[idx], ys[idx]], axis=1).astype(np.float32)

    def _motion_mask_union(self, k: int, keyframes: list[int],
                           h: int, w: int) -> tuple[np.ndarray | None, dict]:
        """(motion mask of the sub-task, meta) of the manipulator's
        SAM-∪-flow rescue, or (None, {}) when there is nothing to union.

        Reads the sub-task's Step-3a' mask — the (sub-task, camera) init
        folder of the episodes tree, beside the prompt subtrees:
        .../subtask_{k:02d}/sampling_points/init_points/<camera>/motion_rle.json
        (COCO RLE + the provenance of the significant pair that produced
        it). Step 3a' writes the file only for significant pairs, so a
        missing file is the normal "no rescue" case. The mask must be
        anchored at the prompt's row-0 key-frame (its 1st — the flow
        window's fixed first frame) and match the frame resolution; a
        stale/mismatched mask is skipped with a note — it must never
        corrupt the SAM masks.
        """
        path = ap.init_points_dir(self.root, self.ep_idx, k,
                                  self.cam_key) / "motion_rle.json"
        if not path.is_file():
            print(f"    no significant motion mask for subtask {k} "
                  f"({path.parent}) — SAM mask only")
            return None, {}
        with open(path) as f:
            meta = json.load(f)
        frame_a, frame_b = (int(t) for t in meta["chosen"])
        mask = decode_rle(meta["mask"])
        if frame_a != keyframes[0]:
            print(f"    motion mask {path} anchored at frame {frame_a}, "
                  f"row-0 key-frame {keyframes[0]} — skipped")
            return None, {}
        if mask.shape != (h, w):
            print(f"    motion mask {path} is {mask.shape}, frame is "
                  f"({h}, {w}) — skipped")
            return None, {}
        return mask, {"motion_mask_file": str(path),
                      "motion_flow_pair": [frame_a, frame_b],
                      "motion_moving_pixels": int(meta.get("moving_pixels",
                                                           [-1])[-1])}

    # --- orchestration ---------------------------------------------------------

    def run(self) -> None:
        print(f"\n{len(self.ep_idxes)} episode(s) selected: {self.ep_idxes}")
        for ep_idx in tqdm(self.ep_idxes, desc="episodes"):
            self._process_episode(ep_idx)
        print(f"\ndone: {len(self.ep_idxes)} episode(s) -> {self.dest}")

    def _process_episode(self, ep_idx: int) -> None:
        if self.folder_mode:
            self._process_folder(ep_idx)
            return
        self.ep_idx = ep_idx
        # One pass per camera of the episode: the per-sub-task detections
        # JSONs Step 3a wrote (…/subtask_{k:02d}/sampling_points/
        # detections/<camera>.json) drive which cameras are processed —
        # every --camera-keys entry among them, or all of them.
        have = sorted({p.stem
                       for k in ap.discover_subtasks(self.root, ep_idx)
                       for p in ap.detections_dir(self.root, ep_idx,
                                                  k).glob("*.json")})
        if not have:
            raise FileNotFoundError(
                f"no Step-3a detections JSON under "
                f"{ap.episode_dir(self.root, ep_idx)}/*/sampling_points/"
                f"detections: run Step 3a first "
                "(.venv-rexomni/bin/python tools/general_test/"
                "run_object_detection.py ...)")
        if self.args.camera_keys:
            cams = [c for c in self.args.camera_keys if c in have]
            missing = [c for c in self.args.camera_keys if c not in have]
            if missing:
                print(f"episode {ep_idx}: camera(s) {missing} have no "
                      f"Step-3a detections JSON under the episode's "
                      f"sampling_points folders — skipped (have: {have})")
            if not cams:
                return
        else:
            cams = have
        for cam in cams:
            self._process_camera(cam)

    def _process_camera(self, cam: str) -> None:
        """3b of one (episode, camera): key-frames and prompts come from
        the sub-task's own Step-3a JSON so both steps agree (the key-frame
        indices and the per-sub-task prompts are recorded next to the
        detections). Outputs land per sub-task and camera under
        …/subtask_{k:02d}/sampling_points/init_points/<cam>/."""
        ep_idx = self.ep_idx
        if cam not in camera_subdirs(self.root, ep_idx):
            print(f"episode {ep_idx} (camera {cam}): skip, no key-frames "
                  f"saved for it (run extract_frames.py --mode key_frames "
                  f"for this camera)")
            return
        self.cam_key = cam
        items = []
        for k in ap.discover_subtasks(self.root, ep_idx):
            path = self._detections_path(ep_idx, k, cam)
            if not path.is_file():
                print(f"  [subtask {k:02d}] skip: no Step-3a detections "
                      f"JSON in {path}")
                continue
            print(f"\nepisode {ep_idx} (camera {cam}, subtask {k:02d}): "
                  f"detections loaded from {path}")
            sub = self._load_detections(ep_idx, k, cam)
            keys = cap_keyframes(
                [int(t) for t in sub.get("keyframes", [])],
                self.args.max_keyframes)
            if not keys:
                print(f"  [subtask {k:02d}] skip: no key-frames in the "
                      f"Step-3a JSON")
                continue
            prompts = sub.get("prompts") or []
            if not prompts:
                print(f"  [subtask {k:02d}] skip: no prompts recorded "
                      f"in the Step-3a JSON (re-run Step 3a — prompts "
                      f"are per-sub-task now)")
                continue
            # roles aligned with prompts (the meta/subtasks.csv column each
            # prompt was read from); a missing/mismatched list means every
            # prompt samples over all key-frames, as before
            prompt_roles = sub.get("prompt_roles") or []
            if prompt_roles and len(prompt_roles) != len(prompts):
                print(f"  [subtask {k:02d}] warning: prompt_roles length "
                      f"mismatch — sampling every prompt over all "
                      f"key-frames")
                prompt_roles = []
            items.append((int(k), keys, sub.get("detections") or {},
                          prompts, prompt_roles))
        for k, keys, seg_dets, prompts, prompt_roles in items:
            seg_dir = str(ap.init_points_dir(self.root, ep_idx, k, cam))
            self._process_segment(seg_dir, k, keys, seg_dets, prompts,
                                  prompt_roles)

    def _process_folder(self, ep_idx: int) -> None:
        """The folder of key-frame images = one sub-task (subtask 00) of a
        synthetic episode labelled --episode-idx; same outputs as the episode
        mode, written flat (the folder is the camera)."""
        self.ep_idx = ep_idx
        detections = self._load_detections(ep_idx, 0)
        print(f"\nepisode {ep_idx}: detections loaded from "
              f"{self._detections_path(ep_idx, 0)}")
        # camera: the one Step 3a recorded (its JSON records it), else an
        # explicit --camera-key, else the folder name
        cam = (detections or {}).get("camera_key") or \
            self.args.camera_key or self.folder.name
        self.cam_key = cam
        print(f"episode {ep_idx} (folder {self.folder.name}, camera {cam})")

        # key-frames come from the Step-3a JSON so both steps agree; they
        # must still be files of the input folder. The prompts are the ones
        # Step 3a recorded for the sub-task (the JSON of a key-frame folder
        # is that one sub-task — there is no sub-task mapping in it).
        keys = cap_keyframes(
            [int(t) for t in detections.get("keyframes", [])],
            self.args.max_keyframes)
        if not keys:
            print(f"  skip: no key-frames in the Step-3a JSON")
            return
        missing = [t for t in keys if t not in self.folder_map]
        if missing:
            raise FileNotFoundError(
                f"key-frame(s) {missing} of the Step-3a JSON not in "
                f"{self.folder} (indices {sorted(self.folder_map)})")
        prompts = detections.get("prompts") or []
        if not prompts:
            print("  skip: no prompts recorded for sub-task 0 in the "
                  "Step-3a JSON (re-run Step 3a — prompts are "
                  "per-sub-task now)")
            return
        items = [(0, keys, detections.get("detections") or {}, prompts, [])]
        for k, keys, seg_dets, prompts, prompt_roles in items:
            seg_dir = os.path.join(self.init_dir, f"ep{ep_idx:06d}",
                                   f"subtask_{k:02d}")
            self._process_segment(seg_dir, k, keys, seg_dets, prompts,
                                  prompt_roles)

    def _process_segment(self, seg_dir: str, k: int, keys: list[int],
                         seg_dets: dict | None, prompts: list[str],
                         prompt_roles: list[str] | None = None) -> None:
        print(f"  [subtask {k:02d}] {len(keys)} key-frames {keys}, "
              f"prompts {prompts}")
        if prompt_roles:
            print(f"    roles: {dict(zip(prompts, prompt_roles))}")
        frames = [self._load_keyframe(k, t) for t in keys]
        for i, prompt in enumerate(prompts):
            role = prompt_roles[i] if prompt_roles and i < len(prompt_roles) \
                else None
            self._process_prompt(seg_dir, k, keys, frames, seg_dets,
                                 prompt, role)

    def _process_prompt(self, seg_dir: str, k: int,
                        keyframes: list[int], frames: list[np.ndarray],
                        seg_dets: dict | None, prompt: str,
                        role: str | None = None) -> None:
        slug = re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")
        pdir = os.path.join(seg_dir, slug)
        os.makedirs(pdir, exist_ok=True)
        npz_path = os.path.join(pdir, "init_points.npz")
        if self.args.skip_done and os.path.isfile(npz_path):
            print(f"    [{slug}] skip: {npz_path} exists")
            return

        keyframes_all = keyframes          # full sub-task list (meta span)
        # Object prompts sample only the transport span: the 2nd to the
        # 2nd-to-last key-frame (gripper close .. open). The object is
        # static on the dropped boundary frames (sub-task start/end), so
        # points there would only duplicate the close/open ones. The
        # manipulator — and any prompt without a recorded role (folder
        # mode, older Step-3a JSONs) — keeps the full span. Fewer than 2
        # frames left -> "insufficient keyframes" below.
        if role == "object":
            keyframes = keyframes[1:-1]
            frames = frames[1:-1]
        top_k = _prompt_top_k(self.args.object_top_k, self.args.manipulator_top_k,
                              role)
        n = len(keyframes)
        h, w = frames[0].shape[:2] if frames else (0, 0)
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

        # Optional motion-mask rescue of the manipulator (the driver's
        # --with-optical-flow-mask): the SAM ∪ optical-flow union of the
        # row-0 mask — the row the no_roma baseline samples from (and
        # RoMAv2's mask gate/crop of the 1st key-frame). The flow mask
        # comes from Step 3a' (run_step3_motion_masks.py), anchored at the
        # same 1st key-frame, so it ORs in-place — an arm the detection
        # missed still gets a mask from its own motion.
        motion_meta: dict = {}
        if role == "manipulator" and n >= 1 \
                and self.args.with_optical_flow_mask:
            fm, motion_meta = self._motion_mask_union(k, keyframes, h, w)
            if fm is not None and fm.any():
                before = int(masks[0].sum())
                sam0 = masks[0].copy()  # pre-union SAM, for the union viz
                masks[0] |= fm
                added = int(masks[0].sum()) - before
                print(f"    motion mask +{added} px onto the row-0 "
                      f"SAM mask")
                any_mask = any_mask or bool(added)
                if self.args.viz_motion_union:
                    self._visualize_union(pdir, frames[0], sam0, fm)

        keypoints = np.zeros((0, n, 2), dtype=np.float32)
        if empty_reason is None and self.args.sampling_mode == "no_roma":
            # Simple baseline (no RoMAv2): uniformly sample top-k points
            # inside the mask of the span's first frame — the sub-task's
            # 1st key-frame for the manipulator (and role-less prompts),
            # the 2nd for the object (the first of its transport span
            # after the [1:-1] trim above). No cross-frame matching; the
            # later keypoint columns carry copies of the sampled points so
            # the frame_indices span — and with it Step-4's trace window
            # and per-frame mask gating — stays identical to the RoMAv2
            # modes (Step 4 anchors on the span's leading stem or its
            # first key-frame, i.e. column 0, in the common case anyway).
            if not masks[0].any():
                empty_reason = "no mask on the sampled key-frame"
            else:
                seed = zlib.crc32(
                    f"{self.ep_idx}:{k}:{self.cam_key}:{prompt}".encode())
                pts = self._sample_in_mask(masks[0], top_k, seed)
                if len(pts):
                    keypoints = np.repeat(pts[:, None, :], n, axis=1)
                    in_mask_need = 1
        elif empty_reason is None:
            # Per-key-frame crop boxes: each key-frame is cropped around its
            # own box (the object can move between key-frames) — the frame's
            # best SAM3 box, else its largest detection box, else the first
            # box available anywhere.
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
            if not any_mask or first_box is None:
                empty_reason = "no mask or box"
            # RoMAv2 matching on the enlarged-bbox crops of all key-frames
            # (the mask/uniform modes). With --sampling-mode mask (default)
            # the per-frame mask is cropped with the same box as the image
            # and fed to RoMAv2, so the candidate pool is sampled inside
            # the object only (frames without a mask stay unconstrained).
            # With --sampling-mode uniform RoMAv2 samples over the whole
            # crops instead and the in-mask filter below keeps only the
            # tracks inside the masks — same crops, same output criterion,
            # different pool.
            if empty_reason is None:
                mask_gated = self.args.sampling_mode == "mask"
                crops, offsets, crop_masks = [], [], []
                for rgb, cb, m in zip(frames, crop_boxes, masks):
                    crop, off = self._enlarge_crop(rgb, cb)
                    if crop is None:
                        empty_reason = "degenerate crop"
                        break
                    crops.append(crop)
                    offsets.append(off)
                    if mask_gated:
                        c0x, c0y = off
                        crop_masks.append(
                            m[c0y:c0y + crop.shape[0], c0x:c0x + crop.shape[1]]
                            if m.any() else None)
                if empty_reason is None:
                    matches = self._match_object(
                        crops, crop_masks if mask_gated else None, top_k)
                    if matches is None:
                        empty_reason = "no matches"
                    else:
                        keypoints, in_mask_need = self._filter_top_k(
                            matches, masks, offsets, h, w, top_k)

        # Save (uniform schema; failures are recorded in init_points.json).
        np.savez(npz_path,
                 keypoints=keypoints,
                 frame_indices=np.asarray(keyframes, dtype=np.int64),
                 masks=masks, boxes=boxes, scores=scores)
        rle = {str(t): encode_rle(masks[j])
               for j, t in enumerate(keyframes) if masks[j].any()}
        with open(os.path.join(pdir, "masks_rle.json"), "w") as f:
            json.dump(rle, f, indent=2)
        # The matching knobs (crops, RoMAv2 strategy, checkpoint) only apply
        # to the mask/uniform modes — no_roma records them as None.
        roma = self.args.sampling_mode != "no_roma"
        meta = {
            "episode": int(self.ep_idx),
            "subtask": int(k),
            "segment": [int(min(keyframes_all)), int(max(keyframes_all)) + 1],
            "camera_key": self.cam_key,
            "prompt": prompt,
            "prompt_slug": slug,
            "role": role,
            "keyframes": [int(t) for t in keyframes],
            "num_keypoints": int(len(keypoints)),
            "top_k": top_k,
            "bbox_scale": self.args.bbox_scale if roma else None,
            "num_corresp": self.args.num_corresp if roma else None,
            "match_top_k": top_k * 4 if roma else None,
            "in_mask_min_frames": int(in_mask_need),
            "strategy": self.args.strategy if roma else None,
            "sampling_mode": self.args.sampling_mode,
            "detections_file": str(self._detections_path(self.ep_idx, k,
                                                          self.cam_key)),
            "sam3_checkpoint": DEFAULT_SAM3_CKPT,
            "romav2_checkpoint": DEFAULT_ROMAV2_CKPT if roma else None,
            "empty_reason": empty_reason,
        }
        if self.args.with_optical_flow_mask:
            meta["motion_mask_file"] = motion_meta.get("motion_mask_file")
            meta["motion_flow_pair"] = motion_meta.get("motion_flow_pair")
            meta["motion_moving_pixels"] = motion_meta.get(
                "motion_moving_pixels")
        with open(os.path.join(pdir, "init_points.json"), "w") as f:
            json.dump(meta, f, indent=2)
        if not self.args.no_viz and len(keypoints):
            self._visualize(pdir, frames, keypoints, masks, boxes, prompt,
                            col0_only=self.args.sampling_mode == "no_roma")
        status = empty_reason or f"{len(keypoints)} keypoints"
        print(f"    [{slug}] {status} -> {pdir}")

    #: union-viz colours in BGR, keyed to the legend below: SAM only,
    #: SAM ∩ motion, motion only.
    _UNION_COLORS = ((0, 0, 255), (0, 255, 255), (0, 255, 0))
    #: legend entries drawn with _UNION_COLORS, in the same order.
    _UNION_LABELS = ("SAM only", "SAM + motion", "motion only")

    def _visualize_union(self, pdir: str, frame: np.ndarray,
                         sam: np.ndarray, motion: np.ndarray) -> None:
        """union_mask.png (--viz-motion-union): the manipulator's row-0
        key-frame tinted with the SAM ∪ motion union — SAM-only red,
        SAM ∩ motion yellow, motion-only green (the pixels the flow
        rescue added, i.e. what the SAM mask alone would have missed),
        blended at the same 0.35 alpha as viz.png."""
        vis = np.ascontiguousarray(frame[:, :, ::-1])  # RGB -> BGR for cv2
        overlay = np.zeros_like(vis)
        overlay[sam & ~motion] = self._UNION_COLORS[0]
        overlay[sam & motion] = self._UNION_COLORS[1]
        overlay[motion & ~sam] = self._UNION_COLORS[2]
        vis = cv2.addWeighted(vis, 1.0, overlay, 0.35, 0)
        # darkened legend strip so the colours stay self-describing
        vis[0:66, 0:150] = (vis[0:66, 0:150] * 0.35).astype(np.uint8)
        for i, label in enumerate(self._UNION_LABELS):
            y = 22 + 20 * i
            cv2.rectangle(vis, (8, y - 9), (24, y + 3),
                          self._UNION_COLORS[i], -1)
            cv2.putText(vis, label, (32, y + 3), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(pdir, "union_mask.png"), vis)

    def _visualize(self, pdir: str, frames: list[np.ndarray],
                   keypoints: np.ndarray, masks: np.ndarray,
                   boxes: np.ndarray, prompt: str,
                   col0_only: bool = False) -> None:
        """Key-frames side-by-side with the masks, boxes and the tracks.

        col0_only: the keypoints exist on the first column only (no_roma)
        — draw them on the first frame's panel without cross-frame lines
        (the other columns are copies, not real tracks).
        """
        imgs = [f[:, :, ::-1] for f in frames]  # RGB -> BGR for cv2
        H0, W0 = imgs[0].shape[:2]
        imgs = [cv2.resize(im, (W0, H0)) for im in imgs]
        stacked = np.hstack(imgs)
        for m, pts in enumerate(keypoints):
            color = tuple(int(c) for c in _PALETTE[m % len(_PALETTE)])
            draw = pts[:1] if col0_only else pts
            for j, (x, y) in enumerate(draw):
                cx, cy = int(x) + j * W0, int(y)
                cv2.circle(stacked, (cx, cy), 3, color, -1)
                if j > 0:
                    px, py = (int(draw[j - 1][0]) + (j - 1) * W0,
                              int(draw[j - 1][1]))
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
