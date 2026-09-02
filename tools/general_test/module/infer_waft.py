#!/usr/bin/env python
"""Run WAFT optical flow inference on a video file or a folder of extracted
frames (WAFTv2 torch.export ``.pt2`` artifact).

A folder input is auto-detected when ``--input`` is a directory: frames are
scanned as ``frame_{idx}.jpg`` / ``frame_{idx}.png``, sorted by index, and
paired exactly like video frames.  Per-frame output files are then named
after the anchor frame's source index instead of the running pair counter.

The model is the WAFTv2 bf16 ``torch.export`` artifact wrapped by
:class:`WAFTv2_PT2`.  A ``.pt2`` path is used as-is; an extension-less base
name gets ``.pt2`` appended (legacy ``.onnx`` / ``.engine`` checkpoints are
rejected).

Usage:
    # Video file (default checkpoint)
    python tools/infer_waft.py --input video.mp4 --start 0

    # Folder of extracted frames
    python tools/infer_waft.py --input frames/ --stride 4

    # Explicit .pt2 artifact
    python tools/infer_waft.py --input video.mp4 \\
        --checkpoint weights/waftv2/waftv2_dinov3_i5_640x480_bf16.pt2
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from flow_models.waftv2.waftv2_pt2 import WAFTv2_PT2
from utils.streaming_utils import scan_image_folder
from utils.visualize_flow import writeFlow, flow_to_image
from utils.video_io import get_video_info, VideoWriter

# ---------------------------------------------------------------------------
# Default checkpoint (.pt2 suffix appended at load time)
# ---------------------------------------------------------------------------

_DEFAULT_CKPT = "weights/waftv2/waftv2_dinov3_i5_640x480"


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def setup_output_dirs(
    output_root: str, video_name: str, output_mode: str,
    motion_threshold: float | None = None,
) -> dict:
    """Create output subdirectories and return their paths."""
    base = os.path.join(output_root, video_name)
    dirs: dict[str, str] = {}
    if output_mode in ("flow", "all"):
        dirs["flow"] = os.path.join(base, "flow")
    if output_mode in ("raw", "all"):
        dirs["raw"] = os.path.join(base, "raw")
    if output_mode in ("overlay", "all"):
        dirs["overlay"] = os.path.join(base, "overlay")
    if motion_threshold is not None and output_mode in ("mask", "flow-mask", "all"):
        dirs["mask"] = os.path.join(base, "mask")
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def write_frame_outputs(
    frame_idx: int,
    flow: np.ndarray,
    output_mode: str,
    output_dirs: dict,
) -> np.ndarray:
    """Write per-frame ``.flo`` file if requested, return flow visualisation.

    Only ``raw`` mode writes per-frame ``.flo`` files.  ``frame_idx`` is the
    anchor frame's source index in folder mode, or the running pair counter
    in video mode.
    """
    flow_vis = flow_to_image(flow, convert_to_bgr=True)

    if output_mode in ("raw", "all") and "raw" in output_dirs:
        name = f"frame_{frame_idx:06d}"
        writeFlow(os.path.join(output_dirs["raw"], f"{name}.flo"), flow)

    return flow_vis


# ---------------------------------------------------------------------------
# Motion mask
# ---------------------------------------------------------------------------

def _compute_motion_mask_gray(flow: np.ndarray, threshold: float) -> np.ndarray:
    """Return a ``[H, W]`` uint8 grayscale mask where ``||flow|| > threshold``
    is 255 (white), else 0."""
    return np.where(np.linalg.norm(flow, axis=-1) > threshold, np.uint8(255), np.uint8(0))


def _save_mask_jpg(
    mask_gray: np.ndarray, frame_idx: int, output_dirs: dict,
) -> None:
    """Save a grayscale mask as a JPEG."""
    if "mask" not in output_dirs:
        return
    name = f"mask_{frame_idx:06d}.jpg"
    cv2.imwrite(os.path.join(output_dirs["mask"], name), mask_gray)


# ---------------------------------------------------------------------------
# Checkpoint resolution
# ---------------------------------------------------------------------------

def _resolve_checkpoint(checkpoint: str) -> str:
    """Return the resolved ``.pt2`` artifact path.

    A ``.pt2`` path is used as-is.  An extension-less base name (the
    default) gets ``.pt2`` appended.  Legacy ``.onnx`` / ``.engine``
    checkpoints are rejected — the WAFT runtimes run the WAFTv2
    ``torch.export`` artifact only.  Shared by ``infer_waft.py`` and
    ``tools/astribot/run_step2_depth_stream.py``.
    """
    path = Path(checkpoint)
    ext = path.suffix.lower()

    if ext == ".pt2":
        return checkpoint
    if ext in (".onnx", ".engine"):
        print(
            f"ERROR: '{checkpoint}' is a legacy ONNX / TensorRT checkpoint. "
            f"The WAFT tools now run the WAFTv2 torch.export artifact — pass "
            f"a .pt2 path (default: {_DEFAULT_CKPT}.pt2)."
        )
        sys.exit(1)

    return checkpoint + ".pt2"


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def create_model(args: argparse.Namespace):
    """Create the WAFTv2_PT2 model (bf16 torch.export .pt2 artifact)."""
    ckpt_path = _resolve_checkpoint(args.checkpoint)
    print(f"Loading PT2 artifact: {ckpt_path}")
    return WAFTv2_PT2(ckpt_path, device=args.device)


# ---------------------------------------------------------------------------
# Frame sources (video file or folder of extracted frames)
# ---------------------------------------------------------------------------

class VideoFrameSource:
    """Frame source backed by a video file (``cv2.VideoCapture``)."""

    def __init__(self, path: str) -> None:
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            print(f"ERROR: Cannot open video file: {path}")
            sys.exit(1)

    def read(self) -> tuple[bool, np.ndarray | None, int | None]:
        """Return ``(ok, frame, None)`` — video mode has no source index."""
        ret, frame = self._cap.read()
        return ret, frame, None

    def close(self) -> None:
        self._cap.release()


class FolderFrameSource:
    """Frame source backed by a scanned ``(frame_index, path)`` file list."""

    def __init__(self, file_list: list[tuple[int, str]]) -> None:
        self._files = file_list
        self._pos = 0

    def read(self) -> tuple[bool, np.ndarray | None, int | None]:
        """Return ``(ok, frame, frame_index)`` for the next image on disk."""
        if self._pos >= len(self._files):
            return False, None, None
        frame_idx, path = self._files[self._pos]
        self._pos += 1
        frame = cv2.imread(path)
        if frame is None:
            print(f"ERROR: Cannot read frame image: {path}")
            sys.exit(1)
        return True, frame, frame_idx

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def infer_frames(args: argparse.Namespace) -> None:
    """Main inference loop over video or folder frame pairs."""

    # ── Probe input & open frame source ──────────────────────────────────
    is_folder = os.path.isdir(args.input)

    if is_folder:
        try:
            # --start is a frame *index* in folder mode (filtered by the
            # scan), unlike video mode where it skips the first N frames.
            file_list, frame_h, frame_w = scan_image_folder(
                args.input, start_frame=args.start
            )
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"ERROR: {exc}")
            sys.exit(1)
        video_name = os.path.basename(os.path.normpath(args.input))
        total_frames = len(file_list)
        fps, width, height = 0.0, frame_w, frame_h
        print(f"Folder: {args.input}")
        print(f"  Resolution: {width}x{height} (from first frame)")
        print(f"  Total frames: {total_frames}")
        source = FolderFrameSource(file_list)
    else:
        video_name = os.path.splitext(os.path.basename(args.input))[0]
        total_frames, fps, width, height = get_video_info(args.input)
        print(f"Video: {args.input}")
        print(f"  Resolution: {width}x{height}, FPS: {fps:.2f}")
        if total_frames > 0:
            print(f"  Total frames: {total_frames}")
        else:
            print(f"  Total frames: unknown (codec limitation)")
        source = VideoFrameSource(args.input)

    # ── Load model ───────────────────────────────────────────────────────
    model = create_model(args)
    print(f"  Target resolution: {model.target_h}x{model.target_w}")

    # Warm-up: video mode discards the first 'start' frames; folder mode
    # already filtered by frame index in scan_image_folder.
    if not is_folder:
        for skip_idx in range(args.start):
            ret, _, _ = source.read()
            if not ret:
                print(
                    f"ERROR: Video ended before skipping {args.start} frames "
                    f"(only {skip_idx} readable)."
                )
                source.close()
                sys.exit(1)

    # Read first anchor frame
    ret, frame_a, frame_a_idx = source.read()
    if not ret:
        if is_folder:
            print(
                f"ERROR: No frame images with index >= {args.start} "
                f"in {args.input}."
            )
        else:
            print(
                f"ERROR: Cannot read frame {args.start} from the video. "
                "The codec may be unsupported by your OpenCV/ffmpeg build. "
                "Try re-encoding to H.264:\n"
                "  ffmpeg -i input.mp4 -c:v libx264 -preset fast -crf 23 output.mp4"
            )
        source.close()
        sys.exit(1)

    # Determine iteration limit.  Folder mode already filtered by --start
    # (frame index) during the scan; video mode still counts all frames.
    avail_frames = total_frames if is_folder else total_frames - args.start
    max_pairs = (
        args.max_frames
        if args.max_frames is not None
        else (
            max(0, avail_frames) // args.stride
            if total_frames > 0
            else None
        )
    )

    # Set up progress bar
    output_dirs = setup_output_dirs(
        args.output_dir, video_name, args.output_mode, args.motion_threshold
    )
    pbar = tqdm(total=max_pairs, desc="Processing", unit="pair")

    # Set up streaming video writers (lazy — opened on first frame).
    output_fps = fps / args.stride if fps > 0 else 30.0
    flow_writer: VideoWriter | None = None
    overlay_writer: VideoWriter | None = None
    mask_writer: VideoWriter | None = None
    if output_dirs.get("flow"):
        flow_writer = VideoWriter(
            os.path.join(output_dirs["flow"], "flow.mp4"), output_fps
        )
    if output_dirs.get("overlay"):
        overlay_writer = VideoWriter(
            os.path.join(output_dirs["overlay"], "overlay.mp4"), output_fps
        )
    if args.motion_threshold is not None and args.output_mode in ("flow-mask", "all"):
        mask_writer = VideoWriter(
            os.path.join(output_dirs["mask"], "mask.mp4"), output_fps
        )

    pair_idx = 0
    start_time = time.time()

    while max_pairs is None or pair_idx < max_pairs:
        # Skip gap frames (stride - 1)
        for _ in range(args.stride - 1):
            source.read()

        ret, frame_b, frame_b_idx = source.read()
        if not ret:
            break  # EOF

        # Run inference — model.__call__ handles pre/post processing.  The
        # model takes RGB (repo-wide pixel-space contract); cv2 frames are
        # BGR, so flip at this boundary (frames stay BGR for the video /
        # overlay encoders below).
        flow = model(
            cv2.cvtColor(frame_a, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(frame_b, cv2.COLOR_BGR2RGB),
        )  # → [H_orig, W_orig, 2] float32

        # Legacy WAFTBase.__call__ sanitised NaN / Inf before returning;
        # keep the same parity for downstream masks / .flo / viz outputs.
        flow = np.nan_to_num(flow, nan=0.0, posinf=0.0, neginf=0.0)

        # Per-frame outputs are named by the anchor frame's source index in
        # folder mode, the running pair counter in video mode.
        out_idx = frame_a_idx if frame_a_idx is not None else pair_idx

        # Write per-frame outputs and stream to video encoders
        flow_vis = write_frame_outputs(
            out_idx, flow, args.output_mode, output_dirs
        )

        if flow_writer is not None:
            flow_writer.write_frame(flow_vis)
        if overlay_writer is not None:
            overlay_writer.write_overlay_frame(frame_a, flow_vis)
        if args.motion_threshold is not None and (
            args.output_mode in ("mask", "all") or mask_writer is not None
        ):
            mask_gray = _compute_motion_mask_gray(flow, args.motion_threshold)
            if args.output_mode in ("mask", "all"):
                _save_mask_jpg(mask_gray, out_idx, output_dirs)
            if mask_writer is not None:
                mask_writer.write_frame(cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2BGR))

        frame_a, frame_a_idx = frame_b, frame_b_idx  # slide forward
        pair_idx += 1
        pbar.update(1)

    pbar.close()
    source.close()

    if pair_idx == 0:
        print("No frame pairs were processed. Check --start and --stride settings.")
        sys.exit(0)

    elapsed = time.time() - start_time
    print(
        f"\nProcessed {pair_idx} pairs in {elapsed:.1f}s "
        f"({pair_idx / elapsed:.1f} pairs/s)"
    )

    # ── Finalise videos ───────────────────────────────────────────────────
    if flow_writer is not None:
        flow_writer.close()
    if overlay_writer is not None:
        overlay_writer.close()
    if mask_writer is not None:
        mask_writer.close()

    print(f"\nOutput saved to: {os.path.join(args.output_dir, video_name)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run WAFT optical flow inference on a video file or a "
                    "folder of extracted frames (WAFTv2 torch.export .pt2).",
    )
    parser.add_argument(
        "--input", required=True, type=str,
        help="Path to input video file or folder of frame images "
             "(frame_{idx}.jpg / .png).",
    )

    # Checkpoint
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=_DEFAULT_CKPT,
        help="Path to the WAFTv2 .pt2 artifact.  A .pt2 path is used as-is; "
             "an extension-less base name gets .pt2 appended.  Legacy "
             f".onnx / .engine checkpoints are rejected.  Default: "
             f"{_DEFAULT_CKPT}.pt2",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        default="./output",
        type=str,
        help="Root output directory (default: ./output).",
    )
    parser.add_argument(
        "--output-mode", "-o",
        default="flow",
        type=str,
        choices=["flow", "raw", "overlay", "mask", "flow-mask", "all"],
        help="What to produce: flow color video, raw .flo data, "
             "overlay video, motion-mask frames (.npz), "
             "motion-mask video, or all of the above.  "
             "``mask`` and ``flow-mask`` require --motion-threshold "
             "(default: flow).",
    )

    # Frame control
    parser.add_argument(
        "--start",
        default=0,
        type=int,
        help="First frame to process: the frame *index* in folder mode "
             "(e.g. frame_0210.jpg for 210), or the number of frames to "
             "skip in video mode (0-based, default: 0).",
    )
    parser.add_argument(
        "--stride",
        default=1,
        type=int,
        help="Gap between paired frames, >=1 (default: 1).",
    )
    parser.add_argument(
        "--max-frames",
        default=None,
        type=int,
        help="Max number of frame PAIRS to process (default: until EOF).",
    )

    # Motion mask
    parser.add_argument(
        "--motion-threshold", "-thr",
        default=None,
        type=float,
        help="Pixel-displacement threshold for a binary moving-pixel mask.  "
             "Use ``--output-mode mask`` for per-frame boolean ``.npz``, "
             "or ``--output-mode flow-mask`` for a mask video.",
    )

    # Misc
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        choices=["cuda", "cpu"],
        help="Device to run the .pt2 artifact on (default: cuda).",
    )

    args = parser.parse_args()
    infer_frames(args)


if __name__ == "__main__":
    main()
