#!/usr/bin/env python3
"""
DA3 multi-camera streaming — unified entry point for all backends.

Usage:
    python tools/run_stream.py \
        --backend da3|vggt_omega \
        --input-dirs /path/cam0 /path/cam1 \
        [--mask-dirs /path/masks0 /path/masks1] \
        --start-frame 0 --max-frames 60

Backends:
- ``da3``        pre-exported DA3 any-view TorchScript (.pt) model
                 (fixed num_views).
- ``vggt_omega`` pre-exported VGGT-Omega torch.export program (.pt2/.pt)
                 (fixed num_views).

``chunk_size`` is derived from the model's fixed ``num_views`` (read from
the export itself) and must be divisible by the number of input folders.
A short final chunk is padded at its start with images from the previous
chunk (or with duplicated images when the whole sequence is shorter than
one chunk); the padded outputs are discarded.

With ``--video``, a trajectory mp4 is rendered after the run: each frame
shows that time step's coloured point cloud with its camera frustums, plus
the growing camera path from the first frame to the current one.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
from datetime import datetime

import torch

# from base_streaming import _copy_file
from depth_models.streaming.da3_streaming import DA3_Streaming
from depth_models.streaming.loop_utils.config_utils import load_config
from depth_models.streaming.vggt_omg_streaming import VGGT_OMG_Streaming

_BACKENDS = {
    "da3": DA3_Streaming,
    "vggt_omega": VGGT_OMG_Streaming,
}


def _report_run_stats(backend: str, save_dir: str, stats: dict) -> None:
    """Print a timing/memory summary and write timings.json into save_dir."""
    total_s = stats["total_s"]
    chunk_times = stats["chunk_times"]
    num_chunks = len(chunk_times)

    # GPU peak reflects inference only (peak stats reset right before run());
    # weights are loaded in the backend __init__ before that point.
    gpu_alloc_mib = None
    gpu_resv_mib = None
    if torch.cuda.is_available():
        gpu_alloc_mib = torch.cuda.max_memory_allocated() / 1024**2
        gpu_resv_mib = torch.cuda.max_memory_reserved() / 1024**2

    # ru_maxrss is a process-lifetime high-water mark (KiB on Linux), so it
    # includes model loading.
    cpu_rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    report = {
        "backend": backend,
        "total_s": round(total_s, 3),
        "num_chunks": num_chunks,
        "chunk_times_s": [round(t, 3) for t in chunk_times],
        "chunk_mean_s": round(sum(chunk_times) / num_chunks, 3) if num_chunks else None,
        "chunk_min_s": round(min(chunk_times), 3) if num_chunks else None,
        "chunk_max_s": round(max(chunk_times), 3) if num_chunks else None,
        "chunks_per_s": round(num_chunks / total_s, 3) if total_s > 0 else None,
        "peak_gpu_allocated_mib": round(gpu_alloc_mib, 1) if gpu_alloc_mib is not None else None,
        "peak_gpu_reserved_mib": round(gpu_resv_mib, 1) if gpu_resv_mib is not None else None,
        "peak_cpu_rss_mib": round(cpu_rss_mib, 1),
    }

    print("\n----------------------------------------")
    print(f"Total run time : {total_s:.3f} s ({num_chunks} chunks)")
    if num_chunks:
        print(
            f"Per-chunk infer: mean {report['chunk_mean_s']:.3f}s  "
            f"min {report['chunk_min_s']:.3f}s  max {report['chunk_max_s']:.3f}s  "
            f"({report['chunks_per_s']:.3f} chunks/s)"
        )
    if report["peak_gpu_allocated_mib"] is not None:
        print(
            f"Peak GPU mem   : {report['peak_gpu_allocated_mib']:.1f} MiB allocated / "
            f"{report['peak_gpu_reserved_mib']:.1f} MiB reserved (inference only)"
        )
    print(
        f"Peak CPU RSS   : {report['peak_cpu_rss_mib']:.1f} MiB "
        "(process lifetime, incl. model load)"
    )
    print("----------------------------------------")

    json_path = os.path.join(save_dir, "timings.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Stats saved: {json_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DA3-Streaming (multi-camera)")
    parser.add_argument(
        "--backend",
        required=True,
        choices=sorted(_BACKENDS),
        help="Inference backend: da3, vggt_omega",
    )
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        required=True,
        help="Input image folders (one per camera); all folders must contain "
        "the same synchronized frame stems",
    )
    parser.add_argument(
        "--mask-dirs",
        nargs="+",
        default=None,
        help="Optional binary motion-mask folders (one per --input-dirs, same "
        "order and image count).  Mask pixels above 127 are moving and their "
        "confidence is zeroed during chunk alignment only; outputs are "
        "unaffected.",
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Max time steps (frames per camera) to process (default: all)",
    )
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument(
        "--model-path",
        default=None,
        help="Model artifact path (.pt / .pt2); overrides the backend's default",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="torchcompile only: run the exported-program runtime instead of "
        "wrapping the loaded module in torch.compile (Inductor).  Slower, but "
        "useful for debugging numeric differences.",
    )
    parser.add_argument("--config", default="src/depth_models/streaming/configs/base_config.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--device",
        default=None,
        choices=["cuda", "cpu"],
        help="Device to run on (default: auto — cuda if available for "
        "pytorch, cuda for torchcompile/torchscript)",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Render a trajectory video after the run: the current time "
        "step's point cloud + camera frustums and the growing camera path",
    )
    parser.add_argument(
        "--video-out",
        default=None,
        help="Video output path (default: <output-dir>/trajectory.mp4)",
    )
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument(
        "--video-size",
        default="960x540",
        help="Video size WxH, e.g. 960x540 (default: 960x540)",
    )
    parser.add_argument(
        "--video-max-points",
        type=int,
        default=100_000,
        help="Max point-cloud points rendered per video frame (default: 100000)",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)

    if args.output_dir:
        save_dir = args.output_dir
    else:
        ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        save_dir = os.path.join("./exps", f"{args.backend}_stream_{ts}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Output: {save_dir}")
    # _copy_file(args.config, save_dir)

    cls = _BACKENDS[args.backend]
    kwargs = dict(
        input_dirs=args.input_dirs,
        save_dir=save_dir,
        config=config,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        interval=args.interval,
        device=args.device,
        mask_dirs=args.mask_dirs,
    )
    if args.backend in ("da3", "vggt_omega"):
        kwargs["model_path"] = args.model_path
    if args.backend == "torchcompile":
        kwargs["compile"] = not args.no_compile

    streaming = cls(**kwargs)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    stats = streaming.run()

    if args.video:
        # Lazy import: visualize_stream pulls in open3d, which is only
        # needed when a video is requested.
        from tools.general_test.visualize_stream import load_stems, render_stream_video

        stems = load_stems(save_dir, args.input_dirs)
        w, h = (int(x) for x in args.video_size.lower().split("x"))
        video_out = args.video_out or os.path.join(save_dir, "trajectory.mp4")
        print(f"\nRendering trajectory video ({len(stems)} frames) -> {video_out}")
        render_stream_video(
            stems,
            args.input_dirs,
            save_dir,
            video_out,
            fps=args.video_fps,
            size=(w, h),
            max_points_per_frame=args.video_max_points,
        )

    _report_run_stats(args.backend, save_dir, stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
