"""Step 3 on a key-frame folder — driver of Step 3a + 3b, one sub-task.

The folder-input variant of run_step3_init_points.py (tools/astribot/):
runs pipeline Step 3 end-to-end on a single folder of key-frame images —
the key-frames of one sub-task of one camera, e.g. a cam_head/ folder saved
by extract_frames.py --mode key_frames. The folder plays sub-task 00 of a
synthetic episode labelled --episode-idx (default 0); its name is the
default camera key. As in the episode driver, the two sub-steps cannot
share a process (RexOmni under .venv-rexomni, the SAM3/RoMAv2 .pt2 runtimes
in the main env) — launch this driver from the **main** environment:

    key-frame folder (one sub-task)
               │
               ▼  Step 3a: run_object_detection.py (RexOmni)
    ┌──────────────────────────────┐
    │  detections JSON per episode │
    └──────────────────────────────┘
               │
               ▼  Step 3b: run_object_init_points.py (SAM3 + RoMAv2)
    ┌──────────────────────────────┐
    │  init_points per prompt      │
    └──────────────────────────────┘

There is no dataset annotation: the prompts come from --text-prompts.
Outputs land under <keyframes-dir>/../step3_output (--out-dir to change):
3a writes detections/ep{episode_idx:06d}.json, 3b the per-prompt
init_points/... under the same episode label. Pass a distinct --episode-idx
per folder to keep several folders' outputs separate under one output root.

Examples
--------
    # Full Step 3 on one sub-task's key-frame folder
    python tools/general_test/pipeline/run_e2e_init_points.py
        --keyframes-dir .../subtask_00/cam_head

    # Re-run only 3b on the existing detections JSON (tuned params).
    # --skip-3a skips the RexOmni pass and reuses the detections JSON —
    # ep{episode_idx:06d}.json must already be on disk (missing -> error)
    python tools/general_test/pipeline/run_e2e_init_points.py
        --keyframes-dir .../subtask_00/cam_head --skip-3a --top-k 64
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_PROMPTS = ["brown coffee cup", "left robot arm black gripper", "right robot arm black gripper"]
DEFAULT_MAX_KEYFRAMES = 8
DEFAULT_TOP_K = 128
DEFAULT_BBOX_SCALE = 1.25
DEFAULT_NUM_CORRESP = 2000
DEFAULT_STRATEGY = "reference"
#: default path of the RexOmni environment (relative to the repo root).
REXOMNI_ENV_DIR = ".venv-rexomni"

_STEP_3A = "tools/general_test/pipeline/run_object_detection.py"
_STEP_3B = "tools/general_test/pipeline/run_object_init_points.py"


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Step 3 end-to-end on a folder of key-frame images: 3a "
                    "RexOmni detections + 3b SAM3/RoMAv2 init points, run "
                    "sequentially (the folder = one sub-task, subtask 00 of "
                    "an episode labelled --episode-idx)."
    )
    parser.add_argument("--keyframes-dir", required=True,
                        help="folder with the key-frame images of one "
                             "sub-task of one camera (the folder name is the "
                             "default camera key)")
    parser.add_argument("--out-dir", "-o", default=None,
                        help="output root (default: <keyframes-dir>/../"
                             "step3_output); Step 3a lands under "
                             "<out-dir>/detections/, Step 3b under "
                             "<out-dir>/init_points/")
    parser.add_argument("--episode-idx", type=int, default=0,
                        help="episode index labelling the outputs (default: "
                             "%(default)s)")
    parser.add_argument("--camera-key", default=None,
                        help="camera key recorded in the outputs (default: "
                             "the folder name)")
    parser.add_argument("--detections-dir", default=None,
                        help="Step-3a detections root (Step 3b only; default: "
                             "<out-dir>/detections)")
    parser.add_argument("--text-prompts", nargs="+", default=DEFAULT_PROMPTS,
                        help="object prompts (default: %(default)s)")
    parser.add_argument("--max-keyframes", type=int, default=DEFAULT_MAX_KEYFRAMES,
                        help="cap the key-frames (evenly spaced); None "
                             "disables the cap (default: %(default)s)")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                        help="final keypoints kept per prompt (Step 3b, "
                             "default: %(default)s)")
    parser.add_argument("--bbox-scale", type=float, default=DEFAULT_BBOX_SCALE,
                        help="enlargement factor of the bounding-box crops fed "
                             "to RoMAv2 (Step 3b, default: %(default)s)")
    parser.add_argument("--num-corresp", type=int, default=DEFAULT_NUM_CORRESP,
                        help="RoMAv2 candidate points sampled in the anchor "
                             "crop before filtering (Step 3b, default: "
                             "%(default)s)")
    parser.add_argument("--strategy", choices=("reference", "cycle"),
                        default=DEFAULT_STRATEGY,
                        help="RoMAv2 matching strategy (Step 3b, default: "
                             "%(default)s)")
    parser.add_argument("--device", default=None, choices=["cuda", "cpu"],
                        help="device (Step 3b; default: auto)")
    parser.add_argument("--skip-done", action="store_true",
                        help="skip when the Step-3a JSON exists, and when the "
                             "Step-3b prompt output already exists")
    parser.add_argument("--no-viz", action="store_true",
                        help="skip the Step-3b viz.png rendering")
    parser.add_argument("--skip-3a", action="store_true",
                        help="do not run Step 3a: reuse the detections JSON "
                             "of a previous run — the JSON must exist under "
                             "--detections-dir / <out-dir>/detections "
                             "(missing -> error)")
    parser.add_argument("--rexomni-env", default=REXOMNI_ENV_DIR,
                        help=f"RexOmni environment dir, relative to the repo "
                             f"root (default: {REXOMNI_ENV_DIR})")
    parser.add_argument("--rexomni-python", default=None,
                        help="explicit .venv-rexomni python executable "
                             "(overrides --rexomni-env)")
    return parser.parse_args(argv)


def _rexomni_python(args, repo_root: Path) -> str:
    """The .venv-rexomni python executable, with a helpful error when the
    environment is missing."""
    if args.rexomni_python:
        return args.rexomni_python
    py = repo_root / args.rexomni_env / "bin" / "python"
    if not py.is_file():
        sys.exit(f"{py} missing: run scripts/general_test/"
                 f"setup_rexomni_env.sh to create the RexOmni environment")
    return str(py)


def _build_3a_cmd(args, repo_root: Path) -> list[str]:
    """Step 3a command: RexOmni detections under .venv-rexomni."""
    cmd = [_rexomni_python(args, repo_root),
           str(repo_root / _STEP_3A),
           "--keyframes-dir", args.keyframes_dir]
    if args.out_dir:
        cmd += ["--out-dir", args.out_dir]
    cmd += ["--episode-idx", str(args.episode_idx)]
    if args.camera_key:
        cmd += ["--camera-key", args.camera_key]
    cmd += ["--text-prompts", *args.text_prompts,
            "--max-keyframes", str(args.max_keyframes)]
    if args.skip_done:
        cmd += ["--skip-done"]
    return cmd


def _build_3b_cmd(args, repo_root: Path) -> list[str]:
    """Step 3b command: SAM3 masks + RoMAv2 keypoints in the main env."""
    cmd = [sys.executable,
           str(repo_root / _STEP_3B),
           "--keyframes-dir", args.keyframes_dir]
    if args.out_dir:
        cmd += ["--out-dir", args.out_dir]
    if args.detections_dir:
        cmd += ["--detections-dir", args.detections_dir]
    cmd += ["--episode-idx", str(args.episode_idx)]
    if args.camera_key:
        cmd += ["--camera-key", args.camera_key]
    cmd += ["--max-keyframes", str(args.max_keyframes),
            "--top-k", str(args.top_k),
            "--bbox-scale", str(args.bbox_scale),
            "--num-corresp", str(args.num_corresp),
            "--strategy", args.strategy]
    if args.device:
        cmd += ["--device", args.device]
    if args.skip_done:
        cmd += ["--skip-done"]
    if args.no_viz:
        cmd += ["--no-viz"]
    return cmd


def _detections_path(args) -> Path:
    """The Step-3a detections JSON of the labelled episode (the path Step 3b
    reads): ep{episode_idx:06d}.json under --detections-dir, else
    <out-dir>/detections (default: <keyframes-dir>/../step3_output/
    detections)."""
    out = Path(args.out_dir) if args.out_dir \
        else Path(args.keyframes_dir).parent / "step3_output"
    det = Path(args.detections_dir) if args.detections_dir \
        else out / "detections"
    return det / f"ep{args.episode_idx:06d}.json"


def _run(cmd: list[str], step_name: str) -> None:
    """Run one sub-step, streaming its output, aborting the pipeline on a
    non-zero exit."""
    # flush the header before the child inherits the stdout fd: with a
    # block-buffered stdout (e.g. redirected to a file) the parent's prints
    # would otherwise only land after the child's whole output.
    print(f"\n===== {step_name} =====", flush=True)
    print(f"$ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(f"{step_name} failed with exit code {r.returncode}")


def main() -> None:
    args = parse_args()
    folder = Path(args.keyframes_dir)
    if not folder.is_dir():
        sys.exit(f"--keyframes-dir {folder} is not a directory")
    repo_root = Path(__file__).resolve().parents[3]
    if not args.skip_3a:
        _run(_build_3a_cmd(args, repo_root),
             "Step 3a — RexOmni detections (key-frame folder)")
    elif not (det := _detections_path(args)).is_file():
        sys.exit(f"--skip-3a: {det} missing — run Step 3a first (or drop "
                 f"--skip-3a to run it now)")
    _run(_build_3b_cmd(args, repo_root),
         "Step 3b — SAM3 masks + RoMAv2 init points")
    print("\nstep 3 done", flush=True)


if __name__ == "__main__":
    main()
