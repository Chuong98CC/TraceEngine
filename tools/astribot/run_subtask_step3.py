"""Step 3 — high-level driver: extract the key-frames, then detections +
init points.

Runs pipeline Step 3 (Sampling Keypoints) end-to-end for the selected
episodes, on the dataset: first Step 1 — extract_frames.py saves the
sub-task key-frames to disk (--mode detect_subtask, then --mode key_frames;
Step 3 only infers on a sparse set of frames, most commonly ~4 per sub-task,
so they are persisted instead of decoding the episode videos online) — then
Step 3a (run_object_detection.py — RexOmni detections, under
.venv-rexomni) and Step 3b (run_object_init_points.py — SAM3 masks +
RoMAv2 keypoints, main env).

The dataset-independent tools live in tools/general_test/ and run on the
saved key-frames alone (folder input — run_e2e_init_points.py is their
standalone driver); this script is the high-level dataset layer that adds
the extraction and drives the sub-steps, importing the shared key-frame
helpers from general_test. The sub-steps cannot share a process (RexOmni
needs Python 3.10 / torch 2.7 while the SAM3/RoMAv2 .pt2 runtimes need the
main env), so they are launched sequentially as subprocesses. Run it from
the **main** environment:

    python tools/astribot/run_subtask_step3.py \
        --repo-id Kronze157/astri_making_coffee_vlva \
        --data-root /data/astri_making_coffee --episode-idxes 0

Examples
--------
    # Full pipeline on episode 0: extract key-frames, 3a, 3b
    python tools/astribot/run_subtask_step3.py \
        --repo-id Kronze157/astri_making_coffee_vlva \
        --data-root /data/astri_making_coffee --episode-idxes 0

    # Reuse the key-frames on disk: re-run only 3b (tuned params)
    python tools/astribot/run_subtask_step3.py \
        --repo-id Kronze157/astri_making_coffee_vlva \
        --data-root /data/astri_making_coffee --episode-idxes 0 \
        --skip-extract --skip-3a --top-k 64

    # Skip RexOmni entirely: extract, then 3b with SAM3 text-only prompts
    python tools/astribot/run_subtask_step3.py \
        --repo-id Kronze157/astri_making_coffee_vlva \
        --data-root /data/astri_making_coffee --episode-idxes 0 \
        --no-rexomni
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from utils.keyframe_utils import (
    discover_episodes,
    keyframes_root,
)

DEFAULT_PROMPTS = ["brown coffee cup", "robot gripper"]
DEFAULT_MAX_KEYFRAMES = 8
DEFAULT_TOP_K = 128
DEFAULT_BBOX_SCALE = 1.5
DEFAULT_NUM_CORRESP = 2000
DEFAULT_STRATEGY = "reference"
#: default path of the RexOmni environment (relative to the repo root).
REXOMNI_ENV_DIR = ".venv-rexomni"

_STEP_1 = "tools/astribot/extract_frames.py"
_STEP_3A = "tools/general_test/pipeline/run_object_detection.py"
_STEP_3B = "tools/general_test/pipeline/run_object_init_points.py"


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Step 3 end-to-end on the dataset: extract the sub-task "
                    "key-frames (Step 1), then 3a RexOmni detections + 3b "
                    "SAM3/RoMAv2 init points, run sequentially."
    )
    parser.add_argument("--repo-id", "-id", required=True,
                        help="dataset repo id (Step 1 extraction needs it; "
                             "also recorded in the Step-3a detections JSON)")
    parser.add_argument("--data-root", "-d", required=True,
                        help="root of the local dataset copy; the default "
                             "key-frames root and output root derive from it")
    parser.add_argument("--keyframes-root", default=None,
                        help="key-frames root to read when --skip-extract "
                             "(default: <data-root>/eps_data/key_frames; "
                             "extraction always writes under the output "
                             "root)")
    parser.add_argument("--camera-idxes", "-c", nargs="+", type=int,
                        default=[0],
                        help="dataset camera indices whose key-frames are "
                             "extracted (Step 1; default: %(default)s — the "
                             "head camera); for non-default cameras also "
                             "pass the matching --camera-keys subdir")
    parser.add_argument("--camera-keys", nargs="+", default=None,
                        help="camera subdir names (e.g. cam_head); the first "
                             "one present on disk wins (default: the first "
                             "RGB camera saved on disk)")
    parser.add_argument("--episode-idxes", "-e", nargs="*", type=int, default=None,
                        help="only process these episode indices (default: all "
                             "episodes with key-frames on disk)")
    parser.add_argument("--max-episodes", "-x", type=int, default=None,
                        help="cap the number of processed episodes")
    parser.add_argument("--out-dir", "-o", default=None,
                        help="output root (default: <data-root>/eps_data); "
                             "Step 1 lands under <out-dir>/key_frames/, "
                             "Step 3a under <out-dir>/detections/, Step 3b "
                             "under <out-dir>/init_points/")
    parser.add_argument("--detections-dir", default=None,
                        help="Step-3a detections root (Step 3b only; default: "
                             "<out-dir>/detections)")
    parser.add_argument("--text-prompts", nargs="+", default=DEFAULT_PROMPTS,
                        help="object prompts (default: %(default)s)")
    parser.add_argument("--max-keyframes", type=int, default=DEFAULT_MAX_KEYFRAMES,
                        help="cap the key-frames per sub-task (evenly spaced); "
                             "None disables the cap (default: %(default)s)")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                        help="final keypoints kept per prompt per sub-task "
                             "(Step 3b, default: %(default)s)")
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
    parser.add_argument("--skip-extract", action="store_true",
                        help="do not run Step 1: reuse the key-frames already "
                             "on disk (--keyframes-root can point elsewhere)")
    parser.add_argument("--skip-done", action="store_true",
                        help="skip episodes whose Step-3a JSON exists, and "
                             "sub-tasks whose Step-3b output already exists")
    parser.add_argument("--no-viz", action="store_true",
                        help="skip the Step-3b viz.png rendering")
    run = parser.add_mutually_exclusive_group()
    run.add_argument("--skip-3a", action="store_true",
                     help="do not run Step 3a: reuse the existing detections "
                          "JSON (Step 3b still reads it)")
    run.add_argument("--no-rexomni", action="store_true",
                     help="skip Step 3a and run Step 3b with SAM3 text-only "
                          "prompts (no detections JSON)")
    parser.add_argument("--rexomni-env", default=REXOMNI_ENV_DIR,
                        help=f"RexOmni environment dir, relative to the repo "
                             f"root (default: {REXOMNI_ENV_DIR})")
    parser.add_argument("--rexomni-python", default=None,
                        help="explicit .venv-rexomni python executable "
                             "(overrides --rexomni-env)")
    return parser.parse_args(argv)


def _keyframes_root(args) -> Path:
    """The key-frames root the sub-steps read: the extraction writes under
    the output root, so both must agree (an external root is only possible
    with --skip-extract)."""
    if args.keyframes_root:
        return Path(args.keyframes_root)
    return Path(args.out_dir) / "key_frames" if args.out_dir \
        else keyframes_root(args.data_root)


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


def _build_extract_cmd(args, repo_root: Path, mode: str) -> list[str]:
    """Step 1 command: extract_frames.py (detect_subtask is idempotent and
    must run before key_frames, which reads its subtask splits)."""
    cmd = [sys.executable,
           str(repo_root / _STEP_1),
           "--repo-id", args.repo_id,
           "--data-root", args.data_root,
           "--mode", mode]
    if args.episode_idxes is not None:
        cmd += ["--episode-idxes", *(str(e) for e in args.episode_idxes)]
    if args.max_episodes is not None:
        cmd += ["--max-episodes", str(args.max_episodes)]
    if mode == "key_frames":
        cmd += ["--camera-idxes", *(str(c) for c in args.camera_idxes)]
    if args.out_dir:
        cmd += ["--out-dir", args.out_dir]
    return cmd


def _build_3a_cmd(args, repo_root: Path) -> list[str]:
    """Step 3a command: RexOmni detections under .venv-rexomni."""
    cmd = [_rexomni_python(args, repo_root),
           str(repo_root / _STEP_3A),
           "--data-root", args.data_root,
           "--keyframes-root", str(_keyframes_root(args))]
    if args.repo_id:
        cmd += ["--repo-id", args.repo_id]
    if args.camera_keys:
        cmd += ["--camera-keys", *args.camera_keys]
    if args.episode_idxes is not None:
        cmd += ["--episode-idxes", *(str(e) for e in args.episode_idxes)]
    if args.max_episodes is not None:
        cmd += ["--max-episodes", str(args.max_episodes)]
    if args.out_dir:
        cmd += ["--out-dir", args.out_dir]
    cmd += ["--text-prompts", *args.text_prompts,
            "--max-keyframes", str(args.max_keyframes)]
    if args.skip_done:
        cmd += ["--skip-done"]
    return cmd


def _build_3b_cmd(args, repo_root: Path) -> list[str]:
    """Step 3b command: SAM3 masks + RoMAv2 keypoints in the main env."""
    cmd = [sys.executable,
           str(repo_root / _STEP_3B),
           "--data-root", args.data_root,
           "--keyframes-root", str(_keyframes_root(args))]
    if args.camera_keys:
        cmd += ["--camera-keys", *args.camera_keys]
    if args.episode_idxes is not None:
        cmd += ["--episode-idxes", *(str(e) for e in args.episode_idxes)]
    if args.max_episodes is not None:
        cmd += ["--max-episodes", str(args.max_episodes)]
    if args.out_dir:
        cmd += ["--out-dir", args.out_dir]
    if args.detections_dir:
        cmd += ["--detections-dir", args.detections_dir]
    cmd += ["--text-prompts", *args.text_prompts,
            "--max-keyframes", str(args.max_keyframes),
            "--top-k", str(args.top_k),
            "--bbox-scale", str(args.bbox_scale),
            "--num-corresp", str(args.num_corresp),
            "--strategy", args.strategy]
    if args.no_rexomni:
        cmd += ["--no-rexomni"]
    if args.device:
        cmd += ["--device", args.device]
    if args.skip_done:
        cmd += ["--skip-done"]
    if args.no_viz:
        cmd += ["--no-viz"]
    return cmd


def _run(cmd: list[str], step_name: str) -> None:
    """Run one step, streaming its output, aborting the pipeline on a
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
    repo_root = Path(__file__).resolve().parents[2]
    if not args.skip_extract:
        if args.keyframes_root:
            sys.exit("--keyframes-root conflicts with key-frame extraction "
                     "(Step 1 writes under --out-dir / <data-root>/eps_data): "
                     "pass --skip-extract to use externally managed "
                     "key-frames")
        _run(_build_extract_cmd(args, repo_root, "detect_subtask"),
             "Step 1a — sub-task split detection (no videos)")
        _run(_build_extract_cmd(args, repo_root, "key_frames"),
             "Step 1b — key-frames saved to disk")
        root = _keyframes_root(args)
        print(f"key-frames on disk: {len(discover_episodes(root))} episode(s) "
              f"under {root}", flush=True)
    if not args.skip_3a and not args.no_rexomni:
        _run(_build_3a_cmd(args, repo_root),
             "Step 3a — RexOmni detections (saved key-frames)")
    _run(_build_3b_cmd(args, repo_root),
         "Step 3b — SAM3 masks + RoMAv2 init points")
    print("\nstep 3 done", flush=True)


if __name__ == "__main__":
    main()
