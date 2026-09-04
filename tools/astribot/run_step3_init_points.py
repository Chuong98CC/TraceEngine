"""Step 3 — high-level driver: extract the key-frames, then detections +
init points.

End-to-end driver of pipeline Step 3 (Sampling Keypoints), on the dataset,
for the selected episodes. It runs the three pipeline steps sequentially as
subprocesses: RexOmni (Step 3a) needs the separate .venv-rexomni env
(Python 3.10 / torch 2.7) while Step 1 and the SAM3/RoMAv2 .pt2 runtimes
(Step 3b) need the main env — launch it from the **main** environment.

The pipeline of one episode:

    episode videos (dataset)
               │
               ▼  Step 1: extract_frames.py (detect_subtask, then key_frames)
    ┌──────────────────────────────┐
    │   key-frames on disk (.jpg)  │
    └──────────────────────────────┘
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

Object prompts are per sub-task: [object, manipulator] of the sub-task's row
in the dataset's meta/subtasks.csv — there is no prompt flag. Step 1
(key_frames) labels every saved segment with its canonical ground-truth
sub-task (subtask_labels.json, matched by execution order), Step 3a reads
the labels to fetch the row of the segment's own sub-task and records the
prompts in its JSON, Step 3b re-reads them from there. For a standalone
key-frame folder (no dataset), run_e2e_init_points.py (tools/general_test/)
drives the same 3a/3b tools. Usage:

    python tools/astribot/run_step3_init_points.py
        --repo-id Kronze157/astri_making_coffee_vlva
        --data-root /data/astri_making_coffee_v1 --episode-idxes 0

Examples
--------
    # Full pipeline on episode 0: extract key-frames, 3a, 3b
    # (--use-inferred-splits prefers the detect_subtask split frames over the
    # dataset's ground-truth subtask_index column)
    python tools/astribot/run_step3_init_points.py
        --repo-id Kronze157/astri_making_coffee_vlva
        --data-root /data/astri_making_coffee_v1 --episode-idxes 0
        --use-inferred-splits

    # Reuse the key-frames on disk: re-run only 3b (tuned params).
    # --skip-extract reuses the key-frames — every selected episode must
    # carry its Step-1 subtask_labels.json (missing -> error) — and
    # --skip-3a reuses the detections JSON (missing -> error)
    python tools/astribot/run_step3_init_points.py
        --repo-id Kronze157/astri_making_coffee_vlva
        --data-root /data/astri_making_coffee_v1 --episode-idxes 0
        --skip-extract --skip-3a --top-k 64
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from utils.keyframe_utils import (
    discover_episodes,
    keyframes_root,
    select_episodes,
    subtask_labels_path,
)

DEFAULT_MAX_KEYFRAMES = 8
DEFAULT_TOP_K = 64
DEFAULT_BBOX_SCALE = 1.25
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
                        help="dataset cameras of the whole pipeline: Step 1 "
                             "extracts their key-frames, and Steps 3a/3b run "
                             "on the first of them present on disk (default: "
                             "%(default)s — the head camera)")
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
    parser.add_argument("--max-keyframes", type=int, default=DEFAULT_MAX_KEYFRAMES,
                        help="cap the key-frames per sub-task (evenly spaced); "
                             "None disables the cap (default: %(default)s)")
    parser.add_argument("--use-inferred-splits", action="store_true",
                        help="prefer the sub-task split frames inferred by "
                             "detect_subtask (subtask_splits.json) over the "
                             "dataset's ground-truth subtask_index column "
                             "(Step 1 key-frames extraction; default: ground "
                             "truth when present)")
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
                             "on disk (--keyframes-root can point elsewhere); "
                             "every selected episode must carry its Step-1 "
                             "subtask_labels.json (missing -> error)")
    parser.add_argument("--skip-done", action="store_true",
                        help="skip episodes whose Step-3a JSON exists, and "
                             "sub-tasks whose Step-3b output already exists")
    parser.add_argument("--no-viz", action="store_true",
                        help="skip the Step-3b viz.png rendering")
    parser.add_argument("--skip-3a", action="store_true",
                        help="do not run Step 3a: reuse the detections JSON "
                             "of a previous run — the JSON must exist for "
                             "every selected episode (missing -> error)")
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


def _detections_root(args) -> Path:
    """Step-3a detections root the sub-steps read/write: --detections-dir,
    else <out-dir>/detections (default: <data-root>/eps_data/detections)."""
    if args.detections_dir:
        return Path(args.detections_dir)
    base = Path(args.out_dir) if args.out_dir \
        else Path(args.data_root) / "eps_data"
    return base / "detections"


def _episodes_missing_labels(args) -> list[int]:
    """Selected episodes whose saved key-frames lack the Step-1
    subtask_labels.json. Under --skip-extract nothing re-extracts, and Step
    3a prompts every segment by its ground-truth label from that file, so
    every selected episode must carry it. [] when all do."""
    root = _keyframes_root(args)
    try:
        eps = select_episodes(root, args.episode_idxes, args.max_episodes)
    except FileNotFoundError as e:
        sys.exit(f"--skip-extract: {e} — drop --skip-extract to extract "
                 f"the key-frames of the requested episodes")
    return [e for e in eps
            if not subtask_labels_path(root, e).is_file()]


def _missing_detections(args) -> list[int]:
    """Selected episodes without their Step-3a detections JSON, one JSON per
    episode (ep{ep:06d}.json) under _detections_root. With --skip-3a the
    sub-steps can only reuse existing detections, so each episode Step 3b
    will process must have run Step 3a before. [] when every selected
    episode has its JSON, or when there are no key-frames to judge (3b then
    reports the missing key-frames itself)."""
    root = _keyframes_root(args)
    if not root.is_dir():
        return []
    eps = select_episodes(root, args.episode_idxes, args.max_episodes)
    det = _detections_root(args)
    return [e for e in eps if not (det / f"ep{e:06d}.json").is_file()]


def _dataset_camera_subdirs(args) -> list[str] | None:
    """Camera-subdir names of the selected dataset cameras: --camera-idxes
    mapped through the dataset's camera_keys to the key tails that
    extract_frames.py writes (e.g. index 0 -> cam_head). Steps 3a/3b read
    the key-frames off disk by those names, so the driver resolves the
    indices for them. The mapping needs the dataset metadata (the driver
    already requires --repo-id / --data-root); when it cannot be opened,
    None is returned and the sub-steps auto-select the first non-depth
    camera on disk (see keyframe_utils.select_camera)."""
    try:
        from lerobot.datasets import LeRobotDatasetMetadata
        meta = LeRobotDatasetMetadata(repo_id=args.repo_id,
                                      root=args.data_root)
        return [meta.camera_keys[i].rsplit(".", 1)[-1]
                for i in args.camera_idxes]
    except Exception:
        return None


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
        if args.use_inferred_splits:
            cmd += ["--use-inferred-splits"]
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
    cam_keys = _dataset_camera_subdirs(args)
    if cam_keys:
        cmd += ["--camera-keys", *cam_keys]
    if args.episode_idxes is not None:
        cmd += ["--episode-idxes", *(str(e) for e in args.episode_idxes)]
    if args.max_episodes is not None:
        cmd += ["--max-episodes", str(args.max_episodes)]
    if args.out_dir:
        cmd += ["--out-dir", args.out_dir]
    cmd += ["--max-keyframes", str(args.max_keyframes)]
    if args.skip_done:
        cmd += ["--skip-done"]
    return cmd


def _build_3b_cmd(args, repo_root: Path) -> list[str]:
    """Step 3b command: SAM3 masks + RoMAv2 keypoints in the main env."""
    cmd = [sys.executable,
           str(repo_root / _STEP_3B),
           "--data-root", args.data_root,
           "--keyframes-root", str(_keyframes_root(args))]
    cam_keys = _dataset_camera_subdirs(args)
    if cam_keys:
        cmd += ["--camera-keys", *cam_keys]
    if args.episode_idxes is not None:
        cmd += ["--episode-idxes", *(str(e) for e in args.episode_idxes)]
    if args.max_episodes is not None:
        cmd += ["--max-episodes", str(args.max_episodes)]
    if args.out_dir:
        cmd += ["--out-dir", args.out_dir]
    if args.detections_dir:
        cmd += ["--detections-dir", args.detections_dir]
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
    elif not (root := _keyframes_root(args)).is_dir():
        sys.exit(f"--skip-extract: no key-frames under {root} — drop "
                 f"--skip-extract to run Step 1 (detect_subtask + "
                 f"key_frames)")
    elif missing := _episodes_missing_labels(args):
        sys.exit(f"--skip-extract: subtask_labels.json missing for "
                 f"episode(s) {missing} under {_keyframes_root(args)} (an "
                 f"extraction predating the labels) — drop --skip-extract "
                 f"to re-extract the key-frames")
    if not args.skip_3a:
        _run(_build_3a_cmd(args, repo_root),
             "Step 3a — RexOmni detections (saved key-frames)")
    elif missing := _missing_detections(args):
        sys.exit(f"--skip-3a: Step-3a detections missing for episode(s) "
                 f"{missing} under {_detections_root(args)} — run Step 3a "
                 f"first (or drop --skip-3a to run it now)")
    _run(_build_3b_cmd(args, repo_root),
         "Step 3b — SAM3 masks + RoMAv2 init points")
    print("\nstep 3 done", flush=True)


if __name__ == "__main__":
    main()
