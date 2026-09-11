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
    │  detections JSON per ep+cam  │
    └──────────────────────────────┘
               │
               ▼  Step 3a' (only with --with-optical-flow-mask):
               │     run_step3_motion_masks.py (WAFT, online frames)
    ┌──────────────────────────────┐
    │  motion_mask per sub-task    │
    └──────────────────────────────┘
               │
               ▼  Step 3b: run_object_init_points.py (SAM3 + RoMAv2)
    ┌──────────────────────────────┐
    │  init_points per cam/prompt  │
    └──────────────────────────────┘

Steps 3a/3b run **per camera**: every --camera-idxes entry whose key-frames
are on disk gets its own detection pass and its own init-points subtree,
saved camera-keyed like Step 2's depth_pose — one detections JSON per
sub-task and camera, and one init_points/<camera>/<prompt_slug>/ subtree
per sub-task and camera (the camera subdir name, e.g. cam_head):

    <out-dir>/ep{ep:03d}/subtask_{k:02d}/sampling_points/
        key_frames/<camera>/      (Step 1b)
        detections/<camera>.json  (Step 3a)
        init_points/<camera>/<prompt_slug>/  (Step 3b)

so multi-camera runs never overwrite each other. The driver's output root
defaults to the episodes root <data-root>/episodes (--out-dir to change);
Step 1a (detect_subtask) writes the splits + gripper plot of an episode
into that same root (ep{ep:03d}/subtask.json + split_graph.png), and Step
2's depth_pose/ and Step 4's traces/ live in it too.

Object prompts are per sub-task: [object, manipulator] of the sub-task's row
in the dataset's meta/subtasks.csv — there is no prompt flag. Step 1
(detect_subtask) resolves the canonical ground-truth label of every segment
into the episode's subtask.json ``subtask_labels`` list by execution order,
Step 3a indexes that list by segment ordinal to fetch the row of the
segment's own sub-task and records the prompts in its JSON, Step 3b
re-reads them from there.

With --with-optical-flow-mask the driver additionally runs **Step 3a'**
(run_step3_motion_masks.py — WAFTv2 motion masks of the sub-task starts,
computed online over the dataset frames between Step 3a and Step 3b) and
forwards the flag to Step 3b, which unions each sub-task's motion mask
into the **manipulator** prompt's row-0 SAM3 mask (SAM ∪ optical flow,
see run_step3_motion_masks.py) — the rescue for an arm the RexOmni
detection missed. Step 3a' saves a mask only when a flow pair in the
window is significant, as motion_rle.json (COCO RLE) in the
(sub-task, camera) init_points folder of the episodes tree — next to the
prompt subfolders of the init points it shaped; Step 3b renders the
resulting SAM ∪ motion union as union_mask.png inside the manipulator
prompt's folder. Step 3a' writes nothing else (no flow.png).

--visualize renders the Step-3b visualizations: the per-prompt viz.png
(key-frames with the masks, the boxes and the keypoints) and, with
--with-optical-flow-mask, the manipulator's union_mask.png — the
motion-only pixels being exactly what the flow rescue added. Nothing is
rendered without the flag.

Whichever mask the manipulator ends up with (SAM ∪ optical flow, or the
plain SAM mask), its no_roma draw is weighted toward the manipulated
object: Step 3b weights every row-0 mask pixel by its distance to the
object prompt's Step-3a box center — 1/(1+(d/R)²), R the mask's median
distance, max-normalized — drops the below-median half (the pixels beyond
R) and samples the top-k proportionally, so the arm keypoints stay on the
side of the mask the object is on, denser toward it
(--no-manipulator-near-object restores the uniform draw).

For a standalone
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

    # Same, with the manipulator mask rescued by optical flow (Step 3a' +
    # Step 3b --with-optical-flow-mask, off by default)
    python tools/astribot/run_step3_init_points.py
        --repo-id Kronze157/astri_making_coffee_vlva
        --data-root /data/astri_making_coffee_v1 --episode-idxes 0
        --with-optical-flow-mask

    # Re-run only 3b (tuned params), reusing what is on disk: --skip-3a
    # implies --skip-extract (the reused detections were made from the
    # key-frames on disk, so extraction is skipped too). The reused
    # episodes must carry their Step-1 subtask.json and the detections
    # JSONs must exist (missing -> error)
    python tools/astribot/run_step3_init_points.py
        --repo-id Kronze157/astri_making_coffee_vlva
        --data-root /data/astri_making_coffee_v1 --episode-idxes 0
        --skip-3a --object-top-k 64
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from utils import astribot_paths as ap
from utils.keyframe_utils import (
    camera_subdirs,
    discover_episodes,
    select_episodes,
)

DEFAULT_MAX_KEYFRAMES = 8
#: final keypoints kept per object prompt per sub-task.
DEFAULT_OBJECT_TOP_K = 64
#: the manipulator is sampled denser than the object prompts: the Step-4
#: static filters remove the stationary keypoints, so the moving
#: manipulator needs more seeds to keep enough survivors (its pass tracks
#: up to 128 role keypoints, the shipped iteration graph trimming the
#: support points to 1088 - 128 = 960 to keep its fixed 1088 queries).
DEFAULT_MANIPULATOR_TOP_K = 128
DEFAULT_BBOX_SCALE = 1.25
DEFAULT_NUM_CORRESP = 2000
DEFAULT_STRATEGY = "reference"
DEFAULT_SAMPLING_MODE = "no_roma"
#: default path of the RexOmni environment (relative to the repo root).
REXOMNI_ENV_DIR = ".venv-rexomni"
#: defaults of the Step-3a' WAFT motion-mask knobs (see the pass's --help).
DEFAULT_MOTION_THRESHOLD = 2.0
DEFAULT_MOTION_RATIO = 0.03

_STEP_1 = "tools/astribot/extract_frames.py"
_STEP_3A = "tools/general_test/pipeline/run_object_detection.py"
_STEP_3AP = "tools/astribot/run_step3_motion_masks.py"
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
                             "episodes root derives from it")
    parser.add_argument("--camera-idxes", "-c", nargs="+", type=int,
                        default=[0],
                        help="dataset cameras of the whole pipeline: Step 1 "
                             "extracts their key-frames, and Steps 3a/3b run "
                             "on each of them whose key-frames are on disk — "
                             "per-camera outputs under the sub-task's own "
                             "sampling_points/ folder "
                             "(…/subtask_{k:02d}/sampling_points/"
                             "detections/<camera>.json and "
                             "init_points/<camera>/; default: %(default)s — "
                             "the head camera)")
    parser.add_argument("--episode-idxes", "-e", nargs="*", type=int, default=None,
                        help="only process these episode indices (default: all "
                             "episodes with key-frames on disk)")
    parser.add_argument("--max-episodes", "-x", type=int, default=None,
                        help="cap the number of processed episodes")
    parser.add_argument("--out-dir", "-o", default=None,
                        help="episodes root (default: <data-root>/episodes); "
                             "Step 1's key-frames land under "
                             "<out-dir>/ep{ep:03d}/subtask_{k:02d}/"
                             "sampling_points/key_frames/<camera>/, Step 3a "
                             "the detections beside them, Step 3b the init "
                             "points under the same sampling_points/")
    parser.add_argument("--max-keyframes", type=int, default=DEFAULT_MAX_KEYFRAMES,
                        help="cap the key-frames per sub-task (evenly spaced); "
                             "None disables the cap (default: %(default)s)")
    parser.add_argument("--use-inferred-splits", action="store_true",
                        help="prefer the sub-task split frames inferred by "
                             "detect_subtask (subtask.json) over the "
                             "dataset's ground-truth subtask_index column "
                             "(Step 1 key-frames extraction; default: ground "
                             "truth when present)")
    parser.add_argument("--object-top-k", type=int,
                        default=DEFAULT_OBJECT_TOP_K,
                        help="final keypoints kept per object prompt per "
                             "sub-task (Step 3b, default: %(default)s; the "
                             "manipulator prompt uses --manipulator-top-k)")
    parser.add_argument("--manipulator-top-k", type=int,
                        default=DEFAULT_MANIPULATOR_TOP_K,
                        help="final keypoints kept per manipulator prompt "
                             "per sub-task (Step 3b — the manipulator is "
                             "sampled denser so enough of its points "
                             "survive the Step-4 static filters; default: "
                             "%(default)s)")
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
    parser.add_argument("--sampling-mode",
                        choices=("mask", "uniform", "no_roma"),
                        default=DEFAULT_SAMPLING_MODE,
                        help="where RoMAv2 samples its candidate points "
                             "(Step 3b, default: %(default)s): 'mask' "
                             "constrains the pool inside the object masks "
                             "by passing them to RoMAv2; 'uniform' samples "
                             "over the whole enlarged crops instead and "
                             "the in-mask top-k filter alone decides — "
                             "same crops and filter, run both modes into "
                             "separate --out-dirs to compare; 'no_roma' "
                             "skips RoMAv2 — the simple baseline uniformly "
                             "sampling top-k points inside the mask of the "
                             "span's first frame only (manipulator: the "
                             "1st key-frame; object: the 2nd, first of its "
                             "transport span)")
    parser.add_argument("--no-manipulator-near-object", action="store_true",
                        help="do not bias the manipulator's no_roma draw "
                             "toward the manipulated object (Step 3b; "
                             "default: its row-0 mask pixels are weighted "
                             "by 1/(1+(d/R)^2) — d the distance to the "
                             "object prompt's Step-3a box center, R the "
                             "mask's median distance — normalized to a max "
                             "of 1 and cut below their median, i.e. drawn "
                             "from the pixels within R of it, spread over "
                             "all of them and denser near the object; the "
                             "object prompt and the RoMAv2 modes are "
                             "unaffected)")
    parser.add_argument("--device", default=None, choices=["cuda", "cpu"],
                        help="device (Step 3b; default: auto)")
    parser.add_argument("--with-optical-flow-mask", action="store_true",
                        help="rescue the manipulator mask with optical flow: "
                             "run the Step-3a' WAFT motion-mask pass (WAFTv2 "
                             "online over each sub-task's opening frames — "
                             "run_step3_motion_masks.py) after Step 3a and "
                             "let Step 3b union each sub-task's motion mask "
                             "into the manipulator prompt's row-0 SAM mask "
                             "(default: off)")
    parser.add_argument("--motion-threshold", type=float,
                        default=DEFAULT_MOTION_THRESHOLD,
                        help="flow-magnitude (pixel displacement) threshold "
                             "of the moving-pixel masks (Step 3a', default: "
                             "%(default)s)")
    parser.add_argument("--motion-ratio", type=float,
                        default=DEFAULT_MOTION_RATIO,
                        help="moving-pixel fraction of the frame above which "
                             "a motion mask is significant (Step 3a' "
                             "early-stops on it; default: %(default)s)")
    parser.add_argument("--visualize", action="store_true",
                        help="render every Step-3b visualization: the "
                             "per-prompt viz.png (key-frames with the masks, "
                             "the boxes and the keypoints — incl. the "
                             "manipulator's near-object marks), and, with "
                             "--with-optical-flow-mask, the manipulator's "
                             "SAM ∪ motion union_mask.png (the pixels the "
                             "flow rescue added). Step 3a' writes only its "
                             "motion_rle.json either way (default: off — "
                             "nothing rendered)")
    parser.add_argument("--skip-extract", action="store_true",
                        help="do not run Step 1: reuse the key-frames already "
                             "on disk under the output root "
                             "(<out-dir>/ep{ep:03d}/subtask_{k:02d}/"
                             "sampling_points/key_frames/<camera>/); every "
                             "selected episode must carry its Step-1 "
                             "subtask.json (missing -> error; implied by "
                             "--skip-3a)")
    parser.add_argument("--skip-done", action="store_true",
                        help="skip (episode, sub-task, camera) triples whose "
                             "Step-3a JSON exists, and sub-tasks whose "
                             "Step-3b output already exists")
    parser.add_argument("--skip-3a", action="store_true",
                        help="do not run Step 3a: reuse the detections JSON "
                             "of a previous run — the JSON must exist for "
                             "every selected (episode, camera) with "
                             "key-frames on disk (missing -> error); "
                             "implies --skip-extract (the detections were "
                             "made from the key-frames on disk)")
    parser.add_argument("--refine-detections", action="store_true",
                        help="hard-filter the raw RexOmni predictions in "
                             "Step 3a: duplicate boxes of one instance "
                             "merge into their union, and a side-named "
                             "prompt keeps only the box on its side (Step "
                             "3a default: off — every raw box is kept)")
    parser.add_argument("--rexomni-env", default=REXOMNI_ENV_DIR,
                        help=f"RexOmni environment dir, relative to the repo "
                             f"root (default: {REXOMNI_ENV_DIR})")
    parser.add_argument("--rexomni-python", default=None,
                        help="explicit .venv-rexomni python executable "
                             "(overrides --rexomni-env)")
    return parser.parse_args(argv)


def _out_root(args) -> Path:
    """The episodes root the whole pipeline reads and writes: --out-dir,
    else the default <data-root>/episodes (see utils.astribot_paths)."""
    return ap.episodes_root(args.data_root, args.out_dir)


def _episodes_missing_labels(args) -> list[int]:
    """Selected episodes whose Step-1 subtask.json is missing. Under
    --skip-extract nothing re-extracts, and Step 3a prompts every segment by
    its ground-truth label from that file's ``subtask_labels`` list, so
    every selected episode must carry it. [] when all do."""
    root = _out_root(args)
    try:
        eps = select_episodes(root, args.episode_idxes, args.max_episodes)
    except FileNotFoundError as e:
        sys.exit(f"--skip-extract: {e} — drop --skip-extract to extract "
                 f"the key-frames of the requested episodes")
    return [e for e in eps if not ap.subtask_json(root, e).is_file()]


def _missing_detections(args) -> dict[int, list[str]]:
    """{episode: [camera, ...]} of the selected (episode, camera) pairs
    without a single Step-3a detections JSON
    (…/subtask_{k:02d}/sampling_points/detections/<camera>.json). With
    --skip-3a the sub-steps can only reuse existing detections, so every
    selected camera with key-frames on disk must have run Step 3a before —
    the same (episode, camera) grid Step 3a would detect (--camera-idxes
    resolved to key-frame subdir names; the cameras on disk when the dataset
    metadata cannot be opened). {} when every expected JSON exists, or when
    there are no key-frames to judge (3b then reports the missing key-frames
    itself)."""
    root = _out_root(args)
    if not root.is_dir():
        return {}
    eps = select_episodes(root, args.episode_idxes, args.max_episodes)
    want = _dataset_camera_subdirs(args)  # None -> metadata unavailable
    missing: dict[int, list[str]] = {}
    for e in eps:
        subdirs = camera_subdirs(root, e)
        cams = [c for c in (want if want is not None else subdirs)
                if c in subdirs and "depth" not in c]
        detections = [ap.detections_dir(root, e, k)
                      for k in ap.discover_subtasks(root, e)]
        got = {c for c in cams
               if any((d / f"{c}.json").is_file() for d in detections)}
        if miss := [c for c in cams if c not in got]:
            missing[e] = miss
    return missing


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
    # --use-inferred-splits must reach BOTH runs. detect_subtask resolves
    # the per-segment labels over one segmentation — the ground-truth splits
    # unless the flag asks for the inferred ones — and key_frames lays the
    # subtask_XX directories out over the reading modes' segmentation; if
    # only one run gets the flag, the two disagree and every label lands on
    # another sub-task's directory (Step 3a would prompt segment k with the
    # wrong object/manipulator). One invocation, one segmentation.
    if args.use_inferred_splits:
        cmd += ["--use-inferred-splits"]
    # every mode works in the one episodes root the whole pipeline shares:
    # detect_subtask writes the episode's subtask.json there (Step 3a reads
    # the labels from it), the reading modes take the splits from it
    cmd += ["--out-dir", str(_out_root(args))]
    return cmd


def _build_3a_cmd(args, repo_root: Path) -> list[str]:
    """Step 3a command: RexOmni detections under .venv-rexomni."""
    cmd = [_rexomni_python(args, repo_root),
           str(repo_root / _STEP_3A),
           "--data-root", args.data_root]
    if args.repo_id:
        cmd += ["--repo-id", args.repo_id]
    cam_keys = _dataset_camera_subdirs(args)
    if cam_keys:
        cmd += ["--camera-keys", *cam_keys]
    if args.episode_idxes is not None:
        cmd += ["--episode-idxes", *(str(e) for e in args.episode_idxes)]
    if args.max_episodes is not None:
        cmd += ["--max-episodes", str(args.max_episodes)]
    cmd += ["--out-dir", str(_out_root(args))]
    cmd += ["--max-keyframes", str(args.max_keyframes)]
    if args.refine_detections:
        cmd += ["--refine-detections"]
    if args.skip_done:
        cmd += ["--skip-done"]
    return cmd


def _build_motion_masks_cmd(args, repo_root: Path) -> list[str]:
    """Step 3a' command: WAFT motion masks of the sub-task starts (main
    env, frames decoded online from the dataset). Runs after Step 3a
    against the same (episode, camera) grid — every camera of the
    --camera-idxes with a Step-3a detections JSON — and writes each
    significant mask into the (sub-task, camera) init_points folder Step 3b
    reads under --with-optical-flow-mask."""
    cmd = [sys.executable,
           str(repo_root / _STEP_3AP),
           "--repo-id", args.repo_id,
           "--data-root", args.data_root,
           "--camera-idxes", *(str(c) for c in args.camera_idxes)]
    if args.episode_idxes is not None:
        cmd += ["--episode-idxes", *(str(e) for e in args.episode_idxes)]
    if args.max_episodes is not None:
        cmd += ["--max-episodes", str(args.max_episodes)]
    cmd += ["--out-dir", str(_out_root(args)),
            "--max-keyframes", str(args.max_keyframes),
            "--motion-threshold", str(args.motion_threshold),
            "--motion-ratio", str(args.motion_ratio)]
    if args.skip_done:
        cmd += ["--skip-done"]
    return cmd


def _build_3b_cmd(args, repo_root: Path) -> list[str]:
    """Step 3b command: SAM3 masks + RoMAv2 keypoints in the main env."""
    cmd = [sys.executable,
           str(repo_root / _STEP_3B),
           "--data-root", args.data_root]
    cam_keys = _dataset_camera_subdirs(args)
    if cam_keys:
        cmd += ["--camera-keys", *cam_keys]
    if args.episode_idxes is not None:
        cmd += ["--episode-idxes", *(str(e) for e in args.episode_idxes)]
    if args.max_episodes is not None:
        cmd += ["--max-episodes", str(args.max_episodes)]
    cmd += ["--out-dir", str(_out_root(args))]
    cmd += ["--max-keyframes", str(args.max_keyframes),
            "--object-top-k", str(args.object_top_k),
            "--manipulator-top-k", str(args.manipulator_top_k),
            "--bbox-scale", str(args.bbox_scale),
            "--num-corresp", str(args.num_corresp),
            "--strategy", args.strategy,
            "--sampling-mode", args.sampling_mode]
    if args.with_optical_flow_mask:
        cmd += ["--with-optical-flow-mask"]
    if args.no_manipulator_near_object:
        cmd += ["--no-manipulator-near-object"]
    if args.visualize:
        cmd += ["--visualize"]
    if args.device:
        cmd += ["--device", args.device]
    if args.skip_done:
        cmd += ["--skip-done"]
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
    # --skip-3a reuses the detections JSON, which Step 3a produced from the
    # key-frames on disk — re-extracting them under the reused detections
    # would be pointless (and could drift from them), so --skip-3a implies
    # --skip-extract.
    if args.skip_3a and not args.skip_extract:
        print("--skip-3a implies --skip-extract: reusing the key-frames "
              "on disk", flush=True)
        args.skip_extract = True
    repo_root = Path(__file__).resolve().parents[2]
    if not args.skip_extract:
        _run(_build_extract_cmd(args, repo_root, "detect_subtask"),
             "Step 1a — sub-task split detection (no videos)")
        _run(_build_extract_cmd(args, repo_root, "key_frames"),
             "Step 1b — key-frames saved to disk")
        root = _out_root(args)
        print(f"key-frames on disk: {len(discover_episodes(root))} episode(s) "
              f"under {root}", flush=True)
    elif not (root := _out_root(args)).is_dir():
        sys.exit(f"--skip-extract: no episodes under {root} — drop "
                 f"--skip-extract (or --skip-3a, which implies it) to run "
                 f"Step 1 (detect_subtask + key_frames)")
    elif missing := _episodes_missing_labels(args):
        sys.exit(f"--skip-extract: subtask.json missing for episode(s) "
                 f"{missing} under {_out_root(args)} — drop --skip-extract "
                 f"(or --skip-3a, which implies it) to run Step 1 "
                 f"(detect_subtask writes it)")
    if not args.skip_3a:
        _run(_build_3a_cmd(args, repo_root),
             "Step 3a — RexOmni detections (saved key-frames)")
    elif missing := _missing_detections(args):
        shown = ", ".join(f"ep{e}: {sorted(cams)}"
                          for e, cams in sorted(missing.items()))
        sys.exit(f"--skip-3a: Step-3a detections missing for {shown} under "
                 f"{_out_root(args)} — run Step 3a first (or drop "
                 f"--skip-3a to run it now)")
    if args.with_optical_flow_mask:
        # Step 3a' reads the detections JSON of the same grid (works under
        # --skip-3a too) and writes the motion masks Step 3b unions.
        _run(_build_motion_masks_cmd(args, repo_root),
             "Step 3a' — WAFT motion masks (sub-task starts, online)")
    _run(_build_3b_cmd(args, repo_root),
         "Step 3b — SAM3 masks + RoMAv2 init points")
    print("\nstep 3 done", flush=True)


if __name__ == "__main__":
    main()
