"""Step 3a — RexOmni object detection on the saved sub-task key-frames.

Detection stage of pipeline Step 3: for every sub-task of every selected
episode, runs the open-vocabulary detector on the key-frames that Step 1
saved to disk, and writes the raw predictions to **one JSON per episode** —
the input of Step 3b (run_object_init_points.py):

    key-frames on disk (sub-task x camera)
               │
               ▼  RexOmni (open-vocabulary detection)
    ┌──────────────────────────────┐
    │  detections JSON per episode │
    └──────────────────────────────┘

Object prompts are per sub-task: the [object, manipulator] of the sub-task's
row in the dataset's meta/subtasks.csv (episode mode), recorded in the JSON
next to the detections; folder mode (--keyframes-dir, one sub-task of one
camera) takes --text-prompts. RexOmni needs its own environment (Python
3.10 / torch 2.7; checkpoint IDEA-Research/Rex-Omni), so run this script
with .venv-rexomni's python — the sys.path bootstrap below exposes the repo
to it.

Examples
--------
    # Episode mode: all sub-tasks of episode 0, prompts from the dataset's
    # meta/subtasks.csv
    .venv-rexomni/bin/python tools/general_test/pipeline/run_object_detection.py
        --data-root /data/astri_making_coffee_v1 --episode-idxes 0

    # One sub-task's key-frame folder: prompts passed explicitly
    .venv-rexomni/bin/python tools/general_test/pipeline/run_object_detection.py
        --keyframes-dir .../subtask_00/cam_head --episode-idx 0
        --text-prompts "red mug" "left robot gripper"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# .venv-rexomni has no editable install of this project: expose the repo
# root (tools/) and src/ the same way the main env's editable install does.
_REPO_ROOT = Path(__file__).resolve().parents[3]
for _p in (_REPO_ROOT, _REPO_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from PIL import Image
from tqdm import tqdm

from utils.keyframe_utils import (
    cap_keyframes,
    discover_episodes,
    discover_folder_frames,
    discover_subtask_frames,
    keyframe_path,
    keyframes_root,
    load_subtask_meta,
    select_camera,
    select_episodes,
    subtask_prompts,
)

DEFAULT_PROMPTS = ["brown coffee cup", "robot gripper"]
#: RexOmni checkpoint (Hugging Face model id); the default is always used.
REXOMNI_MODEL = "IDEA-Research/Rex-Omni"


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Step 3a: RexOmni detection on the saved sub-task "
                    "key-frames, saved as one JSON per episode "
                    "(run under .venv-rexomni)."
    )
    parser.add_argument("--repo-id", "-id", default=None,
                        help="dataset repo id, recorded in the output JSON "
                             "(informational)")
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
                        help="camera key recorded in the JSON (folder mode "
                             "only; default: the folder name)")
    parser.add_argument("--keyframes-root", default=None,
                        help="root of the key-frames saved by Step 1 "
                             "(default: <data-root>/eps_data/key_frames)")
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
                             "detections land under <out-dir>/detections/")
    parser.add_argument("--text-prompts", nargs="+", default=DEFAULT_PROMPTS,
                        help="object prompts to detect (folder mode only — "
                             "the single sub-task has no dataset "
                             "annotations; episode mode reads "
                             "object/manipulator per sub-task from "
                             "meta/subtasks.csv instead; default: "
                             "%(default)s)")
    parser.add_argument("--max-keyframes", type=int, default=8,
                        help="cap the key-frames per sub-task (evenly spaced); "
                             "None disables the cap (default: %(default)s)")
    parser.add_argument("--skip-done", action="store_true",
                        help="skip episodes whose detections JSON already exists")
    return parser.parse_args(argv)


class SubtaskDetectExtract:
    """Per-sub-task RexOmni detection on the saved key-frames.

    The key-frames are the jpgs written by extract_frames.py --mode
    key_frames: the sub-task segments and the frame indices are read from
    the folder/file names, so no dataset or video access is needed — the
    tool runs on the saved frames alone (it also works when the episode
    videos are not local, e.g. streamed from the internet).
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
            try:
                self.meta = load_subtask_meta(args.data_root)
            except FileNotFoundError as e:
                sys.exit(str(e))
            print(f"sub-task prompts: {len(self.meta)} annotation(s) in "
                  f"{args.data_root}/meta/subtasks.csv")
            self.ep_idxes = select_episodes(self.root, args.episode_idxes,
                                            args.max_episodes)
            self.out_dir = args.out_dir or (args.data_root + "/eps_data")
            print(f"key-frames: {self.root}")
            print(f"episodes on disk: {len(discover_episodes(self.root))} -> "
                  f"{len(self.ep_idxes)} selected")
        self.detections_dir = os.path.join(self.out_dir, "detections")
        os.makedirs(self.detections_dir, exist_ok=True)
        self.model = None

    # --- model ------------------------------------------------------------------

    def _ensure_model(self):
        """RexOmni wrapper, loaded once (the .venv-rexomni environment is
        the one this script runs in, so the import is safe here)."""
        if self.model is None:
            from det_seg_models.rex_omni import RexOmniWrapper
            print(f"Loading Rex-Omni ({REXOMNI_MODEL})...")
            self.model = RexOmniWrapper(
                model_path=REXOMNI_MODEL,
                backend="transformers",
                max_tokens=4096,
                temperature=0.0,
                top_p=0.05,
                top_k=1,
                repetition_penalty=1.05,
            )
        return self.model

    # --- key-frames ---------------------------------------------------------------

    def _load_keyframe(self, cam: str, k: int, t: int) -> Image.Image:
        """PIL RGB image of saved key-frame t (RexOmni takes PIL images)."""
        if self.folder_mode:
            path = self.folder_map.get(t)
            if path is None:
                raise FileNotFoundError(
                    f"key-frame {t} not in {self.folder} "
                    f"(indices {sorted(self.folder_map)})")
        else:
            path = keyframe_path(self.root, self.ep_idx, cam, k, t)
        if not path.is_file():
            raise FileNotFoundError(
                f"key-frame {path} missing: run extract_frames.py "
                f"--mode key_frames for episode {self.ep_idx}")
        return Image.open(path).convert("RGB")

    # --- orchestration -----------------------------------------------------------

    def run(self) -> None:
        if self.folder_mode:
            self._process_folder()
            print(f"\ndone: 1 folder -> {self.detections_dir}")
            return
        print(f"\n{len(self.ep_idxes)} episode(s) selected:")
        for ep in self.ep_idxes:
            print(f"  episode {ep}")
        for ep_idx in tqdm(self.ep_idxes, desc="episodes"):
            self._process_episode(ep_idx)
        print(f"\ndone: {len(self.ep_idxes)} episode(s) -> {self.detections_dir}")

    def _process_folder(self) -> None:
        """The folder of key-frame images = one sub-task (subtask 00) of a
        synthetic episode labelled --episode-idx; same JSON schema as the
        episode mode, plus the input folder for provenance."""
        ep_idx = self.args.episode_idx
        out_path = self._detections_path(ep_idx)
        if self.args.skip_done and out_path.is_file():
            print(f"episode {ep_idx}: skip, detections exist in {out_path}")
            return
        cam = self.args.camera_key or self.folder.name
        keys = cap_keyframes([i for i, _ in self.folder_frames],
                             self.args.max_keyframes)
        if not keys:
            print(f"episode {ep_idx}: skip, no key-frame images in {self.folder}")
            return
        prompts = list(self.args.text_prompts)
        print(f"\nepisode {ep_idx} (folder {self.folder.name}, camera {cam}): "
              f"1 sub-task, {len(keys)} key-frames {keys}, prompts {prompts}")
        subtasks = {
            "0": {
                "segment": [min(keys), max(keys) + 1],
                "keyframes": keys,
                "prompts": prompts,
                "detections": self._detect_segment(cam, 0, keys, prompts),
            }
        }
        data = {
            "episode": int(ep_idx),
            "repo_id": self.args.repo_id,
            "camera_key": cam,
            "keyframes_dir": str(self.folder),
            "subtasks": subtasks,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  saved {out_path}")

    def _detections_path(self, ep_idx: int) -> Path:
        return Path(self.detections_dir) / f"ep{ep_idx:06d}.json"

    def _process_episode(self, ep_idx: int) -> None:
        self.ep_idx = ep_idx
        out_path = self._detections_path(ep_idx)
        if self.args.skip_done and out_path.is_file():
            print(f"episode {ep_idx}: skip, detections exist in {out_path}")
            return
        cam = select_camera(self.root, ep_idx, self.args.camera_keys)
        frames_by_sub = discover_subtask_frames(self.root, ep_idx, cam)
        if not frames_by_sub:
            print(f"episode {ep_idx}: skip, no key-frames saved for camera "
                  f"{cam} (run extract_frames.py --mode key_frames)")
            return
        print(f"\nepisode {ep_idx} (camera {cam}): {len(frames_by_sub)} "
              f"sub-task segment(s)")
        subtasks = {}
        for k in sorted(frames_by_sub):
            keys = cap_keyframes(frames_by_sub[k], self.args.max_keyframes)
            if not keys:
                print(f"  [subtask {k:02d}] skip: no key-frames")
                continue
            prompts = subtask_prompts(self.meta.get(k))
            if not prompts:
                print(f"  [subtask {k:02d}] skip: no object/manipulator "
                      f"row for it in meta/subtasks.csv")
                continue
            print(f"  [subtask {k:02d}] {len(keys)} key-frames {keys}, "
                  f"prompts {prompts}")
            subtasks[str(k)] = {
                "segment": [min(keys), max(keys) + 1],
                "keyframes": keys,
                "prompts": prompts,
                "detections": self._detect_segment(cam, k, keys, prompts),
            }
        data = {
            "episode": int(ep_idx),
            "repo_id": self.args.repo_id,
            "camera_key": cam,
            "subtasks": subtasks,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  saved {out_path}")

    def _detect_segment(self, cam: str, k: int, keys: list[int],
                        prompts: list[str]) -> dict:
        """RexOmni detection over the segment's key-frames: one batched
        call; returns {frame_idx: extracted_predictions} (per-category lists
        of {"type": "box", "coords": [x0, y0, x1, y1]} in absolute pixels,
        no scores)."""
        model = self._ensure_model()
        imgs = [self._load_keyframe(cam, k, t) for t in keys]
        results = model.inference(images=imgs, task="detection",
                                  categories=prompts)
        dets = {}
        for t, res in zip(keys, results):
            preds = res["extracted_predictions"] if res["success"] else {}
            dets[str(t)] = preds
            n = sum(len(boxes) for boxes in preds.values())
            print(f"    key-frame {t}: {n} detection(s)")
        return dets


def main():
    SubtaskDetectExtract(parse_args()).run()


if __name__ == "__main__":
    main()
