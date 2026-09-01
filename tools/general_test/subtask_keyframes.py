"""Key-frame discovery shared by the Step-3 tools.

Step 3 runs its inference on the key-frames that Step 1 saved to disk
(tools/astribot/extract_frames.py --mode key_frames) instead of decoding the
episode videos online — Step 3 only consumes a handful of frames per
sub-task, so persisting them is cheap, unlike Step 2 which streams every
frame. The on-disk layout is:

    <keyframes_root>/ep{ep:06d}/subtask_{k:02d}/<camera>/frame_<idx:06d>.jpg

with <camera> the camera-subdir name (e.g. cam_head) and <idx> the absolute
dataset frame index. Both Step 3a (run_subtask_detections.py) and Step 3b
(run_subtask_init_points.py) must agree on those frames, so the discovery
helpers live here; Step 3a also records the frame indices in its per-episode
detections JSON, which Step 3b reads (the helpers only run again under
--no-rexomni).
"""

from __future__ import annotations

import re
from pathlib import Path

_EP_RE = re.compile(r"^ep(\d{6})$")
_SUB_RE = re.compile(r"^subtask_(\d+)$")
_FRAME_RE = re.compile(r"^frame_(\d+)\.(?:jpg|jpeg|png)$")


def keyframes_root(data_root: str) -> Path:
    """Default key-frames root: the key_frames/ folder of extract_frames.py's
    default output root (<data-root>/eps_data)."""
    return Path(data_root) / "eps_data" / "key_frames"


def episode_dir(root: str | Path, ep_idx: int) -> Path:
    """ep{ep_idx:06d} under the key-frames root."""
    return Path(root) / f"ep{ep_idx:06d}"


def discover_episodes(root: str | Path) -> list[int]:
    """Episode indices with key-frames on disk, sorted ([] for a missing
    root)."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(int(m.group(1)) for p in root.iterdir()
                  if p.is_dir() and (m := _EP_RE.match(p.name)))


def camera_subdirs(root: str | Path, ep_idx: int) -> list[str]:
    """Camera-subdir names present in the episode's key-frame layout
    (collected from its subtask folders), sorted."""
    ep = episode_dir(root, ep_idx)
    if not ep.is_dir():
        return []
    names = set()
    for sub in sorted(ep.iterdir()):
        if _SUB_RE.match(sub.name) and sub.is_dir():
            for cam in sub.iterdir():
                if cam.is_dir():
                    names.add(cam.name)
    return sorted(names)


def discover_subtask_frames(root: str | Path, ep_idx: int,
                            cam_subdir: str) -> dict[int, list[int]]:
    """{subtask_k: sorted absolute frame indices} of the saved key-frames of
    one camera ({} when the episode or camera has none)."""
    ep = episode_dir(root, ep_idx)
    out: dict[int, list[int]] = {}
    if not ep.is_dir():
        return out
    for sub in sorted(ep.iterdir()):
        m = _SUB_RE.match(sub.name)
        if not m or not sub.is_dir():
            continue
        cam = sub / cam_subdir
        if not cam.is_dir():
            continue
        frames = sorted(int(fm.group(1)) for f in cam.iterdir()
                        if f.is_file() and (fm := _FRAME_RE.match(f.name)))
        if frames:
            out[int(m.group(1))] = frames
    return out


def keyframe_path(root: str | Path, ep_idx: int, cam_subdir: str,
                  subtask_k: int, frame_idx: int) -> Path:
    """Path of a saved key-frame jpg (as written by extract_frames.py
    --mode key_frames)."""
    return (episode_dir(root, ep_idx) / f"subtask_{subtask_k:02d}"
            / cam_subdir / f"frame_{frame_idx:06d}.jpg")


def cap_keyframes(keys: list[int], max_keyframes: int | None) -> list[int]:
    """Evenly spaced subset of the key-frames when they exceed the cap
    (None disables the cap)."""
    keys = list(keys)
    if not max_keyframes or len(keys) <= max_keyframes:
        return keys
    idx = sorted({round(i * (len(keys) - 1) / (max_keyframes - 1))
                  for i in range(max_keyframes)})
    return [keys[i] for i in idx]


def select_episodes(root: str | Path,
                    episode_idxes: list[int] | None = None,
                    max_episodes: int | None = None) -> list[int]:
    """Episodes to process: the requested indices (validated against the
    key-frames on disk), or all discovered ones, capped by max_episodes."""
    eps = discover_episodes(root)
    if episode_idxes is not None:
        missing = [e for e in episode_idxes if e not in eps]
        if missing:
            raise FileNotFoundError(
                f"episode(s) {missing} have no key-frames under {root}")
        eps = [e for e in eps if e in episode_idxes]
    if max_episodes is not None:
        eps = eps[: max_episodes]
    return eps


def select_camera(root: str | Path, ep_idx: int,
                  camera_keys: list[str] | None = None) -> str:
    """The episode's camera subdir: the first requested one present on disk,
    else the first non-depth camera saved."""
    subdirs = camera_subdirs(root, ep_idx)
    if not subdirs:
        raise FileNotFoundError(
            f"episode {ep_idx}: no key-frame folders under "
            f"{episode_dir(root, ep_idx)}")
    if camera_keys:
        for c in camera_keys:
            if c in subdirs:
                return c
        raise ValueError(
            f"--camera-keys {camera_keys} not among the saved cameras "
            f"{subdirs} (key-frames root {root})")
    return next(c for c in subdirs if "depth" not in c)


def discover_folder_frames(folder: str | Path) -> list[tuple[int, Path]]:
    """Sorted (frame_idx, path) pairs of the key-frame images in an
    arbitrary folder (one sub-task of one camera, e.g. a cam_head/ folder of
    the episode layout). Indices come from ``frame_<digits>.<ext>`` file
    names when every image matches; otherwise the files are enumerated
    0..N-1 by sorted name. This is the input mode of run_folder_step3.py:
    the folder plays the role of <episode>/subtask_00/<camera>, and both
    Step 3a and Step 3b resolve the same indices from it."""
    folder = Path(folder)
    files = sorted(p for p in folder.iterdir()
                   if p.is_file() and p.suffix.lower()
                   in (".jpg", ".jpeg", ".png"))
    parsed = [_FRAME_RE.match(p.name) for p in files]
    if files and all(parsed):
        return [(int(m.group(1)), p) for m, p in zip(parsed, files)]
    return list(enumerate(files))
