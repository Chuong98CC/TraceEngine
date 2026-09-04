"""Key-frame discovery shared by the Step-3 tools.

Step 3 runs its inference on the key-frames that Step 1 saved to disk
(tools/astribot/extract_frames.py --mode key_frames) instead of decoding the
episode videos online — Step 3 only consumes a handful of frames per
sub-task, so persisting them is cheap, unlike Step 2 which streams every
frame. The on-disk layout is:

    <keyframes_root>/ep{ep:06d}/subtask_{k:02d}/<camera>/frame_<idx:06d>.jpg

with <camera> the camera-subdir name (e.g. cam_head) and <idx> the absolute
dataset frame index. Both Step 3a (run_object_detection.py) and Step 3b
(run_object_init_points.py) must agree on those frames, so the discovery
helpers live here; Step 3a also records the frame indices in its per-episode
detections JSON, which Step 3b reads.

The key_frames mode additionally writes subtask_labels.json per episode (see
SUBTASK_LABELS_FILE): the canonical dataset subtask label of every saved
segment, matched by ground-truth execution order — the canonical ids need
not equal the segment ordinals (the frame-table subtask_index of an episode
can run e.g. [0, 2, 1, 3, 5, 4]). Step 3a reads the file to fetch the
[object, manipulator] prompts of the right sub-task of every segment.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

_EP_RE = re.compile(r"^ep(\d{6})$")
_SUB_RE = re.compile(r"^subtask_(\d+)$")
_FRAME_RE = re.compile(r"^frame_(\d+)\.(?:jpg|jpeg|png)$")

#: dataset subtask annotations (<data-root>/meta/subtasks.csv) — optional
#: per-sub-task prompt columns hand-edited next to the parquet layout
#: (see tools/modify_subtask_parque.py).
SUBTASK_META_FILE = "subtasks.csv"
#: prompt columns of the annotations, in output order: the manipulated
#: object, then the manipulator acting on it.
SUBTASK_PROMPT_COLUMNS = ("object", "manipulator")
#: dataset-native per-episode sub-task order annotation
#: (<data-root>/meta/lerobot_annotations.json): each episode lists its
#: sub-tasks in execution order (start/end times + label text). The
#: ground-truth sub-task order of an episode; extract_frames.py falls back
#: to it when the frame table has no subtask_index column.
SUBTASK_ORDER_FILE = "lerobot_annotations.json"
#: per-episode segment-label map written by extract_frames.py --mode
#: key_frames next to the key-frames: <episode>/subtask_labels.json holds
#: {segment ordinal: canonical subtask label} — the ordinal-to-label match
#: is by ground-truth execution order (the frame-table subtask_index runs
#: need not equal the ordinals), so Step 3a can fetch the [object,
#: manipulator] prompts of the right sub-task of every segment.
SUBTASK_LABELS_FILE = "subtask_labels.json"


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


def subtask_labels_path(root: str | Path, ep_idx: int) -> Path:
    """subtask_labels.json of the episode's key-frames (written by
    extract_frames.py --mode key_frames next to the frames)."""
    return episode_dir(root, ep_idx) / SUBTASK_LABELS_FILE


def load_subtask_labels(root: str | Path,
                        ep_idx: int) -> dict[int, int | None]:
    """{segment ordinal: canonical subtask label} of the episode's saved
    key-frames — the ordinal-to-label match of the key_frames extraction
    (None marks a segment beyond the ground-truth order, i.e. unlabelled).
    Raises FileNotFoundError when the key-frames carry no labels file (an
    extraction that predates it)."""
    path = subtask_labels_path(root, ep_idx)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} missing: the key-frames carry no ground-truth "
            f"sub-task labels — re-run extract_frames.py --mode key_frames "
            f"(or run run_step3_init_points.py without --skip-extract)")
    data = json.loads(path.read_text())
    return {int(k): (int(v) if v is not None else None)
            for k, v in (data.get("segments") or {}).items()}


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
    0..N-1 by sorted name. This is the input mode of run_e2e_init_points.py:
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


def load_subtask_meta(data_root: str | Path) -> dict[int, dict[str, str]]:
    """Subtask annotations of a dataset copy: {subtask_index: row} of
    <data_root>/meta/subtasks.csv, row values keyed by column name.

    The file is the spreadsheet-edited companion of the dataset-native
    meta/subtasks.parquet, so headers and cells are stripped (header names
    often keep spreadsheet spaces, e.g. ``" manipulator"``), unnamed columns
    are dropped, and empty cells are absent from the row dict. Rows whose
    subtask_index does not parse as an int are skipped. Raises
    FileNotFoundError when the dataset has no annotations, ValueError when
    the index column is missing."""
    path = Path(data_root) / "meta" / SUBTASK_META_FILE
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} missing: the dataset must annotate every sub-task "
            f"with object/manipulator prompts for Step 3; export "
            f"meta/subtasks.parquet with tools/modify_subtask_parque.py "
            f"--to-csv, edit the columns, then convert back with --to-parquet")
    with open(path, newline="") as f:
        lines = list(csv.reader(f))
    if not lines:
        return {}
    cols: dict[str, int] = {}
    for i, name in enumerate(lines[0]):
        name = name.strip()
        if name and name not in cols:  # first occurrence of a name wins
            cols[name] = i
    if "subtask_index" not in cols:
        raise ValueError(f"{path}: no subtask_index column "
                         f"(found: {sorted(cols)})")
    meta: dict[int, dict[str, str]] = {}
    for line in lines[1:]:
        cells = [c.strip() for c in line]
        idx_col = cols["subtask_index"]
        if idx_col >= len(cells):
            continue
        try:
            k = int(cells[idx_col])
        except ValueError:
            continue
        if k in meta:
            continue
        row = {name: cells[i] for name, i in cols.items()
               if name != "subtask_index" and i < len(cells) and cells[i]}
        meta[k] = row
    return meta


def subtask_prompts(row: dict[str, str] | None) -> list[str]:
    """RexOmni/SAM3 text prompts of one annotation row: the non-empty
    [object, manipulator] values ([] when the row is missing or carries
    neither)."""
    row = row or {}
    return [row[c] for c in SUBTASK_PROMPT_COLUMNS if row.get(c)]
