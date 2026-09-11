"""The Astribot output layout, in one place.

Every produced and consumed artifact lives under

    <data_root>/episodes/<ep{idx:03d}>/<subtask_{k:02d}>/<task>/<camera>/

with the exception of the two episode-level Step-1 files, which sit directly
in the episode dir (``subtask.json`` and ``split_graph.png``).  No module
outside this one should build those paths by hand.
"""

import re
from pathlib import Path
from typing import Optional

#: Root folder name under <data_root> (overridable per tool via --out-dir).
EPISODES_DIR = "episodes"

#: Per-sub-task task dirs.
FRAMES = "frames"
VIDEOS = "videos"
DEPTH_POSE = "depth_pose"
SAMPLING_POINTS = "sampling_points"
TRACES = "traces"
VISUALIZATION = "visualization"
TASKS = (FRAMES, VIDEOS, DEPTH_POSE, SAMPLING_POINTS, TRACES, VISUALIZATION)

#: Sub-artifacts of the sampling_points task.
KEY_FRAMES = "key_frames"
DETECTIONS = "detections"
INIT_POINTS = "init_points"

#: Episode-level Step-1 files.
SUBTASK_JSON = "subtask.json"
SPLIT_GRAPH = "split_graph.png"

_EP_RE = re.compile(r"^ep(\d{3})$")
_SUB_RE = re.compile(r"^subtask_(\d{2})$")


def episode_name(ep_idx: int) -> str:
    return f"ep{int(ep_idx):03d}"


def subtask_name(subtask_k: int) -> str:
    return f"subtask_{int(subtask_k):02d}"


def frame_stem(idx: int) -> str:
    """Per-frame file stem inside frames/ (Step-1 media, not depth_pose)."""
    return f"frame_{int(idx):06d}"


def parse_episode(name: str) -> Optional[int]:
    m = _EP_RE.match(name)
    return int(m.group(1)) if m else None


def parse_subtask(name: str) -> Optional[int]:
    m = _SUB_RE.match(name)
    return int(m.group(1)) if m else None


def episodes_root(data_root: str, out_dir: Optional[str] = None) -> Path:
    """<data_root>/episodes, or the explicit --out-dir override."""
    return Path(out_dir) if out_dir else Path(data_root) / EPISODES_DIR


def episode_dir(root, ep_idx: int) -> Path:
    return Path(root) / episode_name(ep_idx)


def subtask_json(root, ep_idx: int) -> Path:
    """Step 1's merged splits + labels file of one episode."""
    return episode_dir(root, ep_idx) / SUBTASK_JSON


def split_graph(root, ep_idx: int) -> Path:
    """Step 1's gripper plot of one episode."""
    return episode_dir(root, ep_idx) / SPLIT_GRAPH


def subtask_dir(root, ep_idx: int, subtask_k: int) -> Path:
    return episode_dir(root, ep_idx) / subtask_name(subtask_k)


def task_dir(root, ep_idx: int, subtask_k: int, task: str,
             camera: Optional[str] = None) -> Path:
    path = subtask_dir(root, ep_idx, subtask_k) / task
    return path / camera if camera else path


def camera_dir(root, ep_idx: int, subtask_k: int, task: str,
               camera: str) -> Path:
    return task_dir(root, ep_idx, subtask_k, task, camera)


# --- task sugar (call sites read better than task_dir(..., DEPTH_POSE)) -----

def depth_pose_dir(root, ep_idx: int, subtask_k: int, camera: str) -> Path:
    return camera_dir(root, ep_idx, subtask_k, DEPTH_POSE, camera)


def frames_dir(root, ep_idx: int, subtask_k: int,
               camera: Optional[str] = None) -> Path:
    return task_dir(root, ep_idx, subtask_k, FRAMES, camera)


def key_frames_dir(root, ep_idx: int, subtask_k: int,
                   camera: Optional[str] = None) -> Path:
    path = task_dir(root, ep_idx, subtask_k, SAMPLING_POINTS) / KEY_FRAMES
    return path / camera if camera else path


def videos_dir(root, ep_idx: int, subtask_k: int,
               camera: Optional[str] = None) -> Path:
    return task_dir(root, ep_idx, subtask_k, VIDEOS, camera)


def detections_dir(root, ep_idx: int, subtask_k: int) -> Path:
    return task_dir(root, ep_idx, subtask_k, SAMPLING_POINTS) / DETECTIONS


def init_points_dir(root, ep_idx: int, subtask_k: int,
                    camera: Optional[str] = None) -> Path:
    path = task_dir(root, ep_idx, subtask_k, SAMPLING_POINTS) / INIT_POINTS
    return path / camera if camera else path


def traces_dir(root, ep_idx: int, subtask_k: int,
               camera: Optional[str] = None) -> Path:
    return task_dir(root, ep_idx, subtask_k, TRACES, camera)


def visualization_dir(root, ep_idx: int, subtask_k: int,
                      camera: Optional[str] = None) -> Path:
    return task_dir(root, ep_idx, subtask_k, VISUALIZATION, camera)


# --- discovery -------------------------------------------------------------

def discover_episodes(root) -> list[int]:
    """Episode indices present on disk, sorted. An episode counts when its
    dir exists and carries a subtask_* dir or subtask.json (so a partially
    processed episode stays discoverable)."""
    root = Path(root)
    if not root.is_dir():
        return []
    out = []
    for p in root.iterdir():
        ep_idx = parse_episode(p.name) if p.is_dir() else None
        if ep_idx is None:
            continue
        if (p / SUBTASK_JSON).is_file() or any(
                q.is_dir() and parse_subtask(q.name) is not None
                for q in p.iterdir()):
            out.append(ep_idx)
    return sorted(out)


def discover_subtasks(root, ep_idx: int) -> list[int]:
    ep = episode_dir(root, ep_idx)
    if not ep.is_dir():
        return []
    return sorted(
        k for k in (parse_subtask(p.name) for p in ep.iterdir() if p.is_dir())
        if k is not None
    )


def discover_cameras(root, ep_idx: int, subtask_k: int, task: str) -> list[str]:
    """Camera subdir names of one (episode, subtask, task), sorted."""
    path = task_dir(root, ep_idx, subtask_k, task)
    if not path.is_dir():
        return []
    return sorted(p.name for p in path.iterdir() if p.is_dir())
